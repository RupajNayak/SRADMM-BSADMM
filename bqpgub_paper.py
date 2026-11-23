#!/usr/bin/env python3
"""
The paper's methods for BQP-GUB with VARIABLE group sizes and dense or sparse R, plus baselines.

    maximize f(x) = p'x + x'Rx   s.t.  sum_{j in C_k} x_j = 1 (k = 1..m),  x binary

Methods (paper settings; see the manuscript):
  aadmm   ADMM for the shifted relaxation (delta = lambda_N + eps), CG on the null space, fixed penalty
          -> relaxed point -> certified upper bound U -> group-wise rounding -> swap local search ("+LS")
  pal     proximal ascent with adaptive local updates -> rounding -> LS
  madmm   MADMM-ALU, binary-set splitting (rho_min = rho_0 = 0.1 ||R||_2) -> LS
Baselines:
  fista   accelerated projected gradient on the SAME relaxation (no ADMM), stopped at AADMM's certified gap
  pal_m1 / pal_m5 / pal_m50   PAL with fixed number of local steps (tests the adaptive rule)
  ls_ms   multistart swap local search from random starts, time-matched to the slowest ADMM-type method
  greedy  greedy construction + LS (informed start, no relaxation)
  ils     iterated local search (perturb + LS), time-matched

Usage
  python bqpgub_paper.py data/ --pattern "*.npz" --out paper_results.csv --time-limit 600 --resume
  python bqpgub_paper.py data/ --methods aadmm,pal,madmm --threads 1
  python bqpgub_paper.py --summarize paper_results.csv --latex paper_tables.tex

Memory: a dense n = 10,000 matrix is ~0.8 GB; it is loaded once, symmetrised in place (blockwise) and never copied.
Instance formats: see load_instance() (npz / mat / json / txt, same as bqpgub_run.py).
"""
import os, sys
_th = "1"                                          # BLAS threads: 1 = reproducible, comparable timings
for _i, _a in enumerate(sys.argv):
    if _a == "--threads" and _i + 1 < len(sys.argv): _th = sys.argv[_i + 1]
    if _a.startswith("--threads="): _th = _a.split("=", 1)[1]
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _th

import argparse, csv, glob, json, re, time, warnings
__version__ = "v5-match-to"
NPZ_MODE = "ask"          # how to turn npz keys (sizes, perm) into groups: ask | perm | contig | invperm
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, LinearOperator
warnings.filterwarnings("ignore", category=DeprecationWarning)

BLK = 1024


class Deadline:
    def __init__(self, seconds): self.t = time.perf_counter() + (seconds if seconds and seconds > 0 else 1e18)
    def hit(self): return time.perf_counter() > self.t


# ================================================================ groups
class Layout:
    """Variable-size groups; padded (m x smax) views make projections vectorised."""

    def __init__(self, groups, n):
        self.n, self.m = n, len(groups)
        self.sizes = np.array([len(g) for g in groups]); self.smax = int(self.sizes.max())
        self.idx = np.full((self.m, self.smax), -1)
        for k, g in enumerate(groups): self.idx[k, :len(g)] = g
        rows, cols = np.nonzero(self.idx >= 0)
        self.gi = np.empty(n, int); self.gj = np.empty(n, int)
        self.gi[self.idx[rows, cols]] = rows; self.gj[self.idx[rows, cols]] = cols
        self.pos = np.arange(1, self.smax + 1); self.ar = np.arange(self.m)
        self.inside = self.pos[None, :] <= self.sizes[:, None]
        self.invsz = 1.0 / self.sizes

    def pad(self, v, fill):
        V = np.full((self.m, self.smax), fill, dtype=float); V[self.gi, self.gj] = v; return V

    def proj(self, v):                                    # Euclidean projection onto the product of simplices
        V = self.pad(v, -1e30)
        U = -np.sort(-V, axis=1)
        css = np.cumsum(U, axis=1) - 1.0
        r = ((U - css / self.pos > 0) & self.inside).sum(axis=1)
        th = css[self.ar, r - 1] / r
        return np.maximum(V - th[:, None], 0.0)[self.gi, self.gj]

    def onehot(self, w):                                  # projection onto Y: largest entry per group (first on ties)
        a = self.pad(w, -np.inf).argmax(axis=1)
        y = np.zeros(self.n); y[self.idx[self.ar, a]] = 1.0
        return y

    def gmax(self, g): return self.pad(g, -np.inf).max(axis=1)

    def null(self, v):                                    # orthogonal projector onto {Az = 0}: subtract group means
        return v - (np.bincount(self.gi, weights=v, minlength=self.m) * self.invsz)[self.gi]

    def random_binary(self, rng):
        sel = rng.integers(0, self.sizes)                 # same stream as rng.integers(0, s, size=m) for equal sizes
        x = np.zeros(self.n); x[self.idx[self.ar, sel]] = 1.0
        return x


# ================================================================ instance + loading
def _symmetrise_inplace(R):
    n = R.shape[0]; need = False
    sc = max(1.0, float(np.max(np.abs(R[:min(n, 64), :min(n, 64)]))))
    for i in range(0, n, BLK):
        for j in range(i, n, BLK):
            if not np.allclose(R[i:i + BLK, j:j + BLK], R[j:j + BLK, i:i + BLK].T, rtol=1e-12, atol=1e-12 * sc):
                need = True; break
        if need: break
    if need:
        for i in range(0, n, BLK):
            for j in range(i, n, BLK):
                a = 0.5 * (R[i:i + BLK, j:j + BLK] + R[j:j + BLK, i:i + BLK].T)
                R[i:i + BLK, j:j + BLK] = a; R[j:j + BLK, i:i + BLK] = a.T
    return R


class Inst:
    def __init__(self, name, p, R, groups):
        self.name = name
        self.p = np.asarray(p, dtype=float).ravel(); self.n = len(self.p)
        chk = np.sort(np.concatenate([np.asarray(g, int) for g in groups]))
        assert len(chk) == self.n and (chk == np.arange(self.n)).all(), "groups must partition {0..n-1}"
        self.L = Layout([np.asarray(g, int) for g in groups], self.n); self.m = self.L.m
        if sp.issparse(R):
            R = R.tocsr(); R = ((R + R.T) * 0.5).tocsr()
        else:
            R = np.ascontiguousarray(R, dtype=float)
            nz = int(np.count_nonzero(R[:BLK])) / max(1, R[:BLK].size)
            if nz < 0.25:                                      # genuinely sparse data stored densely
                R = sp.csr_matrix(R); R = ((R + R.T) * 0.5).tocsr()
            else:
                R = _symmetrise_inplace(R)
        self.R = R; self.dense = not sp.issparse(R)
        self.diag = np.asarray(R.diagonal()).ravel()
        self._nrm = self._lam = None; self.setup_t = 0.0

    def f(self, x): return float(self.p @ x + x @ (self.R @ x))
    def fd(self, x, d): return self.f(x) + d * (self.m - float(x @ x))
    def gd(self, x, d): return self.p + 2.0 * (self.R @ x) - 2.0 * d * x
    def nnz(self): return int(np.count_nonzero(self.R)) if self.dense else int(self.R.nnz)

    def _rowabs_max(self):
        if self.dense:
            return max(float(np.abs(self.R[i:i + BLK]).sum(axis=1).max()) for i in range(0, self.n, BLK))
        return float(abs(self.R).sum(axis=1).max())

    @property
    def norm2(self):
        if self._nrm is None:
            t0 = time.perf_counter()
            if self.n <= 400:
                Rd = self.R.toarray() if not self.dense else self.R
                self._nrm = float(np.max(np.abs(np.linalg.eigvalsh(Rd))))
            else:
                op = LinearOperator((self.n, self.n), matvec=lambda v: self.R @ v, dtype=float)
                self._nrm = float(abs(eigsh(op, k=1, which="LM", return_eigenvectors=False, tol=1e-8)[0]))
            self.setup_t += time.perf_counter() - t0
        return self._nrm

    @property
    def lam(self):                                             # lambda_N + eps  (shift delta of the paper)
        if self._lam is None:
            t0 = time.perf_counter(); L = self.L
            kap = 1.0 + self._rowabs_max()
            if self.n <= 400:
                Rd = self.R.toarray() if not self.dense else self.R
                same = (L.gi[:, None] == L.gi[None, :])
                Pm = np.eye(self.n) - same * L.invsz[L.gi][:, None]
                lam = float(np.linalg.eigvalsh(Pm @ Rd @ Pm - kap * (np.eye(self.n) - Pm)).max())
                self._lam = lam + 1e-9 * max(1.0, abs(lam))
            else:
                def mv(v):
                    Pv = L.null(v); return L.null(self.R @ Pv) - kap * (v - Pv)
                op = LinearOperator((self.n, self.n), matvec=mv, dtype=float)
                lam = float(eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0])
                self._lam = lam + 1e-7 * max(1.0, abs(lam))
            self.setup_t += time.perf_counter() - t0
        return self._lam


def _pick(d, names):
    for k in names:
        if k in d: return d[k]
    raise KeyError(f"none of {names} found; variables = {_describe(d)}")


def _describe(d):
    out = []
    for k, v in d.items():
        a = v if isinstance(v, list) else None
        shp = f"list[{len(a)}]" if a is not None else (tuple(np.shape(v)) if not sp.issparse(v) else f"sparse{v.shape}")
        out.append(f"{k}{shp}")
    return ", ".join(out)


def _is_partition(groups, n):
    if not groups: return False
    allv = np.sort(np.concatenate([np.asarray(g, int).ravel() for g in groups]))
    return len(allv) == n and (allv == np.arange(n)).all()


def groups_from_vars(vars_, n, m=None):
    """Find the group structure in a dict of variables, whatever its name/encoding:
       list of index lists | 0/1 matrix (rows or columns = groups) | label vector | vector of group sizes."""
    skip = {"n", "m", "p", "R", "Q", "quad", "c", "b", "linear", "e"}
    cands = []
    # (0) numbered variables C1, C2, ..., Cm (one index list per group)
    fam = {}
    for k in vars_:
        mm = re.fullmatch(r"([A-Za-z_]+)(\d+)", k)
        if mm and k not in skip: fam.setdefault(mm.group(1), []).append((int(mm.group(2)), k))
    for pref, lst in fam.items():
        lst.sort()
        if len(lst) >= 1 and [i for i, _ in lst] == list(range(1, len(lst) + 1)):
            arrs = [np.asarray(vars_[k], dtype=float).ravel() for _, k in lst]
            base = min(int(a.min()) for a in arrs if a.size)
            gs = [(a - base).astype(int) for a in arrs]
            if _is_partition(gs, n): cands.append((0, pref + "1..", gs))
    # (0b) sizes + perm (npz): meaning set by NPZ_MODE
    if "sizes" in vars_ and "perm" in vars_ and not cands:
        sz = np.asarray(vars_["sizes"]).ravel().astype(int); pm = np.asarray(vars_["perm"]).ravel().astype(int)
        if sz.sum() == n and pm.size == n:
            pm = pm - int(pm.min())
            if np.array_equal(pm, np.arange(n)): NPZ_MODE_EFF = "contig"          # identity perm: no ambiguity
            else: NPZ_MODE_EFF = NPZ_MODE
            if NPZ_MODE_EFF == "ask":
                raise ValueError("file has 'sizes' and 'perm'. Their meaning must be set explicitly: "
                                 "--npz-groups perm | contig | invperm  (see the message in the chat). Variables: " + _describe(vars_))
            e = np.cumsum(np.r_[0, sz]); blocks = [np.arange(e[r], e[r + 1]) for r in range(len(sz))]
            inv = np.argsort(pm)
            gs = {"contig": blocks, "perm": [pm[b] for b in blocks], "invperm": [inv[b] for b in blocks]}[NPZ_MODE_EFF]
            if _is_partition(gs, n): cands.append((0, "sizes+perm:" + NPZ_MODE_EFF, gs))
    for k, v in vars_.items():
        if k in skip: continue
        if isinstance(v, list):                                                  # cell array of index lists
            arrs = [np.asarray(a).ravel() for a in v]
            if arrs and all(np.issubdtype(a.dtype, np.number) for a in arrs):
                base = min(int(a.min()) for a in arrs if a.size)
                gs = [(a - base).astype(int) for a in arrs]
                if _is_partition(gs, n): cands.append((0, k, gs))
            continue
        a = v.toarray() if sp.issparse(v) else np.asarray(v)
        if a.dtype == object and a.size > 0:                                     # MATLAB cell loaded as object array
            try:
                arrs = [np.asarray(x).ravel() for x in a.ravel()]; base = min(int(x.min()) for x in arrs if x.size)
                gs = [(x - base).astype(int) for x in arrs]
                if _is_partition(gs, n): cands.append((0, k, gs))
            except Exception: pass
            continue
        if a.ndim == 2 and a.shape[0] > 1 and a.shape[1] > 1 and set(np.unique(a)) <= {0, 1}:   # 0/1 GUB matrix
            for M in (a, a.T):
                if M.shape[1] == n and (M.sum(axis=0) == 1).all():
                    cands.append((1, k, [np.nonzero(M[r])[0] for r in range(M.shape[0])])); break
            continue
        a = a.ravel()
        if a.size == n and np.allclose(a, np.round(a)) and a.size > 0:           # label vector
            u = np.unique(a)
            if 1 < len(u) < n or (len(u) == n and m == n):
                if m is None or len(u) == m: cands.append((2, k, [np.nonzero(a == x)[0] for x in u]))
        if m is not None and a.size == m and np.allclose(a, np.round(a)) and int(a.sum()) == n and (a >= 1).all():  # sizes
            e = np.cumsum(np.r_[0, a.astype(int)]); cands.append((3, k, [np.arange(e[r], e[r + 1]) for r in range(m)]))
    if not cands:
        raise ValueError("cannot find the group structure. Variables in the file: " + _describe(vars_))
    cands.sort(key=lambda t: t[0]); pr, name, gs = cands[0]
    if m is not None and len(gs) != m:
        raise ValueError(f"group variable '{name}' gives {len(gs)} groups but m = {m}. Variables: " + _describe(vars_))
    return gs


def _inst_from_vars(name, d):
    p = np.asarray(_pick(d, ("p", "c", "b", "linear")), dtype=float).ravel(); n = len(p)
    R = _pick(d, ("R", "Q", "quad"))
    if not sp.issparse(R):
        R = np.asarray(R, dtype=float)
        if R.ndim == 1 and R.size == n * n: R = R.reshape(n, n)
    m = int(np.asarray(d["m"]).ravel()[0]) if "m" in d else None
    return Inst(name, p, R, groups_from_vars(d, n, m))


_HEAD = re.compile(r"(?m)^[ \t]*([A-Za-z_]\w*)[ \t]*=[ \t]*")


def parse_matlab_text(path):
    """MATLAB-style text:  n = 20 / m = 5 / p = [a;b;...] / R = [r11 r12 ...; ...] / (groups in some form)."""
    txt = open(path).read()
    heads = [(mm.group(1), mm.end(), mm.start()) for mm in _HEAD.finditer(txt)]
    d = {}
    for idx, (name, start, _) in enumerate(heads):
        end = heads[idx + 1][2] if idx + 1 < len(heads) else len(txt)
        raw = txt[start:end].strip().rstrip(";").strip()
        if raw[:1] == "[":
            inner = raw[1:raw.rindex("]")]
            d[name] = np.array(inner.replace(";", " ").replace(",", " ").split(), dtype=float)
            nrow = len([r for r in re.split(r";|\n", inner) if r.strip()])
            if nrow > 1 and d[name].size % nrow == 0: d[name] = d[name].reshape(nrow, -1)
        elif raw[:1] == "{":
            d[name] = [np.array(b.replace(";", " ").replace(",", " ").split(), dtype=float)
                       for b in re.findall(r"\[([^\]]*)\]", raw)] or \
                      [np.array(r.replace(",", " ").split(), dtype=float) for r in re.split(r";|\n", raw[1:raw.rindex("}")]) if r.strip()]
        else:
            try: d[name] = float(raw.split()[0])
            except Exception: pass
    for k in ("n", "m"):
        if k in d: d[k] = int(d[k])
    if "R" in d and "n" in d and np.asarray(d["R"]).size == d["n"] ** 2: d["R"] = np.asarray(d["R"]).reshape(d["n"], d["n"])
    return d


def load_instance(path):
    name = os.path.splitext(os.path.basename(path))[0]; ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        d = dict(np.load(path, allow_pickle=True))
        if "R_data" in d:
            d["R"] = sp.csr_matrix((d["R_data"], d["R_indices"], d["R_indptr"]), shape=tuple(d["R_shape"]))
        elif "R" in d and getattr(d["R"], "dtype", None) == object:
            d["R"] = d["R"].item()
        return _inst_from_vars(name, d)
    if ext == ".mat":
        from scipy.io import loadmat
        return _inst_from_vars(name, {k: v for k, v in loadmat(path).items() if not k.startswith("__")})
    if ext == ".json":
        d = json.load(open(path))
        if "R_val" in d: d["R"] = sp.coo_matrix((d["R_val"], (d["R_row"], d["R_col"])), shape=(len(d["p"]),) * 2)
        return _inst_from_vars(name, d)
    if ext in (".txt", ".dat"):
        head = open(path).read(200)
        if re.match(r"\s*n\s*=", head): return _inst_from_vars(name, parse_matlab_text(path))
        toks = open(path).read().split("\n"); n, m = map(int, toks[0].split()[:2])       # older plain format
        lab = np.array(toks[1].split(), dtype=int); p = np.array(toks[2].split(), dtype=float)
        tri = np.array([list(map(float, t.split())) for t in toks[3:] if t.strip()])
        R = sp.coo_matrix((tri[:, 2], (tri[:, 0].astype(int) - 1, tri[:, 1].astype(int) - 1)), shape=(n, n))
        return Inst(name, p, R, [np.nonzero(lab == v)[0] for v in np.unique(lab)])
    raise ValueError(f"unsupported file type: {path}")


# ================================================================ certificate, rounding, local search
def certified_bound(inst, x, delta):
    """For ANY x in P:  max_P f_delta <= f_delta(x) + sum_k max_{j in C_k} g_j - g'x,  g = grad f_delta(x)."""
    g = inst.gd(x, delta)
    return inst.fd(x, delta) + float(inst.L.gmax(g).sum() - g @ x)


def ls_polish(inst, xb, dl=None):
    """Steepest-ascent swap local search inside groups (binary feasible in, binary feasible out)."""
    L, p, R, d = inst.L, inst.p, inst.R, inst.diag
    n = inst.n; sparse = not inst.dense
    Rc = R.tocsc() if sparse else None
    selvar = L.idx[L.ar, L.pad(xb, -np.inf).argmax(axis=1)]           # selected variable per group
    x = xb.copy(); w = np.asarray(R @ x).ravel(); cand = np.arange(n)
    while True:
        cur = selvar[L.gi]
        Rjc = np.asarray(R[cand, cur]).ravel()
        gain = p - p[cur] + 2.0 * (w - w[cur]) + d + d[cur] - 2.0 * Rjc
        gain[cand == cur] = -np.inf
        b = int(np.argmax(gain))
        if gain[b] <= 1e-10 or (dl is not None and dl.hit()): break
        jn, jo = b, int(cur[b])
        if sparse: w += (Rc[:, jn] - Rc[:, jo]).toarray().ravel()
        else: w += R[jn] - R[jo]                                      # symmetric: rows are contiguous
        x[jo], x[jn] = 0.0, 1.0; selvar[L.gi[b]] = jn
    return x


def pipeline_tail(inst, xrelaxed):
    """rounding + LS: returns (f_round, f_ls, seconds)"""
    t0 = time.perf_counter()
    xr = inst.L.onehot(xrelaxed); fr = inst.f(xr)
    xl = ls_polish(inst, xr)
    return fr, inst.f(xl), time.perf_counter() - t0


# ================================================================ AADMM (paper)
def _cg(Kmv, b, z0, tol, maxit, dl):
    z = z0.copy(); r = b - Kmv(z); d = r.copy(); rs = float(r @ r)
    bn = float(np.sqrt(b @ b)) + 1e-300; it = 0
    while it < maxit and np.sqrt(rs) > tol * bn and not dl.hit():
        Kd = Kmv(d); a = rs / float(d @ Kd)
        z += a * d; r -= a * Kd; rs2 = float(r @ r); d = r + (rs2 / rs) * d; rs = rs2; it += 1
    return z, it


def aadmm(inst, time_limit=0, delta=None, rho0=None, eps_abs=1e-7, eps_rel=1e-6, max_iter=2000, cg_tol=1e-10, cg_maxit=1000):
    n, L, R, p = inst.n, inst.L, inst.R, inst.p
    delta = inst.lam if delta is None else delta
    rho = (0.5 * inst.norm2 + 1e-6) if rho0 is None else rho0
    x0 = L.invsz[L.gi].copy(); q = L.null(p + 2.0 * (R @ x0))
    z = np.zeros(n); y = x0.copy(); u = np.zeros(n); yh, uh = y.copy(), u.copy()
    dl = Deadline(time_limit); t0 = time.perf_counter(); cg_total = 0; converged = False; hit = False
    for k in range(max_iter):
        c0 = 2.0 * delta + rho
        b = q + rho * L.null(yh - uh)
        z, its = _cg(lambda v: L.null(c0 * v - 2.0 * (R @ v)), b, z, cg_tol, cg_maxit, dl); cg_total += its
        x = x0 + z
        y_new = np.maximum(x + uh, 0.0); u_new = uh + x - y_new
        r = x - y_new; s = -rho * (y_new - yh)
        rn, sn = float(np.linalg.norm(r)), float(np.linalg.norm(s))
        eps_p = np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y_new))
        eps_d = np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u_new)
        y, u = y_new, u_new; yh, uh = y.copy(), u.copy()
        if rn <= eps_p and sn <= eps_d: converged = True; break
        if dl.hit(): hit = True; break
    t = time.perf_counter() - t0
    xP = L.proj(x)
    return dict(xP=xP, delta=delta, iters=k + 1, converged=converged, hit=hit, time=t, cg=cg_total)


# ================================================================ PAL (paper)
def pal(inst, time_limit=0, rho=None, M0=5, Mmin=2, Mmax=50, max_outer=3000, tol=1e-9, th_low=1e-4, th_high=1e-2, x0=None):
    n, L, R, p = inst.n, inst.L, inst.R, inst.p
    rho = (2.0 * max(inst.lam, 0.0) + 0.1 * inst.norm2 + 1e-8) if rho is None else rho
    Lip = rho + 2.0 * inst.norm2
    x = L.invsz[L.gi].copy() if x0 is None else x0.copy()
    fx = inst.f(x); M = M0; dl = Deadline(time_limit); t0 = time.perf_counter(); inner_total = 0
    ascent_ok = True; hit = False; converged = False
    for k in range(max_outer):
        xk = x; xi = xk; gm = None
        for i in range(M):
            g = p + 2.0 * (R @ xi) - rho * (xi - xk)
            xn = L.proj(xi + g / Lip)
            if i == 0: gm = Lip * float(np.linalg.norm(xn - xk))
            xi = xn
            if dl.hit(): hit = True; break
        inner_total += M; x = xi; f_new = inst.f(x)
        if f_new < fx + 0.5 * rho * float(np.sum((x - xk) ** 2)) - 1e-9 * max(1.0, abs(fx)): ascent_ok = False
        ratio = (f_new - fx) / max(abs(f_new), 1.0); fx = f_new
        if hit: break
        if gm <= tol * max(1.0, np.linalg.norm(p)): converged = True; break
        if ratio < th_low: M = min(2 * M, Mmax)
        elif ratio > th_high: M = max(M // 2, Mmin)
    return dict(x=x, iters=k + 1, inner=inner_total, converged=converged, hit=hit, ascent_ok=ascent_ok, time=time.perf_counter() - t0)


# ================================================================ MADMM-ALU (paper, binary-set splitting)
def _x_update(inst, y, u, rho, x_init, inner, tol, convex, dl):
    L, R, p = inst.L, inst.R, inst.p
    Lip = 2.0 * inst.norm2 + rho; c = y - u; x = x_init.copy(); z = x.copy(); t = 1.0
    for _ in range(inner):
        g = -p - 2.0 * (R @ z) + rho * (z - c)
        xn = L.proj(z - g / Lip); dx = float(np.linalg.norm(xn - x))
        if convex:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)); z = xn + ((t - 1.0) / tn) * (xn - x); t = tn
        else: z = xn
        x = xn
        if dx <= tol * max(1.0, float(np.linalg.norm(x))) or dl.hit(): break
    return x


def madmm_alu(inst, time_limit=0, c_min=0.1, M0=5, Mmin=5, Mmax=50, mu=10.0, tau=1.0, K=None, inner=None,
              eps_abs=1e-6, eps_rel=1e-4, inner_tol=1e-9, seed=0):
    n, L = inst.n, inst.L
    K = K or (300 if n <= 20 else 100); inner = inner or (300 if n <= 20 else 100)
    rho_min = c_min * inst.norm2; rho = rho_min
    rng = np.random.default_rng(seed); y = L.random_binary(rng); u = np.zeros(n); x = y.copy()
    M = M0; best_x, best_v = y.copy(), inst.f(y); c_prev = np.inf
    dl = Deadline(time_limit); t0 = time.perf_counter(); converged = False; hit = False
    for k in range(K):
        convex = rho >= 2.0 * inst.lam; y_old = y.copy(); yl = y.copy()
        for _ in range(M):
            x = _x_update(inst, yl, u, rho, x, inner, inner_tol, convex, dl)
            yl = L.onehot(x + u)
            if dl.hit(): hit = True; break
        y = yl; u = u + x - y; r = x - y; s = -rho * (y - y_old)
        rn, sn = float(np.linalg.norm(r)), float(np.linalg.norm(s))
        for cand in (y, L.onehot(x)):
            v = inst.f(cand)
            if v > best_v: best_v, best_x = v, cand.copy()
        c = rn ** 2 + sn ** 2
        eps_p = np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y))
        eps_d = np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u)
        if hit: break
        if rn <= eps_p and sn <= eps_d: converged = True; break
        rn_ = rho
        if rn > mu * sn: rn_ = max(rho_min, rho * (1.0 + tau))
        elif sn > mu * rn: rn_ = max(rho_min, rho / (1.0 + tau))
        if rn_ != rho: u = u * (rho / rn_); rho = rn_
        M = min(2 * M, Mmax) if c < c_prev else max(int(np.ceil(M / 2)), Mmin); c_prev = c
    return dict(x=best_x, value=best_v, iters=k + 1, converged=converged, hit=hit, time=time.perf_counter() - t0)


# ================================================================ baselines
def fista_relax(inst, delta, target_gap, time_limit=0, max_iter=200000, check=10):
    """Accelerated projected gradient (with function restart) on max_P f_delta; stops when the certified
    relative gap (U-f_delta)/max(1,|U|) <= target_gap."""
    L, n = inst.L, inst.n
    Lip = 2.0 * (inst.norm2 + abs(delta)) + 1e-12
    x = L.invsz[L.gi].copy(); w = x.copy(); t = 1.0; fo = inst.fd(x, delta)
    dl = Deadline(time_limit); t0 = time.perf_counter(); hit = False; gap = np.inf
    for k in range(max_iter):
        g = inst.gd(w, delta); xn = L.proj(w + g / Lip)
        tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        w = xn + ((t - 1.0) / tn) * (xn - x); x, t = xn, tn
        if (k + 1) % check == 0:
            fn = inst.fd(x, delta)
            if fn < fo: w = x.copy(); t = 1.0                          # function-value restart
            fo = fn
            U = certified_bound(inst, x, delta); gap = (U - fn) / max(1.0, abs(U))
            if gap <= target_gap: break
            if dl.hit(): hit = True; break
    return dict(xP=x, iters=k + 1, gap=gap, hit=hit, time=time.perf_counter() - t0)


def ls_multistart(inst, budget, min_starts, seed):
    rng = np.random.default_rng(seed); best = -np.inf; t0 = time.perf_counter(); done = 0
    while done < min_starts or time.perf_counter() - t0 < budget:
        x = ls_polish(inst, inst.L.random_binary(rng)); best = max(best, inst.f(x)); done += 1
    return best, time.perf_counter() - t0, done


def greedy_start(inst, rng):
    L, R, p, d = inst.L, inst.R, inst.p, inst.diag
    x = np.zeros(inst.n); w = np.zeros(inst.n)
    for k in rng.permutation(L.m):
        g = L.idx[k, :L.sizes[k]]
        j = int(g[np.argmax(p[g] + d[g] + 2.0 * w[g])])                # marginal gain of adding j
        x[j] = 1.0
        w += (R[j] if inst.dense else np.asarray(R[j].todense()).ravel())
    return x


def ils(inst, budget, seed, x0=None, frac=0.05):
    L = inst.L; rng = np.random.default_rng(seed); t0 = time.perf_counter()
    x = ls_polish(inst, L.random_binary(rng) if x0 is None else x0); fx = inst.f(x); it = 0
    while time.perf_counter() - t0 < budget:
        y = x.copy(); q = max(2, int(frac * L.m))
        for k in rng.choice(L.m, size=min(q, L.m), replace=False):
            g = L.idx[k, :L.sizes[k]]; y[g] = 0.0; y[int(rng.choice(g))] = 1.0
        y = ls_polish(inst, y); fy = inst.f(y); it += 1
        if fy >= fx: x, fx = y, fy
    return fx, time.perf_counter() - t0, it


# ================================================================ runner
FIELDS = ["name", "n", "m", "nnz", "setup_t", "lamN", "norm2",
          "aadmm_U", "aadmm_frel", "aadmm_fround", "aadmm_fls", "aadmm_t", "aadmm_tls", "aadmm_it", "aadmm_cg", "aadmm_conv", "aadmm_hit",
          "pal_frel", "pal_fround", "pal_fls", "pal_t", "pal_tls", "pal_it", "pal_inner", "pal_conv", "pal_hit", "pal_ascent_ok",
          "madmm_f", "madmm_fls", "madmm_t", "madmm_tls", "madmm_it", "madmm_conv", "madmm_hit",
          "fista_gap", "fista_fls", "fista_t", "fista_it", "fista_hit",
          "pal_m1_fls", "pal_m1_t", "pal_m5_fls", "pal_m5_t", "pal_m50_fls", "pal_m50_t",
          "lsms_f", "lsms_t", "lsms_starts", "greedy_fls", "greedy_t", "ils_f", "ils_t", "ils_it", "budget"]
ALL = ["aadmm", "pal", "madmm", "fista", "pal_fixed", "ls_ms", "greedy", "ils"]


def run_instance(inst, a):
    row = dict(name=inst.name, n=inst.n, m=inst.m, nnz=inst.nnz())
    lam, nrm = inst.lam, inst.norm2
    row.update(lamN=lam, norm2=nrm, setup_t=inst.setup_t)
    meth = set(a.methods.split(","))
    if "all" in meth: meth = set(ALL)
    tl = a.time_limit; tot = {}
    if "aadmm" in meth:
        r = aadmm(inst, tl); xP = r["xP"]; U = certified_bound(inst, xP, r["delta"])
        fr, fl, tls = pipeline_tail(inst, xP)
        row.update(aadmm_U=U, aadmm_frel=inst.f(xP), aadmm_fround=fr, aadmm_fls=fl, aadmm_t=r["time"], aadmm_tls=tls,
                   aadmm_it=r["iters"], aadmm_cg=r["cg"], aadmm_conv=r["converged"], aadmm_hit=r["hit"])
        tot["aadmm"] = r["time"] + tls
        row["_gap"] = (U - inst.fd(xP, r["delta"])) / max(1.0, abs(U))
    if "pal" in meth:
        r = pal(inst, tl); fr, fl, tls = pipeline_tail(inst, r["x"])
        row.update(pal_frel=inst.f(r["x"]), pal_fround=fr, pal_fls=fl, pal_t=r["time"], pal_tls=tls, pal_it=r["iters"],
                   pal_inner=r["inner"], pal_conv=r["converged"], pal_hit=r["hit"], pal_ascent_ok=r["ascent_ok"])
        tot["pal"] = r["time"] + tls
    if "madmm" in meth:
        r = madmm_alu(inst, tl, c_min=a.madmm_cmin, seed=a.seed); t0 = time.perf_counter()
        xl = ls_polish(inst, r["x"]); tls = time.perf_counter() - t0
        row.update(madmm_f=r["value"], madmm_fls=inst.f(xl), madmm_t=r["time"], madmm_tls=tls, madmm_it=r["iters"],
                   madmm_conv=r["converged"], madmm_hit=r["hit"])
        tot["madmm"] = r["time"] + tls
    if "fista" in meth:
        tgt = row.pop("_gap", None) or a.fista_gap
        r = fista_relax(inst, lam, tgt, tl); fr, fl, tls = pipeline_tail(inst, r["xP"])
        row.update(fista_gap=r["gap"], fista_fls=fl, fista_t=r["time"] + tls, fista_it=r["iters"], fista_hit=r["hit"])
    row.pop("_gap", None)
    if "pal_fixed" in meth:
        for M in (1, 5, 50):
            r = pal(inst, tl, M0=M, Mmin=M, Mmax=M); _, fl, tls = pipeline_tail(inst, r["x"])
            row[f"pal_m{M}_fls"] = fl; row[f"pal_m{M}_t"] = r["time"] + tls
    budget = max(tot.values()) if tot else a.budget
    if a.match_to == "aadmm" and "aadmm" in tot: budget = tot["aadmm"]      # give LS-ms / ILS exactly the time AADMM+LS used
    if a.match_cap: budget = min(budget, a.match_cap)       # cap on the matched LS-ms / ILS budget
    row["budget"] = budget
    if "ls_ms" in meth:
        v, t, s = ls_multistart(inst, budget, a.min_starts, a.seed + 1); row.update(lsms_f=v, lsms_t=t, lsms_starts=s)
    if "greedy" in meth:
        t0 = time.perf_counter(); xg = ls_polish(inst, greedy_start(inst, np.random.default_rng(a.seed)))
        row.update(greedy_fls=inst.f(xg), greedy_t=time.perf_counter() - t0)
    if "ils" in meth:
        v, t, it = ils(inst, budget, a.seed + 2); row.update(ils_f=v, ils_t=t, ils_it=it)
    return row


def natural_key(s): return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


# ================================================================ summary
def _num(v):
    try:
        v = float(v); return v if np.isfinite(v) else None
    except (TypeError, ValueError): return None


def summarize(path, latex=None):
    import pandas as pd
    D = pd.read_csv(path); vcols = [c for c in ["aadmm_fls", "pal_fls", "madmm_fls", "fista_fls", "pal_m1_fls", "pal_m5_fls",
                                                "pal_m50_fls", "lsms_f", "greedy_fls", "ils_f"] if c in D]
    D["best"] = D[vcols].max(axis=1)
    if "aadmm_U" in D: D["cert_gap"] = (D.aadmm_U - D.best) / D.aadmm_U.abs() * 100
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
    lab = {"aadmm_fls": "AADMM+LS", "pal_fls": "PAL+LS", "madmm_fls": "MADMM+LS", "fista_fls": "FISTA+LS", "lsms_f": "LS-ms",
           "greedy_fls": "Greedy+LS", "ils_f": "ILS", "pal_m1_fls": "PAL M=1", "pal_m5_fls": "PAL M=5", "pal_m50_fls": "PAL M=50"}
    rows = []
    for n, g in D.groupby("n"):
        r = {"n": n, "#": len(g)}
        for c in vcols:
            if g[c].notna().any(): r["gap " + lab[c]] = ((g.best - g[c]) / g.best.abs() * 100).mean()
        if "cert_gap" in g: r["certified gap"] = g.cert_gap.mean()
        for c in ["aadmm", "pal", "madmm"]:
            if f"{c}_t" in g and g[f"{c}_t"].notna().any(): r["t " + c] = (g[f"{c}_t"] + g.get(f"{c}_tls", 0)).mean()
        rows.append(r)
    T = pd.DataFrame(rows); print(T.round(2).to_string(index=False))
    if latex:
        gcols = [c for c in T.columns if c.startswith("gap ")]
        with open(latex, "w") as f:
            f.write("% mean % below the best known value per dimension; times in seconds\n\\begin{table}[H]\n\\centering\n\\footnotesize\n"
                    "\\setlength{\\tabcolsep}{3pt}\n\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"
                    "\\begin{tabular}{rr" + "r" * (len(gcols) + 1) + "}\n\\toprule\n$n$ & \\# & " +
                    " & ".join(c.replace("gap ", "") for c in gcols) + " & Cert.\\ gap\\\\\n\\midrule\n")
            for _, r in T.iterrows():
                f.write(f"{int(r['n'])} & {int(r['#'])} & " + " & ".join("--" if pd.isna(r[c]) else f"{r[c]:.2f}" for c in gcols)
                        + " & " + ("--" if "certified gap" not in r or pd.isna(r["certified gap"]) else f"{r['certified gap']:.1f}") + " \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}}\n\\caption{Gap (\\%) below the best known value; Cert.\\ gap: 100(U-best)/U.}\n\\end{table}\n")
        print("LaTeX written to", latex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*"); ap.add_argument("--pattern", default="*"); ap.add_argument("--out", default="paper_results.csv")
    ap.add_argument("--methods", default="aadmm,pal,madmm,fista,ls_ms,greedy,ils",
                    help="comma list of: " + ",".join(ALL) + "  (or 'all')")
    ap.add_argument("--time-limit", type=float, default=600.0, help="wall-clock cap per method (s); hits are flagged in the CSV")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--min-starts", type=int, default=10)
    ap.add_argument("--budget", type=float, default=5.0, help="time budget for ls_ms/ils if no ADMM-type method is run")
    ap.add_argument("--match-to", default="slowest", choices=["slowest", "aadmm"], help="time budget of ls_ms/ils: the slowest ADMM-type method (default) or AADMM+LS")
    ap.add_argument("--match-cap", type=float, default=0, help="cap (s) on the time budget given to ls_ms and ils (0 = match the slowest ADMM-type method exactly)")
    ap.add_argument("--madmm-cmin", type=float, default=0.1, help="MADMM-ALU rho_min = rho0 = cmin*||R||_2")
    ap.add_argument("--fista-gap", type=float, default=1e-3); ap.add_argument("--threads", default="1")
    ap.add_argument("--resume", action="store_true"); ap.add_argument("--summarize", metavar="CSV"); ap.add_argument("--latex")
    ap.add_argument("--npz-groups", default="ask", choices=["ask", "perm", "contig", "invperm"], help="meaning of npz keys sizes+perm")
    ap.add_argument("--max-n", type=int, default=0, help="skip instances larger than this (0 = none)")
    a = ap.parse_args()
    global NPZ_MODE
    NPZ_MODE = a.npz_groups
    print(f"bqpgub_paper {__version__}", flush=True)
    if a.summarize: summarize(a.summarize, a.latex); return
    files = []
    for pth in a.paths: files += sorted(glob.glob(os.path.join(pth, a.pattern))) if os.path.isdir(pth) else sorted(glob.glob(pth))
    files = sorted({f for f in files if os.path.isfile(f)}, key=lambda s: natural_key(os.path.basename(s)))
    if not files: sys.exit("no instance files found")
    first = not os.path.exists(a.out)
    if a.resume and not first:
        done = {r["name"] for r in csv.DictReader(open(a.out))}
        files = [f for f in files if os.path.splitext(os.path.basename(f))[0] not in done]; print(f"resume: {len(done)} done, {len(files)} left")
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if first: w.writeheader()
        for path in files:
            try:
                inst = load_instance(path)
                if a.max_n and inst.n > a.max_n: print(f"[skip] {inst.name}: n={inst.n} > --max-n"); continue
                t0 = time.perf_counter(); row = run_instance(inst, a); del inst
            except Exception as ex:
                print(f"[skip] {path}: {ex!r}"); continue
            w.writerow(row); f.flush()
            g = lambda k: row.get(k)
            parts = [f"{lab}={g(k):.6g}" for k, lab in (("aadmm_fls", "AADMM+LS"), ("pal_fls", "PAL+LS"), ("madmm_fls", "MADMM+LS"),
                                                          ("lsms_f", "LS-ms"), ("ils_f", "ILS")) if g(k) is not None]
            print(f"{row['name']:>16} n={row['n']:<6} " + "  ".join(parts) + f"  [{time.perf_counter() - t0:.1f}s]", flush=True)
    summarize(a.out, a.latex)


if __name__ == "__main__":
    main()
