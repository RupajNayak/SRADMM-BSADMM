#!/usr/bin/env python3
"""
Runner for BQP-GUB / CMISP instances in the MATLAB-style text format

    n = 20
    m = 4
    p = [0;1;0;...]
    R = [1 0 0 ...;0 1 1 ...;...]
    C1 = [20 3 4 12]        (1-based indices)
    ...

Methods (all return a binary feasible point after rounding + swap local search):
  sradmm : ADMM on the shifted relaxation (fixed penalty, no extrapolation)
  fista  : accelerated projected gradient on the SAME relaxation
  pal    : proximal ascent with adaptive local updates
  ls10   : multistart swap local search, 10 random starts
  ls_same: multistart LS with the wall-clock time used by sradmm+LS
  exact  : enumeration (small) or HiGHS on the Adams-Sherali linearization

Usage:
  python cmisp_runner.py instances/*.txt --out results.csv [--exact-max-n 100]
"""
import argparse, glob, itertools, re, time, os, sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, eigsh

# ----------------------------------------------------------------- parsing
def load_instance(path):
    txt = open(path).read()
    def scalar(key):
        return int(re.search(rf'^\s*{key}\s*=\s*(\d+)', txt, re.M).group(1))
    def vec(key):
        s = re.search(rf'^\s*{key}\s*=\s*\[(.*?)\]', txt, re.M | re.S).group(1)
        return np.array([float(t) for t in re.split(r'[;,\s]+', s.strip()) if t])
    n, m = scalar('n'), scalar('m')
    p = vec('p')
    s = re.search(r'^\s*R\s*=\s*\[(.*?)\]', txt, re.M | re.S).group(1)
    rows = [[float(t) for t in r.split()] for r in s.split(';') if r.strip()]
    R = np.array(rows)
    groups = []
    for k in range(1, m + 1):
        groups.append(vec(f'C{k}').astype(int) - 1)
    assert p.shape == (n,) and R.shape == (n, n), (p.shape, R.shape, n)
    assert sorted(np.concatenate(groups).tolist()) == list(range(n)), "groups must partition 0..n-1"
    return dict(name=os.path.splitext(os.path.basename(path))[0], n=n, m=m,
                p=p, R=(R + R.T) / 2.0, groups=groups)

class Inst:
    def __init__(self, d):
        self.name, self.n, self.m = d['name'], d['n'], d['m']
        self.p, self.R, self.groups = d['p'], d['R'], d['groups']
        self.Rs = sp.csr_matrix(self.R) if self.n > 2000 else self.R
        self.gid = np.empty(self.n, dtype=int)
        for k, g in enumerate(self.groups):
            self.gid[g] = k
        self.sizes = np.array([len(g) for g in self.groups])
        self.x0 = 1.0 / self.sizes[self.gid]                      # group barycentres
        self.rdiag = np.asarray(self.R.diagonal()).ravel()
    def f(self, x):
        return float(self.p @ x + x @ (self.R @ x))
    def PiN(self, v):                                              # subtract group means
        s = np.bincount(self.gid, weights=v, minlength=self.m)
        return v - (s / self.sizes)[self.gid]
    def proj_P(self, v):                                           # projection onto product of simplices
        out = np.empty_like(v)
        for g in self.groups:
            out[g] = proj_simplex(v[g])
        return out

def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    k = np.arange(1, len(v) + 1)
    rho = np.nonzero(u - css / k > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1), 0.0)

# ------------------------------------------------- spectral shift lambda_N
def lambda_N(I, eps_rel=1e-9):
    n = I.n
    if n <= 400:                                                   # dense, exact
        P = np.eye(n) - np.array([[1.0 / I.sizes[I.gid[i]] if I.gid[i] == I.gid[j] else 0.0
                                   for j in range(n)] for i in range(n)])
        kappa = 1.0 + np.abs(I.R).sum(1).max()
        M = P @ I.R @ P - kappa * (np.eye(n) - P)
        lam = float(np.linalg.eigvalsh((M + M.T) / 2)[-1])
        norm2 = float(np.abs(np.linalg.eigvalsh(I.R)).max())
    else:
        kappa = 1.0 + np.abs(I.R).sum(1).max()
        def mv(z):
            z = np.asarray(z).ravel()
            pz = I.PiN(z)
            return I.PiN(I.Rs @ pz) - kappa * (z - pz)
        op = LinearOperator((n, n), matvec=mv, dtype=float)
        lam = float(eigsh(op, k=1, which='LA', tol=1e-10, return_eigenvectors=False)[0])
        norm2 = float(abs(eigsh(LinearOperator((n, n), matvec=lambda z: I.Rs @ z, dtype=float),
                                k=1, which='LM', tol=1e-8, return_eigenvectors=False)[0]))
        eps_rel = 1e-7
    delta = lam + eps_rel * max(1.0, abs(lam))
    return lam, delta, norm2

# -------------------------------------------------- certified upper bound
def certified_bound(I, xbar, delta):
    g = I.p + 2 * (I.Rs @ xbar - delta * xbar)
    fd = I.f(xbar) + delta * (I.m - xbar @ xbar)
    gm = sum(g[gr].max() for gr in I.groups)
    return float(fd + gm - g @ xbar)

# ------------------------------------------------------------- rounding / LS
def round_groupwise(I, xr):
    x = np.zeros(I.n)
    for g in I.groups:
        x[g[np.argmax(xr[g])]] = 1.0                               # smallest index on ties
    return x

def local_search(I, x, deadline=None):
    sel = np.array([g[np.argmax(x[g])] for g in I.groups])
    x = np.zeros(I.n); x[sel] = 1.0
    w = I.Rs @ x
    while True:
        if deadline is not None and time.perf_counter() > deadline:
            break
        best, bk, bj = 1e-12, -1, -1
        for k, g in enumerate(I.groups):
            j = sel[k]
            d = (I.p[g] - I.p[j] + 2 * (w[g] - w[j]) + I.rdiag[g] + I.rdiag[j]
                 - 2 * np.asarray(I.R[j, g]).ravel())
            d[g == j] = -np.inf
            a = int(np.argmax(d))
            if d[a] > best:
                best, bk, bj = d[a], k, g[a]
        if bk < 0:
            break
        j = sel[bk]
        x[j], x[bj] = 0.0, 1.0
        w = w + np.asarray(I.R[:, bj] - I.R[:, j]).ravel() if not sp.issparse(I.Rs) \
            else w + I.Rs[:, bj].toarray().ravel() - I.Rs[:, j].toarray().ravel()
        sel[bk] = bj
    return x, I.f(x)

def multistart_ls(I, budget=None, starts=None, seed=0):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter(); best = -np.inf; cnt = 0
    while True:
        if starts is not None and cnt >= starts: break
        if budget is not None and time.perf_counter() - t0 >= budget and cnt > 0: break
        x = np.zeros(I.n)
        for g in I.groups:
            x[rng.choice(g)] = 1.0
        dl = None if budget is None else t0 + budget
        _, fx = local_search(I, x, dl)
        best = max(best, fx); cnt += 1
    return best, cnt, time.perf_counter() - t0

# ------------------------------------------------------------------ SR-ADMM
def sradmm(I, delta, norm2, rho=None, eps_abs=1e-7, eps_rel=1e-6, kmax=2000, cg_tol=1e-10):
    n = I.n
    rho = 0.5 * norm2 if rho is None else rho
    x0 = I.x0.copy()
    Kop = LinearOperator((n, n), dtype=float,
                         matvec=lambda z: I.PiN((2 * delta + rho) * np.asarray(z).ravel()
                                                - 2 * (I.Rs @ np.asarray(z).ravel())))
    Rx0 = I.Rs @ x0
    y = x0.copy(); u = np.zeros(n); z = np.zeros(n)
    it = 0; conv = False
    for it in range(1, kmax + 1):
        b = I.PiN(I.p + rho * (y - u) + 2 * Rx0)
        z, _ = cg(Kop, b, x0=z, rtol=cg_tol, atol=0.0, maxiter=1000)
        x = x0 + z
        ynew = np.maximum(x + u, 0.0)
        u = u + x - ynew
        r = np.linalg.norm(x - ynew); s = rho * np.linalg.norm(ynew - y)
        y = ynew
        ep = np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y))
        ed = np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u)
        if r <= ep and s <= ed:
            conv = True; break
    xbar = I.proj_P(x)
    return xbar, it, conv

# -------------------------------------------------------------------- FISTA
def fista(I, delta, norm2, eps=1e-9, kmax=20000):
    L = 2.0 * (abs(delta) + norm2)                                 # Lipschitz bound of grad f_delta
    x = I.x0.copy(); yv = x.copy(); t = 1.0
    for it in range(1, kmax + 1):
        g = I.p + 2 * (I.Rs @ yv - delta * yv)
        xn = I.proj_P(yv + g / L)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        yv = xn + ((t - 1) / tn) * (xn - x)
        done = np.linalg.norm(xn - x) <= eps * max(1.0, np.linalg.norm(x))
        x, t = xn, tn
        if done: return x, it, True
    return x, kmax, False

# ---------------------------------------------------------------------- PAL
def pal(I, lamN, norm2, kmax=3000, M0=5, Mmin=2, Mmax=50, tlo=1e-4, thi=1e-2, epsG=1e-9, tmax=None):
    rho = 2 * max(lamN, 0.0) + 0.1 * norm2 + 1e-8
    L = rho + 2 * norm2
    x = I.x0.copy(); M = M0; t0 = time.perf_counter(); hit = False
    for k in range(kmax):
        w = x.copy(); G = None
        for i in range(M):
            wp = I.proj_P(w + (I.p + 2 * (I.Rs @ w) - rho * (w - x)) / L)
            if i == 0: G = L * np.linalg.norm(wp - x)
            w = wp
        fo, fn = I.f(x), I.f(w); x = w
        if G <= epsG * max(1.0, np.linalg.norm(I.p)): return x, k + 1, True, False
        th = (fn - fo) / max(abs(fn), 1.0)
        M = min(2 * M, Mmax) if th < tlo else (max(M // 2, Mmin) if th > thi else M)
        if tmax is not None and time.perf_counter() - t0 > tmax: hit = True; break
    return x, k + 1, False, hit

# -------------------------------------------------------------------- exact
def exact_enumerate(I, limit=3_000_000):
    N = int(np.prod(I.sizes.astype(object)))
    if N > limit: return None
    best = -np.inf
    for combo in itertools.product(*I.groups):
        x = np.zeros(I.n); x[list(combo)] = 1.0
        best = max(best, I.f(x))
    return best

def exact_highs(I, tlimit=300):
    """Adams-Sherali linearization, variables x_i and y_ij (i<j in different groups)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n = I.n
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if I.gid[i] != I.gid[j]]
    pid = {pr: n + t for t, pr in enumerate(pairs)}
    nv = n + len(pairs)
    c = np.zeros(nv)
    c[:n] = -(I.p + I.rdiag)                                       # maximize -> minimize negative
    for (i, j), v in pid.items():
        c[v] = -2 * I.R[i, j]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for g in I.groups:                                             # sum_{C_k} x = 1
        for j in g: rows.append(r); cols.append(j); vals.append(1.0)
        lo.append(1); hi.append(1); r += 1
    for i in range(n):                                             # sum_{j in C_l} y_ij = x_i, l != group(i)
        for l, g in enumerate(I.groups):
            if l == I.gid[i]: continue
            for j in g:
                rows.append(r); cols.append(pid[(min(i, j), max(i, j))]); vals.append(1.0)
            rows.append(r); cols.append(i); vals.append(-1.0)
            lo.append(0); hi.append(0); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    integ = np.zeros(nv); integ[:n] = 1
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ,
               bounds=Bounds(np.zeros(nv), np.ones(nv)), options=dict(time_limit=tlimit))
    return (-res.fun if res.x is not None else None), res.status

# ------------------------------------------------------------------- driver
def run_one(I, exact_max_n=100, seed=0, ls_budget_cap=120.0):
    row = dict(name=I.name, n=I.n, m=I.m)
    t0 = time.perf_counter()
    lam, delta, norm2 = lambda_N(I)
    setup_t = time.perf_counter() - t0
    row.update(lamN=lam, norm2=norm2, setup_t=setup_t)

    t0 = time.perf_counter()
    xb, it, conv = sradmm(I, delta, norm2); ta = time.perf_counter() - t0
    row.update(aadmm_U=certified_bound(I, xb, delta), aadmm_frel=I.f(xb))
    xr = round_groupwise(I, xb); row['aadmm_fround'] = I.f(xr)
    t1 = time.perf_counter(); _, fls = local_search(I, xr); tls = time.perf_counter() - t1
    row.update(aadmm_fls=fls, aadmm_t=setup_t + ta + tls, aadmm_tls=tls,
               aadmm_it=it, aadmm_conv=conv)

    t0 = time.perf_counter(); xf, itf, cf = fista(I, delta, norm2); tf = time.perf_counter() - t0
    _, ffls = local_search(I, round_groupwise(I, xf))
    row.update(fista_fls=ffls, fista_t=setup_t + tf, fista_it=itf, fista_hit=not cf)

    t0 = time.perf_counter(); xp, itp, cp, hp = pal(I, lam, norm2, tmax=300.0); tp = time.perf_counter() - t0
    xr = round_groupwise(I, xp); row['pal_fround'] = I.f(xr)
    _, pfls = local_search(I, xr)
    row.update(pal_fls=pfls, pal_t=setup_t + tp, pal_it=itp, pal_conv=cp, pal_hit=hp)

    f10, s10, t10 = multistart_ls(I, starts=10, seed=seed)
    row.update(ls10=f10, ls10_t=t10)
    bud = min(row['aadmm_t'], ls_budget_cap)                      # same wall-clock time as SR-ADMM+LS
    fs, ss, ts = multistart_ls(I, budget=bud, seed=seed + 1)
    row.update(ls_same=fs, ls_same_t=ts, ls_same_starts=ss)

    row.update(exact=np.nan, exact_status='')
    if I.n <= exact_max_n:
        e = exact_enumerate(I)
        if e is not None:
            row.update(exact=e, exact_status='enumeration')
        else:
            v, st = exact_highs(I); row.update(exact=v if v is not None else np.nan, exact_status=f'highs_{st}')
    vals = [row[k] for k in ['aadmm_fls', 'fista_fls', 'pal_fls', 'ls10', 'ls_same'] if not np.isnan(row[k])]
    if not np.isnan(row['exact']): vals.append(row['exact'])
    row['ref'] = max(vals)
    row['cert_gap_pct'] = 100 * (row['aadmm_U'] - row['ref']) / abs(row['aadmm_U'])
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+'); ap.add_argument('--out', default='results.csv')
    ap.add_argument('--exact-max-n', type=int, default=100); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    paths = sorted(sum([glob.glob(f) for f in a.files], []))
    rows = []
    for pth in paths:
        I = Inst(load_instance(pth)); r = run_one(I, a.exact_max_n, a.seed); rows.append(r)
        print(f"{I.name}: n={I.n} SR+LS={r['aadmm_fls']:.4g} FISTA+LS={r['fista_fls']:.4g} PAL+LS={r['pal_fls']:.4g} "
              f"LS10={r['ls10']:.4g} LSsame={r['ls_same']:.4g} exact={r['exact']} U={r['aadmm_U']:.4g}", flush=True)
        pd.DataFrame(rows).to_csv(a.out, index=False)

if __name__ == '__main__':
    main()
