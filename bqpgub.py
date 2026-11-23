"""
Reference implementation for the revised BQP-GUB manuscript.

Problem:   max  f(x) = p^T x + x^T R x
           s.t. A x = e (one variable per group), x in {0,1}^n,
with m groups of equal size s (n = m*s); group k = variables k*s .. (k+1)*s-1.

Contents
  * Instance generators (dense mixed-sign BQP-GUB, sparse CMISP-type)
  * delta selection (null-space curvature) and the shifted concave function f_delta
  * certified upper bound (Frank-Wolfe type certificate)
  * AADMM            : ADMM (+ residual balancing, Nesterov/restart) for the shifted relaxation
  * MADMM-ALU        : inexact proximal ascent with adaptive local update counts
  * rounding + 1-swap local search
  * exact enumeration and HiGHS (QSAP linearisation) reference
"""
import itertools
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import milp, LinearConstraint, Bounds


# ----------------------------------------------------------------------------
# Instances
# ----------------------------------------------------------------------------
class Instance:
    def __init__(self, R, p, m, s, name=""):
        self.R = R
        self.p = np.asarray(p, dtype=float)
        self.m, self.s = m, s
        self.n = m * s
        self.name = name
        self._lam = None
        self._nrm = None

    # --- basic operations -------------------------------------------------
    def P(self, v):
        """Projection onto null(A): subtract the group means."""
        V = v.reshape(self.m, self.s)
        return (V - V.mean(axis=1, keepdims=True)).ravel()

    def f(self, x):
        return float(self.p @ x + x @ (self.R @ x))

    def grad_f(self, x):
        return self.p + 2.0 * (self.R @ x)

    # --- spectral quantities ----------------------------------------------
    @property
    def lam(self):
        """lambda_max of R restricted to null(A) (may be negative)."""
        if self._lam is None:
            self._lam = self._restricted_lambda_max()
        return self._lam

    @property
    def norm2(self):
        if self._nrm is None:
            if self.n <= 400:
                Rd = self.R.toarray() if sp.issparse(self.R) else self.R
                self._nrm = float(np.max(np.abs(np.linalg.eigvalsh(Rd))))
            else:
                self._nrm = float(abs(spla.eigsh(self.R, k=1, which="LM",
                                                 return_eigenvectors=False, tol=1e-8)[0]))
        return self._nrm

    def _restricted_lambda_max(self):
        n, m, s = self.n, self.m, self.s
        if n <= 400:
            Rd = self.R.toarray() if sp.issparse(self.R) else np.asarray(self.R)
            Pm = np.kron(np.eye(m), np.eye(s) - np.ones((s, s)) / s)
            w = np.linalg.eigvalsh(Pm @ Rd @ Pm - (1.0 + np.abs(Rd).sum(1).max()) * (np.eye(n) - Pm))
            lam = float(w.max())
            return lam + 1e-9 * max(1.0, abs(lam))
        # shift the kernel of P (range of A^T) far down so it cannot be the top eigenvalue
        Mbig = 1.0 + float(np.abs(self.R).sum(axis=1).max())

        def mv(v):
            Pv = self.P(v)
            return self.P(self.R @ Pv) - Mbig * (v - Pv)

        op = spla.LinearOperator((n, n), matvec=mv, dtype=float)
        lam = float(spla.eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0])
        return lam + 1e-7 * max(1.0, abs(lam))

    # --- shifted concave function -------------------------------------------
    def f_delta(self, x, delta):
        return self.f(x) + delta * (self.m - float(x @ x))

    def grad_f_delta(self, x, delta):
        return self.grad_f(x) - 2.0 * delta * x


def make_dense(m, s, seed, rmax=50, pmax=10):
    rng = np.random.default_rng(seed)
    n = m * s
    B = rng.integers(-rmax, rmax + 1, size=(n, n)).astype(float)
    R = np.triu(B) + np.triu(B, 1).T
    p = rng.integers(-pmax, pmax + 1, size=n).astype(float)
    return Instance(R, p, m, s, name=f"dense-n{n}-s{seed}")


def make_cmisp(m, s, avg_deg, seed, wmax=10):
    """Sparse CMISP-type instance: p = 0, R = W/2 for a random weighted graph."""
    rng = np.random.default_rng(seed)
    n = m * s
    ne = int(n * avg_deg / 2)
    i = rng.integers(0, n, size=ne)
    j = rng.integers(0, n, size=ne)
    keep = (i // s) != (j // s)  # edges inside a cluster can never be selected together
    i, j = i[keep], j[keep]
    w = rng.integers(1, wmax + 1, size=i.size).astype(float)
    W = sp.coo_matrix((w, (i, j)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)  # symmetric, duplicate edges collapse to max weight
    R = (W * 0.5).tocsr()
    return Instance(R, np.zeros(n), m, s, name=f"cmisp-n{n}-s{seed}")


# ----------------------------------------------------------------------------
# Projection and certificate
# ----------------------------------------------------------------------------
def proj_simplices(v, m, s):
    """Row-wise Euclidean projection onto the unit simplex."""
    V = v.reshape(m, s)
    U = -np.sort(-V, axis=1)
    css = np.cumsum(U, axis=1) - 1.0
    ind = np.arange(1, s + 1)
    cond = U - css / ind > 0
    k = cond.sum(axis=1)
    theta = css[np.arange(m), k - 1] / k
    return np.maximum(V - theta[:, None], 0.0).ravel()


def certified_bound(inst, x, delta):
    """
    For ANY x in P = {x>=0, Ax=e}:   max_P f_delta  <=  f_delta(x) + sum_k max_{j in C_k} g_j - g^T x,
    g = grad f_delta(x), because f_delta is concave on {Ax=e} (delta >= lam).
    Returns this value (an upper bound on the BQP-GUB optimum).
    """
    g = inst.grad_f_delta(x, delta)
    return inst.f_delta(x, delta) + float(g.reshape(inst.m, inst.s).max(axis=1).sum() - g @ x)


def round_groups(x, m, s):
    idx = x.reshape(m, s).argmax(axis=1)
    xb = np.zeros(m * s)
    xb[np.arange(m) * s + idx] = 1.0
    return xb


# ----------------------------------------------------------------------------
# AADMM
# ----------------------------------------------------------------------------
def _cg(Kmv, b, z0, tol, maxit):
    z = z0.copy()
    r = b - Kmv(z)
    d = r.copy()
    rs = float(r @ r)
    bn = float(np.sqrt(b @ b)) + 1e-300
    it = 0
    while it < maxit and np.sqrt(rs) > tol * bn:
        Kd = Kmv(d)
        a = rs / float(d @ Kd)
        z += a * d
        r -= a * Kd
        rs2 = float(r @ r)
        d = r + (rs2 / rs) * d
        rs = rs2
        it += 1
    return z, it


def aadmm(inst, delta=None, rho0=None, eps_abs=1e-7, eps_rel=1e-6, max_iter=2000,
          accelerate=False, adapt=False, adapt_iters=100, mu=10.0, tau=2.0, eta=0.999,
          cg_tol=1e-10, cg_maxit=1000, verbose=False):
    """
    ADMM for   min  h(x) + iota_{Ax=e}(x) + iota_{y>=0}(y)   s.t. x = y,
    h(x) = x^T(delta I - R)x - p^T x   (convex on {Ax=e} when delta >= lam).
    Returns dict with relaxed iterate x_P in P, history and counters.
    """
    n, m, s = inst.n, inst.m, inst.s
    if delta is None:
        delta = inst.lam
    R, p = inst.R, inst.p
    if rho0 is None:
        rho0 = 0.5 * inst.norm2 + 1e-6
    rho = rho0
    x0 = np.full(n, 1.0 / s)
    q = inst.P(p + 2.0 * (R @ x0))
    z = np.zeros(n)

    y = x0.copy(); u = np.zeros(n)
    y_prev = y.copy(); u_prev = u.copy()
    yh, uh = y.copy(), u.copy()
    alpha = 1.0
    c_prev = np.inf
    t0 = time.perf_counter()
    cg_total = 0
    nrho = 0
    hist = []
    converged = False

    for k in range(max_iter):
        c0 = 2.0 * delta + rho
        Kmv = lambda v: inst.P(c0 * v - 2.0 * (R @ v))
        b = q + rho * inst.P(yh - uh)
        z, its = _cg(Kmv, b, z, cg_tol, cg_maxit)
        cg_total += its
        x = x0 + z
        y_new = np.maximum(x + uh, 0.0)
        u_new = uh + x - y_new
        r = x - y_new
        sres = -rho * (y_new - yh)
        rn, sn = float(np.linalg.norm(r)), float(np.linalg.norm(sres))
        eps_pri = np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y_new))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u_new)
        hist.append((k, rn, sn, rho))
        y_prev, u_prev = y, u
        y, u = y_new, u_new
        if rn <= eps_pri and sn <= eps_dual:
            converged = True
            break

        # residual balancing (only during the first adapt_iters iterations, then frozen)
        changed = False
        if adapt and k < adapt_iters:
            if rn > mu * sn:
                rho_new = rho * tau
            elif sn > mu * rn:
                rho_new = rho / tau
            else:
                rho_new = rho
            if rho_new != rho:
                u = u * (rho / rho_new)           # rescale the scaled dual variable
                u_prev = u_prev * (rho / rho_new)
                rho = rho_new
                nrho += 1
                changed = True
        if changed or not accelerate:
            yh, uh = y.copy(), u.copy()
            alpha, c_prev = 1.0, np.inf
            continue

        # Nesterov-type extrapolation with restart (Goldstein et al., 2014)
        c = rho * (float(np.sum((u - uh) ** 2)) + float(np.sum((y - yh) ** 2)))
        if c < eta * c_prev:
            alpha_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * alpha ** 2))
            w = (alpha - 1.0) / alpha_new
            yh = y + w * (y - y_prev)
            uh = u + w * (u - u_prev)
            alpha, c_prev = alpha_new, c
        else:
            alpha = 1.0
            yh, uh = y_prev.copy(), u_prev.copy()
            c_prev = c_prev / eta
    t = time.perf_counter() - t0
    x_P = proj_simplices(x, m, s)
    return dict(x=x, y=y, x_P=x_P, delta=delta, iters=k + 1, converged=converged,
                time=t, cg_iters=cg_total, rho_changes=nrho, rho_final=rho, hist=hist)


# ----------------------------------------------------------------------------
# MADMM-ALU (proximal ascent with adaptive local updates)
# ----------------------------------------------------------------------------
def madmm_alu(inst, rho=None, M0=5, Mmin=2, Mmax=50, max_outer=3000, tol=1e-9,
              th_low=1e-4, th_high=1e-2, x0=None):
    """
    x^{k+1} ~ argmax_{x in P}  f(x) - rho/2 ||x - x^k||^2  computed inexactly by M^k projected-gradient
    steps (step 1/L, L = rho + 2||R||_2) started at x^k.  M^k is adapted from the relative ascent.
    """
    n, m, s = inst.n, inst.m, inst.s
    lam_plus = max(inst.lam, 0.0)
    if rho is None:
        rho = 2.0 * lam_plus + 0.1 * inst.norm2 + 1e-8
    L = rho + 2.0 * inst.norm2
    R, p = inst.R, inst.p
    x = np.full(n, 1.0 / s) if x0 is None else x0.copy()
    fx = inst.f(x)
    M = M0
    t0 = time.perf_counter()
    inner_total = 0
    f_hist = [fx]
    ascent_ok = True
    for k in range(max_outer):
        xk = x
        xi = xk
        gm_norm = None
        for i in range(M):
            g = p + 2.0 * (R @ xi) - rho * (xi - xk)
            xn = proj_simplices(xi + g / L, m, s)
            if i == 0:
                gm_norm = L * float(np.linalg.norm(xn - xk))   # gradient mapping of f at x^k
            xi = xn
        inner_total += M
        x = xi
        f_new = inst.f(x)
        if f_new < fx + 0.5 * rho * float(np.sum((x - xk) ** 2)) - 1e-9 * max(1.0, abs(fx)):
            ascent_ok = False
        ratio = (f_new - fx) / max(abs(f_new), 1.0)
        fx = f_new
        f_hist.append(fx)
        if gm_norm <= tol * max(1.0, np.linalg.norm(p)):
            break
        if ratio < th_low:
            M = min(2 * M, Mmax)
        elif ratio > th_high:
            M = max(M // 2, Mmin)
    t = time.perf_counter() - t0
    return dict(x=x, iters=k + 1, inner=inner_total, time=t, f_hist=f_hist,
                ascent_ok=ascent_ok, rho=rho, gm=gm_norm)


# ----------------------------------------------------------------------------
# Rounding / local search
# ----------------------------------------------------------------------------
def local_search_swap(inst, xb, max_moves=10 ** 6):
    """Steepest-ascent 1-swap local search inside groups (binary feasible in, binary feasible out)."""
    m, s, n = inst.m, inst.s, inst.n
    R, p = inst.R, inst.p
    sparse = sp.issparse(R)
    Rc = R.tocsc() if sparse else None
    diag = np.asarray(R.diagonal()).ravel() if sparse else np.diag(R).copy()
    sel = xb.reshape(m, s).argmax(axis=1)           # selected local index per group
    x = xb.copy()
    w = np.asarray(R @ x).ravel()
    gid = np.repeat(np.arange(m), s)
    cand = np.arange(n)
    moves = 0
    while moves < max_moves:
        cur_glob = gid * s + sel[gid]               # currently selected variable of each candidate's group
        if sparse:
            Rjc = np.asarray(R[cand, cur_glob]).ravel()
        else:
            Rjc = R[cand, cur_glob]
        gain = (p[cand] - p[cur_glob] + 2.0 * (w[cand] - w[cur_glob])
                + diag[cand] + diag[cur_glob] - 2.0 * Rjc)
        gain[cand == cur_glob] = -np.inf
        b = int(np.argmax(gain))
        if gain[b] <= 1e-10:
            break
        jn, jo = b, int(cur_glob[b])
        if sparse:
            w += (Rc[:, jn] - Rc[:, jo]).toarray().ravel()
        else:
            w += R[:, jn] - R[:, jo]
        x[jo], x[jn] = 0.0, 1.0
        sel[gid[b]] = jn - gid[b] * s
        moves += 1
    return x


def random_start_ls(inst, starts, seed, time_budget=0.0):
    """Multistart 1-swap local search from random feasible points.
    Runs at least `starts` starts and continues until `time_budget` seconds have elapsed."""
    rng = np.random.default_rng(seed)
    best = -np.inf
    t0 = time.perf_counter()
    done = 0
    while done < starts or time.perf_counter() - t0 < time_budget:
        sel = rng.integers(0, inst.s, size=inst.m)
        x = np.zeros(inst.n)
        x[np.arange(inst.m) * inst.s + sel] = 1.0
        x = local_search_swap(inst, x)
        best = max(best, inst.f(x))
        done += 1
    return best, time.perf_counter() - t0, done


# ----------------------------------------------------------------------------
# Exact / MIP reference
# ----------------------------------------------------------------------------
def exact_enumeration(inst):
    m, s = inst.m, inst.s
    best, arg = -np.inf, None
    Rd = inst.R.toarray() if sp.issparse(inst.R) else inst.R
    for sel in itertools.product(range(s), repeat=m):
        idx = np.arange(m) * s + np.array(sel)
        v = inst.p[idx].sum() + Rd[np.ix_(idx, idx)].sum()
        if v > best:
            best, arg = v, idx.copy()
    return best, arg


def highs_reference(inst, time_limit=120.0):
    """Tight QSAP linearisation (Adams-Sherali type); returns (value, dual_bound, status, time)."""
    n, m, s = inst.n, inst.m, inst.s
    Rd = inst.R.toarray() if sp.issparse(inst.R) else inst.R
    gid = np.repeat(np.arange(m), s)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if gid[i] != gid[j]]
    pid = {pr: n + k for k, pr in enumerate(pairs)}
    nv = n + len(pairs)
    c = np.zeros(nv)
    c[:n] = inst.p + np.diag(Rd)
    for (i, j), v in pid.items():
        c[v] = 2.0 * Rd[i, j]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for k in range(m):
        for j in range(k * s, (k + 1) * s):
            rows.append(r); cols.append(j); vals.append(1.0)
        lo.append(1.0); hi.append(1.0); r += 1
    for i in range(n):
        for l in range(m):
            if l == gid[i]:
                continue
            for j in range(l * s, (l + 1) * s):
                key = (i, j) if i < j else (j, i)
                rows.append(r); cols.append(pid[key]); vals.append(1.0)
            rows.append(r); cols.append(i); vals.append(-1.0)
            lo.append(0.0); hi.append(0.0); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    integrality = np.zeros(nv); integrality[:n] = 1
    t0 = time.perf_counter()
    res = milp(-c, constraints=LinearConstraint(A, lo, hi), integrality=integrality,
               bounds=Bounds(np.zeros(nv), np.ones(nv)),
               options=dict(time_limit=time_limit, disp=False))
    t = time.perf_counter() - t0
    val = -res.fun if res.x is not None else np.nan
    dual = -res.mip_dual_bound if getattr(res, "mip_dual_bound", None) is not None else np.nan
    return dict(value=val, dual_bound=dual, status=res.status, message=res.message, time=t,
                x=None if res.x is None else np.round(res.x[:n]))


# ----------------------------------------------------------------------------
# One-instance driver
# ----------------------------------------------------------------------------
def run_methods(inst, ls_starts=10, seed=0, **aadmm_kw):
    out = {}
    delta = inst.lam
    a = aadmm(inst, delta=delta, **aadmm_kw)
    xP = a["x_P"]
    U = certified_bound(inst, xP, delta)
    t0 = time.perf_counter()
    xr = round_groups(xP, inst.m, inst.s)
    f_r = inst.f(xr)
    xl = local_search_swap(inst, xr)
    f_l = inst.f(xl)
    t_ls = time.perf_counter() - t0
    out["aadmm"] = dict(delta=delta, U=U, f_relaxed=inst.f(xP), f_round=f_r, f_ls=f_l,
                        time=a["time"], time_ls=t_ls, iters=a["iters"], converged=a["converged"],
                        cg=a["cg_iters"], rho_changes=a["rho_changes"])
    mm = madmm_alu(inst)
    xr2 = round_groups(mm["x"], inst.m, inst.s)
    t0 = time.perf_counter()
    f_r2 = inst.f(xr2)
    xl2 = local_search_swap(inst, xr2)
    f_l2 = inst.f(xl2)
    t_ls2 = time.perf_counter() - t0
    out["madmm"] = dict(f_relaxed=inst.f(mm["x"]), f_round=f_r2, f_ls=f_l2, time=mm["time"],
                        time_ls=t_ls2, iters=mm["iters"], inner=mm["inner"],
                        ascent_ok=mm["ascent_ok"])
    best, tt, nst = random_start_ls(inst, ls_starts, seed)
    out["ls"] = dict(value=best, time=tt, starts=nst)
    budget = out["madmm"]["time"] + out["madmm"]["time_ls"]
    best2, tt2, nst2 = random_start_ls(inst, ls_starts, seed + 1, time_budget=budget)
    out["ls_matched"] = dict(value=best2, time=tt2, starts=nst2)
    return out
