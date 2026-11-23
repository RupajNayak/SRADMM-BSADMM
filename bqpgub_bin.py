"""
MADMM-ALU with the binary-set splitting (nondegenerate ADMM for the nonconvex problem):

    min  phi(x) + iota_X(x) + iota_Y(y)   s.t.  x = y,       phi(x) = -(p^T x + x^T R x),

X = {Ax = e, 0 <= x <= 1} (product of simplices), Y = X intersect {0,1}^n (one-hot per group).
Local sweeps (u fixed):   x <- argmin_X phi(x) + rho/2 ||x - y + u||^2   (inexact: FISTA / projected gradient),
                          y <- Pi_Y(x + u)   (one-hot at the largest entry of each group)
then the dual update u += x - y.  Penalty with floor rho_min and rescaling of u; adaptive number of sweeps M.
"""
import time
import numpy as np
from bqpgub import proj_simplices, local_search_swap, round_groups


def proj_onehot(v, m, s):
    return round_groups(v, m, s)


def _x_update(inst, y, u, rho, x_init, inner, tol, convex):
    """approx. argmin_{x in X}  -p^T x - x^T R x + rho/2 ||x - y + u||^2  (nonconvex unless rho >= 2 lam_N)"""
    m, s = inst.m, inst.s
    R, p = inst.R, inst.p
    L = 2.0 * inst.norm2 + rho
    c = y - u
    x = x_init.copy()
    z = x.copy()
    t = 1.0
    for _ in range(inner):
        g = -p - 2.0 * (R @ z) + rho * (z - c)
        xn = proj_simplices(z - g / L, m, s)
        dx = float(np.linalg.norm(xn - x))
        if convex:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = xn + ((t - 1.0) / tn) * (xn - x)
            t = tn
        else:
            z = xn
        x = xn
        if dx <= tol * max(1.0, float(np.linalg.norm(x))):
            break
    return x


def madmm_alu_bin(inst, rho_min, rho0=None, M0=5, Mmin=5, Mmax=50, mu=10.0, tau=1.0, K=300,
                  eps_abs=1e-6, eps_rel=1e-4, inner=300, inner_tol=1e-9, seed=0):
    n, m, s = inst.n, inst.m, inst.s
    rng = np.random.default_rng(seed)
    sel = rng.integers(0, s, size=m)
    y = np.zeros(n)
    y[np.arange(m) * s + sel] = 1.0
    u = np.zeros(n)
    rho = max(rho_min, rho0 if rho0 is not None else rho_min)
    M = M0
    x = y.copy()
    best_val, best_x = inst.f(y), y.copy()
    c_prev = np.inf
    t0 = time.perf_counter()
    inner_total = 0
    converged = False
    for k in range(K):
        convex = rho >= 2.0 * inst.lam
        y_old = y.copy()
        yl = y.copy()
        for i in range(M):
            x = _x_update(inst, yl, u, rho, x, inner, inner_tol, convex)
            yl = proj_onehot(x + u, m, s)
            inner_total += 1
        y = yl
        u = u + x - y
        r = x - y
        sres = -rho * (y - y_old)
        rn, sn = float(np.linalg.norm(r)), float(np.linalg.norm(sres))
        # keep the best binary candidate seen so far (y is binary; also the rounded x)
        for cand in (y, proj_onehot(x, m, s)):
            v = inst.f(cand)
            if v > best_val:
                best_val, best_x = v, cand.copy()
        c = rn ** 2 + sn ** 2
        eps_pri = np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y))
        eps_dual = np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u)
        if rn <= eps_pri and sn <= eps_dual:
            converged = True
            break
        # penalty with floor + rescaling of the scaled dual
        if rn > mu * sn:
            rho_new = max(rho_min, rho * (1.0 + tau))
        elif sn > mu * rn:
            rho_new = max(rho_min, rho / (1.0 + tau))
        else:
            rho_new = rho
        if rho_new != rho:
            u = u * (rho / rho_new)
            rho = rho_new
        # adaptive number of local sweeps
        if c < c_prev:
            M = min(2 * M, Mmax)
        else:
            M = max(int(np.ceil(M / 2)), Mmin)
        c_prev = c
    t = time.perf_counter() - t0
    return dict(x=best_x, value=best_val, iters=k + 1, converged=converged, time=t, inner=inner_total,
                rho_final=rho)
