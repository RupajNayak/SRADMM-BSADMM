#!/usr/bin/env python3
"""
bqpgub_bench.py -- parallel benchmark for BQP-GUB:  AADMM, MADMM-ALU, PAL, multistart local search,
and (for benchmarking only) a MIP reference with HiGHS, CPLEX or Gurobi.

    max  p^T x + x^T R x     s.t.  exactly one variable selected in each group,  x binary.

Input files (one instance per file)
  *.txt  Matlab-style text:  n = ..;  m = ..;  p = [..;..];  R = [row;row;..];  C1 = [i j k ..]; ... (1-based indices)
  *.npz  keys R, p, sizes, perm (seed, bs ignored):  group k = perm[off[k]:off[k+1]], off = cumsum(sizes)

Examples
  python bqpgub_bench.py data/*.txt data/*.npz --workers 8
  python bqpgub_bench.py bqpgubs-250-1.npz --mip highs --mip-time 600 --madmm-starts 5
  python bqpgub_bench.py data/*.npz --methods aadmm,pal --time-limit 900 --workers 4
  python bqpgub_bench.py --selftest

Parallelism: instances are loaded one after another into *shared memory* (one copy of R, no matter how many
workers use it); every (instance, method) pair is a separate process task.  Wall-clock times are measured
while other tasks run; use --workers 1 for clean timings.
"""
import os
import sys


def _argval(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if a.startswith(flag + "="):
            return a.split("=", 1)[1]
    return default


# BLAS threads per process (must be set before numpy is imported; spawned workers re-run this code)
_BT = _argval("--blas-threads", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _BT

import argparse
import csv
import itertools
import json
import re
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import get_context, shared_memory

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# =============================================================================
# Instance
# =============================================================================
class GInstance:
    """Dense symmetric R (a numpy view, possibly on shared memory), p, groups of arbitrary sizes."""

    def __init__(self, R, p, groups, lam=None, norm2=None, name=""):
        self.R = R
        self.p = np.asarray(p, float).ravel()
        self.groups = [np.asarray(g, dtype=np.int64) for g in groups]
        self.m = len(self.groups)
        self.n = self.p.size
        self.name = name
        self.gid = np.empty(self.n, dtype=np.int64)
        for k, g in enumerate(self.groups):
            self.gid[g] = k
        self.gsize = np.array([g.size for g in self.groups])
        self.classes = {}
        for s in np.unique(self.gsize):
            ks = np.where(self.gsize == s)[0]
            self.classes[int(s)] = (ks, np.stack([self.groups[k] for k in ks]))
        self.lam = lam
        self.norm2 = norm2

    # --- basic operations ---
    def P(self, v):
        """orthogonal projection onto {Az = 0} (subtract group means)"""
        out = np.array(v, dtype=float, copy=True)
        for s, (ks, idx) in self.classes.items():
            V = v[idx]
            out[idx] = V - V.mean(axis=1, keepdims=True)
        return out

    def barycentre(self):
        return 1.0 / self.gsize[self.gid]

    def f(self, x):
        return float(self.p @ x + x @ (self.R @ x))

    def grad_f(self, x):
        return self.p + 2.0 * (self.R @ x)

    def f_delta(self, x, delta):
        return self.f(x) + delta * (self.m - float(x @ x))

    def grad_f_delta(self, x, delta):
        return self.grad_f(x) - 2.0 * delta * x


def _symmetrize_inplace(R, blk=1024):
    n = R.shape[0]
    for i in range(0, n, blk):
        ie = min(i + blk, n)
        for j in range(i, n, blk):
            je = min(j + blk, n)
            if i == j:
                A = R[i:ie, i:ie]
                R[i:ie, i:ie] = 0.5 * (A + A.T)
            else:
                S = 0.5 * (R[i:ie, j:je] + R[j:je, i:ie].T)
                R[i:ie, j:je] = S
                R[j:je, i:ie] = S.T


def _rowabs_max(R, blk=1024):
    mx = 0.0
    for i in range(0, R.shape[0], blk):
        mx = max(mx, float(np.abs(R[i:i + blk]).sum(axis=1).max()))
    return mx


def spectral_data(inst, rigorous=False):
    """lambda_N (largest eigenvalue of R on null(A), with a tiny safety margin), ||R||_2, and delta."""
    R, n = inst.R, inst.n
    rowmax = _rowabs_max(R)
    if n <= 400:
        w = np.linalg.eigvalsh(R)
        norm2 = float(np.max(np.abs(w)))
        G = np.zeros((inst.m, n)); G[inst.gid, np.arange(n)] = 1.0
        Pm = np.eye(n) - G.T @ (G / inst.gsize[:, None])
        wn = np.linalg.eigvalsh(Pm @ R @ Pm - (1.0 + rowmax) * (np.eye(n) - Pm))
        lam = float(wn.max()); eps = 1e-9
    else:
        norm2 = float(abs(spla.eigsh(spla.aslinearoperator(R), k=1, which="LM", return_eigenvectors=False, tol=1e-8)[0]))
        big = 1.0 + rowmax

        def mv(v):
            Pv = inst.P(v)
            return inst.P(R @ Pv) - big * (v - Pv)
        op = spla.LinearOperator((n, n), matvec=mv, dtype=float)
        lam = float(spla.eigsh(op, k=1, which="LA", return_eigenvectors=False, tol=1e-10)[0]); eps = 1e-7
    lam = lam + eps * max(1.0, abs(lam))
    delta = rowmax if rigorous else lam      # ||R||_inf >= lambda_max(R) >= lambda_N : unconditionally valid
    return lam, norm2, delta


# =============================================================================
# Loaders (return p, dense symmetric R in a numpy array, groups)
# =============================================================================
def peek_n(path):
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=True) as z:
            return int(z["p"].size)
    with open(path) as fh:
        for line in fh:
            mt = re.match(r"\s*n\s*=\s*([0-9.eE+-]+)", line)
            if mt:
                return int(float(mt.group(1)))
    raise ValueError(f"no 'n =' line in {path}")


def _nums(s):
    return np.array([float(v) for v in re.split(r"[;\s,]+", s.strip()) if v], dtype=float)


def load_txt(path):
    t = open(path).read()
    n = int(float(re.search(r"^\s*n\s*=\s*([0-9.eE+-]+)", t, flags=re.M).group(1)))
    m = int(float(re.search(r"^\s*m\s*=\s*([0-9.eE+-]+)", t, flags=re.M).group(1)))
    p = _nums(re.search(r"^\s*p\s*=\s*\[(.*?)\]", t, flags=re.M | re.S).group(1))
    body = re.search(r"^\s*R\s*=\s*\[(.*?)\]", t, flags=re.M | re.S).group(1)
    rows = [r for r in re.split(r"[;\n]", body) if r.strip()]
    R = np.empty((n, n)); assert len(rows) == n, f"R has {len(rows)} rows, expected {n}"
    for i, r in enumerate(rows):
        R[i] = _nums(r)
    groups = {}
    for mt in re.finditer(r"^\s*C(\d+)\s*=\s*\[(.*?)\]", t, flags=re.M | re.S):
        groups[int(mt.group(1))] = _nums(mt.group(2)).astype(np.int64) - 1          # 1-based -> 0-based
    groups = [groups[k] for k in sorted(groups)]
    assert len(groups) == m and p.size == n
    return p, R, groups


def load_npz(path, perm_mode="groups"):
    with np.load(path, allow_pickle=True) as z:
        R = z["R"]
        if R.dtype == object:
            R = R.item(); R = R.toarray() if sp.issparse(R) else np.asarray(R)
        R = np.ascontiguousarray(R, dtype=np.float64)
        p = np.asarray(z["p"], float).ravel()
        sizes = np.asarray(z["sizes"]).astype(np.int64)
        perm = np.asarray(z["perm"]).astype(np.int64) if "perm" in z.files else np.arange(p.size)
    if perm_mode == "inverse":
        inv = np.empty_like(perm); inv[perm] = np.arange(perm.size); perm = inv
    off = np.r_[0, np.cumsum(sizes)]
    groups = [perm[off[k]:off[k + 1]] for k in range(len(sizes))]
    return p, R, groups


# =============================================================================
# Algorithms  (parameters as in the manuscript)
# =============================================================================
def proj_simplices(inst, v):
    out = np.empty(inst.n)
    for s, (ks, idx) in inst.classes.items():
        V = v[idx]
        U = -np.sort(-V, axis=1)
        css = np.cumsum(U, axis=1) - 1.0
        k = (U - css / np.arange(1, s + 1) > 0).sum(axis=1)
        theta = css[np.arange(V.shape[0]), k - 1] / k
        out[idx] = np.maximum(V - theta[:, None], 0.0)
    return out


def round_groups(inst, x):
    out = np.zeros(inst.n)
    for s, (ks, idx) in inst.classes.items():
        j = x[idx].argmax(axis=1)
        out[idx[np.arange(idx.shape[0]), j]] = 1.0
    return out


def selection(inst, xb):
    return np.array([g[np.argmax(xb[g])] for g in inst.groups], dtype=np.int64)


def certified_bound(inst, x, delta):
    """z* <= f_delta(x) + sum_k max_{j in C_k} g_j - g^T x   for every x in P, g = grad f_delta(x)   (delta >= lambda_N)."""
    g = inst.grad_f_delta(x, delta)
    mx = sum(float(g[idx].max(axis=1).sum()) for s, (ks, idx) in inst.classes.items())
    return inst.f_delta(x, delta) + mx - float(g @ x)


def local_search_swap(inst, xb, max_moves=10 ** 7):
    """steepest-ascent one-swap local search (R symmetric: rows used instead of columns)"""
    n, R, p = inst.n, inst.R, inst.p
    diag = np.diag(R).copy()
    gid = inst.gid
    sel = selection(inst, xb)
    x = xb.copy(); w = R @ x; cand = np.arange(n); moves = 0
    while moves < max_moves:
        cur = sel[gid]
        gain = p[cand] - p[cur] + 2.0 * (w[cand] - w[cur]) + diag[cand] + diag[cur] - 2.0 * R[cand, cur]
        gain[cand == cur] = -np.inf
        b = int(np.argmax(gain))
        if gain[b] <= 1e-10:
            break
        jn, jo = b, int(cur[b])
        w += R[jn] - R[jo]
        x[jo], x[jn] = 0.0, 1.0; sel[gid[b]] = jn; moves += 1
    return x


def _cg(Kmv, b, z0, tol, maxit):
    z = z0.copy(); r = b - Kmv(z); d = r.copy(); rs = float(r @ r)
    bn = float(np.sqrt(b @ b)) + 1e-300; it = 0
    while it < maxit and np.sqrt(rs) > tol * bn:
        Kd = Kmv(d); a = rs / float(d @ Kd)
        z += a * d; r -= a * Kd
        rs2 = float(r @ r); d = r + (rs2 / rs) * d; rs = rs2; it += 1
    return z, it


def run_aadmm(inst, delta, tmax, eps_abs=1e-7, eps_rel=1e-6, max_iter=2000, cg_tol=1e-10, cg_maxit=1000):
    """fixed-penalty AADMM:  x in {Ax=e} (CG on the null space),  y >= 0,  rho = 0.5 ||R||_2"""
    n = inst.n; R, p = inst.R, inst.p
    rho = 0.5 * inst.norm2 + 1e-6
    x0 = inst.barycentre()
    q = inst.P(p + 2.0 * (R @ x0))
    z = np.zeros(n); y = x0.copy(); u = np.zeros(n)
    c0 = 2.0 * delta + rho
    Kmv = lambda v: inst.P(c0 * v - 2.0 * (R @ v))
    t0 = time.perf_counter(); conv = False; timed_out = False; cg_total = 0
    for k in range(max_iter):
        z, its = _cg(Kmv, q + rho * inst.P(y - u), z, cg_tol, cg_maxit); cg_total += its
        x = x0 + z
        y_new = np.maximum(x + u, 0.0)
        u = u + x - y_new
        rn = np.linalg.norm(x - y_new); sn = rho * np.linalg.norm(y_new - y); y = y_new
        if rn <= np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y)) and \
           sn <= np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u):
            conv = True; break
        if time.perf_counter() - t0 > tmax:
            timed_out = True; break
    t = time.perf_counter() - t0
    xP = proj_simplices(inst, x)
    U = certified_bound(inst, xP, delta)
    return xP, dict(U=U, delta=delta, iters=k + 1, converged=conv, timed_out=timed_out, cg_iters=cg_total, time=t)


def run_pal(inst, tmax, M0=5, Mmin=2, Mmax=50, max_outer=3000, tol=1e-9, th_low=1e-4, th_high=1e-2):
    """proximal ascent with adaptive local updates (dual-free)"""
    rho = 2.0 * max(inst.lam, 0.0) + 0.1 * inst.norm2 + 1e-8
    L = rho + 2.0 * inst.norm2
    R, p = inst.R, inst.p
    x = inst.barycentre(); fx = inst.f(x); M = M0
    t0 = time.perf_counter(); ok = True; timed_out = False; gm = np.inf
    for k in range(max_outer):
        xk = x; xi = xk
        for i in range(M):
            xn = proj_simplices(inst, xi + (p + 2.0 * (R @ xi) - rho * (xi - xk)) / L)
            if i == 0:
                gm = L * float(np.linalg.norm(xn - xk))
            xi = xn
        x = xi
        f_new = inst.f(x)
        if f_new < fx + 0.5 * rho * float(np.sum((x - xk) ** 2)) - 1e-9 * max(1.0, abs(fx)):
            ok = False
        ratio = (f_new - fx) / max(abs(f_new), 1.0); fx = f_new
        if gm <= tol * max(1.0, np.linalg.norm(p)):
            break
        if time.perf_counter() - t0 > tmax:
            timed_out = True; break
        if ratio < th_low: M = min(2 * M, Mmax)
        elif ratio > th_high: M = max(M // 2, Mmin)
    return x, dict(iters=k + 1, converged=(not timed_out and k + 1 < max_outer), timed_out=timed_out, ascent_ok=ok,
                   time=time.perf_counter() - t0, rho=rho)


def _x_update(inst, y, u, rho, x_init, inner, tol, convex, deadline):
    L = 2.0 * inst.norm2 + rho
    c = y - u; x = x_init.copy(); z = x.copy(); t = 1.0
    for _ in range(inner):
        g = -inst.p - 2.0 * (inst.R @ z) + rho * (z - c)
        xn = proj_simplices(inst, z - g / L)
        dx = float(np.linalg.norm(xn - x))
        if convex:
            tn = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t)); z = xn + ((t - 1.0) / tn) * (xn - x); t = tn
        else:
            z = xn
        x = xn
        if dx <= tol * max(1.0, float(np.linalg.norm(x))) or time.perf_counter() > deadline:
            break
    return x


def run_madmm(inst, seed, tmax, c_min=0.1, M0=5, Mmin=5, Mmax=50, mu=10.0, tau=1.0, K=None,
              eps_abs=1e-6, eps_rel=1e-4, inner=None, inner_tol=1e-9):
    """ADMM with binary-set splitting (x in polytope, y in {one-hot per group}); rho_min = rho0 = c_min ||R||_2"""
    n = inst.n
    small = n <= 20
    K = K or (300 if small else 100)
    inner = inner or (300 if small else 100)
    rho_min = c_min * inst.norm2
    rng = np.random.default_rng(seed)
    y = np.zeros(n)
    for g in inst.groups:
        y[g[rng.integers(0, g.size)]] = 1.0
    u = np.zeros(n); rho = rho_min; M = M0
    x = y.copy(); best_val, best_x = inst.f(y), y.copy(); c_prev = np.inf
    t0 = time.perf_counter(); deadline = t0 + tmax; conv = False; timed_out = False
    for k in range(K):
        convex = rho >= 2.0 * inst.lam
        y_old = y.copy(); yl = y.copy()
        for i in range(M):
            x = _x_update(inst, yl, u, rho, x, inner, inner_tol, convex, deadline)
            yl = round_groups(inst, x + u)
            if time.perf_counter() > deadline:
                break
        y = yl; u = u + x - y
        rn = float(np.linalg.norm(x - y)); sn = float(rho * np.linalg.norm(y - y_old))
        for cand in (y, round_groups(inst, x)):
            v = inst.f(cand)
            if v > best_val:
                best_val, best_x = v, cand.copy()
        if time.perf_counter() > deadline:
            timed_out = True; break
        c = rn ** 2 + sn ** 2
        if rn <= np.sqrt(n) * eps_abs + eps_rel * max(np.linalg.norm(x), np.linalg.norm(y)) and \
           sn <= np.sqrt(n) * eps_abs + eps_rel * rho * np.linalg.norm(u):
            conv = True; break
        if rn > mu * sn: rho_new = max(rho_min, rho * (1.0 + tau))
        elif sn > mu * rn: rho_new = max(rho_min, rho / (1.0 + tau))
        else: rho_new = rho
        if rho_new != rho:
            u = u * (rho / rho_new); rho = rho_new
        M = min(2 * M, Mmax) if c < c_prev else max(int(np.ceil(M / 2)), Mmin)
        c_prev = c
    return best_x, dict(iters=k + 1, converged=conv, timed_out=timed_out, time=time.perf_counter() - t0, rho_final=rho)


def run_ls(inst, starts, seed, seconds):
    rng = np.random.default_rng(seed); best = -np.inf; best_x = None; done = 0
    t0 = time.perf_counter()
    while done < starts or time.perf_counter() - t0 < seconds:
        x = np.zeros(inst.n)
        for g in inst.groups:
            x[g[rng.integers(0, g.size)]] = 1.0
        x = local_search_swap(inst, x); v = inst.f(x)
        if v > best:
            best, best_x = v, x
        done += 1
    return best_x, dict(starts=done, time=time.perf_counter() - t0)


# =============================================================================
# MIP references (benchmark only)
# =============================================================================
def run_mip(inst, solver, time_limit, threads):
    if solver == "highs":
        return _mip_highs(inst, time_limit)
    if solver == "cplex":
        return _mip_cplex(inst, time_limit, threads)
    if solver == "gurobi":
        return _mip_gurobi(inst, time_limit, threads)
    raise ValueError(solver)


def _mip_highs(inst, time_limit):
    """HiGHS on the Adams-Sherali linearisation y_ij = x_i x_j (i, j in different groups); O(n^2) variables"""
    from scipy.optimize import milp, LinearConstraint, Bounds
    n, m, R = inst.n, inst.m, inst.R
    gid = inst.gid
    iu, ju = np.triu_indices(n, 1)
    keep = gid[iu] != gid[ju]
    iu, ju = iu[keep], ju[keep]
    npair = iu.size
    pid = np.full((n, n), -1, dtype=np.int64)
    pid[iu, ju] = n + np.arange(npair); pid[ju, iu] = pid[iu, ju]
    nv = n + npair
    c = np.zeros(nv); c[:n] = inst.p + np.diag(R); c[n:] = 2.0 * R[iu, ju]
    rows, cols, vals, lo, hi = [], [], [], [], []
    r = 0
    for g in inst.groups:
        rows += [r] * g.size; cols += g.tolist(); vals += [1.0] * g.size; lo.append(1.0); hi.append(1.0); r += 1
    for i in range(n):
        for l, g in enumerate(inst.groups):
            if l == gid[i]:
                continue
            rows += [r] * g.size; cols += pid[i, g].tolist(); vals += [1.0] * g.size
            rows.append(r); cols.append(i); vals.append(-1.0); lo.append(0.0); hi.append(0.0); r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, nv))
    integ = np.zeros(nv); integ[:n] = 1
    t0 = time.perf_counter()
    res = milp(-c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(np.zeros(nv), np.ones(nv)),
               options=dict(time_limit=time_limit, disp=False))
    t = time.perf_counter() - t0
    val = float(-res.fun) if res.x is not None else float("nan")
    dual = float(-res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else float("nan")
    x = None if res.x is None else np.round(res.x[:n])
    return x, dict(solver="HiGHS", value=val, bound=dual, status=("optimal" if res.status == 0 else f"limit/other({res.status})"), time=t,
                   formulation="linearisation (n + #pairs variables)")


def _mip_cplex(inst, time_limit, threads):
    """CPLEX on the nonconvex binary QP  max p'x + x'Rx  (optimalitytarget = global); needs `pip install cplex`"""
    import cplex
    n, m, R = inst.n, inst.m, inst.R
    c = cplex.Cplex()
    c.set_log_stream(None); c.set_error_stream(None); c.set_warning_stream(None); c.set_results_stream(None)
    c.objective.set_sense(c.objective.sense.maximize)
    c.variables.add(obj=inst.p.tolist(), types="B" * n)
    # CPLEX objective is  p'x + 0.5 x'Qx  ->  Q = 2R
    c.objective.set_quadratic([cplex.SparsePair(ind=list(range(n)), val=(2.0 * R[i]).tolist()) for i in range(n)])
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=g.tolist(), val=[1.0] * g.size) for g in inst.groups],
                             senses="E" * m, rhs=[1.0] * m)
    c.parameters.optimalitytarget.set(3)                  # global optimum for nonconvex problems
    c.parameters.timelimit.set(float(time_limit))
    c.parameters.clocktype.set(2)                          # wall-clock time (default may count CPU time over all threads)
    if threads:
        c.parameters.threads.set(int(threads))
    t0 = time.perf_counter(); w0 = time.time()
    c.solve()
    t = time.perf_counter() - t0
    wall_clock = time.time() - w0
    sol = c.solution
    try:
        val = float(sol.get_objective_value()); x = np.round(np.array(sol.get_values()))
    except Exception:
        val, x = float("nan"), None
    try:
        bound = float(sol.MIP.get_best_objective())
    except Exception:
        bound = float("nan")
    return x, dict(solver="CPLEX", value=val, bound=bound, status=sol.get_status_string(), time=t,
                   wall_clock=wall_clock, formulation="nonconvex binary QP (global)")


def _mip_gurobi(inst, time_limit, threads):
    """Gurobi on the nonconvex binary QP  max p'x + x'Rx  (NonConvex=2); needs `pip install gurobipy` (+ licence)"""
    import gurobipy as gp
    from gurobipy import GRB
    n, m, R = inst.n, inst.m, inst.R
    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    mdl = gp.Model(env=env)
    x = mdl.addMVar(n, vtype=GRB.BINARY)
    mdl.setObjective(inst.p @ x + x @ R @ x, GRB.MAXIMIZE)
    for g in inst.groups:
        mdl.addConstr(x[g].sum() == 1.0)
    mdl.Params.NonConvex = 2                              # global optimisation of the nonconvex objective
    mdl.Params.TimeLimit = float(time_limit)
    if threads:
        mdl.Params.Threads = int(threads)
    t0 = time.perf_counter(); w0 = time.time()
    mdl.optimize()
    t = time.perf_counter() - t0
    wall_clock = time.time() - w0                         # unlike perf_counter, keeps counting while a Mac sleeps
    status = {GRB.OPTIMAL: "optimal", GRB.TIME_LIMIT: "time limit exceeded"}.get(mdl.Status, f"status {mdl.Status}")
    try:
        val = float(mdl.ObjVal); xv = np.round(np.array(x.X))
    except Exception:
        val, xv = float("nan"), None
    try:
        bound = float(mdl.ObjBound)
    except Exception:
        bound = float("nan")
    return xv, dict(solver="Gurobi", value=val, bound=bound, status=status, time=t,
                    solver_runtime=float(mdl.Runtime), wall_clock=wall_clock,
                    formulation="nonconvex binary QP (global)")


# =============================================================================
# Worker task
# =============================================================================
def _attach(shm_name, n):
    try:
        shm = shared_memory.SharedMemory(name=shm_name, track=False)          # Python >= 3.13
    except TypeError:
        shm = shared_memory.SharedMemory(name=shm_name)
        try:                                                                  # avoid resource_tracker double-unlink
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
    return shm, np.ndarray((n, n), dtype=np.float64, buffer=shm.buf)


def worker(spec):
    t_start = time.perf_counter()
    shm = None
    try:
        shm, R = _attach(spec["shm"], spec["n"])
        groups = np.split(np.asarray(spec["gcat"]), np.cumsum(spec["gsizes"])[:-1])
        inst = GInstance(R, spec["p"], groups, lam=spec["lam"], norm2=spec["norm2"])
        meth = spec["method"]
        out = dict(method=meth, tag=spec.get("tag", meth), ok=True)
        if meth == "aadmm":
            xP, info = run_aadmm(inst, spec["delta"], spec["tmax"])
            xr = round_groups(inst, xP)
            t0 = time.perf_counter(); xl = local_search_swap(inst, xr); tls = time.perf_counter() - t0
            out.update(info, f_relaxed=inst.f(xP), f_round=inst.f(xr), f_ls=inst.f(xl), time_ls=tls, sel=selection(inst, xl).tolist())
        elif meth == "pal":
            x, info = run_pal(inst, spec["tmax"])
            xr = round_groups(inst, x)
            t0 = time.perf_counter(); xl = local_search_swap(inst, xr); tls = time.perf_counter() - t0
            out.update(info, f_relaxed=inst.f(x), f_round=inst.f(xr), f_ls=inst.f(xl), time_ls=tls, sel=selection(inst, xl).tolist())
        elif meth == "madmm":
            x, info = run_madmm(inst, spec["seed"], spec["tmax"])
            t0 = time.perf_counter(); xl = local_search_swap(inst, x); tls = time.perf_counter() - t0
            out.update(info, f_round=inst.f(x), f_ls=inst.f(xl), time_ls=tls, seed=spec["seed"], sel=selection(inst, xl).tolist())
        elif meth == "ls":
            x, info = run_ls(inst, spec["starts"], spec["seed"], spec["seconds"])
            out.update(info, f_ls=inst.f(x), sel=selection(inst, x).tolist())
        elif meth == "mip":
            x, info = run_mip(inst, spec["solver"], spec["mip_time"], spec["mip_threads"])
            out.update(info)
            if x is not None:
                xr = round_groups(inst, x)                      # feasible binary point recovered from the MIP solution
                out["f_check"] = inst.f(xr)
        else:
            raise ValueError(meth)
        out["wall"] = time.perf_counter() - t_start
        return out
    except Exception as e:
        return dict(method=spec["method"], tag=spec.get("tag", spec["method"]), ok=False, error=f"{type(e).__name__}: {e}",
                    trace=traceback.format_exc(limit=4))
    finally:
        try:
            del R, inst
        except Exception:
            pass
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


# =============================================================================
# Driver
# =============================================================================
def fmt(v, d=1):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.{d}f}"


def finalize(meta, results, args):
    """combine task results into one row"""
    row = dict(instance=meta["name"], n=meta["n"], m=meta["m"], lambda_N=meta["lam"], delta=meta["delta"], delta_mode=meta["delta_mode"],
               norm2=meta["norm2"], load_s=meta["load_s"], spectral_s=meta["spec_s"])
    errs = [r for r in results if not r.get("ok", True)]
    row["errors"] = "; ".join(f"{r['tag']}: {r['error']}" for r in errs)
    by = {}
    for r in results:
        if r.get("ok", True):
            by.setdefault(r["method"], []).append(r)
    cands = []
    sels = {}
    if "aadmm" in by:
        a = by["aadmm"][0]
        row.update(bound_U=a["U"], aadmm_relaxed=a["f_relaxed"], aadmm_round=a["f_round"], aadmm_ls=a["f_ls"], aadmm_time=a["time"] + a["time_ls"],
                   aadmm_iters=a["iters"], aadmm_converged=a["converged"], aadmm_timed_out=a["timed_out"])
        cands.append(a["f_ls"]); sels["aadmm"] = a["sel"]
    if "pal" in by:
        a = by["pal"][0]
        row.update(pal_relaxed=a["f_relaxed"], pal_round=a["f_round"], pal_ls=a["f_ls"], pal_time=a["time"] + a["time_ls"], pal_iters=a["iters"],
                   pal_converged=a["converged"], pal_timed_out=a["timed_out"], pal_ascent_ok=a["ascent_ok"])
        cands.append(a["f_ls"]); sels["pal"] = a["sel"]
    if "madmm" in by:
        rs = sorted(by["madmm"], key=lambda r: r["seed"])
        raw = np.array([r["f_round"] for r in rs]); lsv = np.array([r["f_ls"] for r in rs])
        row.update(madmm_starts=len(rs), madmm_first_raw=raw[0], madmm_first_ls=lsv[0], madmm_best_raw=raw.max(), madmm_best_ls=lsv.max(),
                   madmm_mean_raw=raw.mean(), madmm_mean_ls=lsv.mean(), madmm_min_ls=lsv.min(),
                   madmm_time_mean=float(np.mean([r["time"] + r["time_ls"] for r in rs])), madmm_converged_all=all(r["converged"] for r in rs),
                   madmm_timed_out_any=any(r["timed_out"] for r in rs))
        b = rs[int(np.argmax(lsv))]
        cands.append(lsv.max()); sels["madmm"] = b["sel"]
    if "ls" in by:
        a = by["ls"][0]
        row.update(ls_best=a["f_ls"], ls_time=a["time"], ls_starts=a["starts"])
        cands.append(a["f_ls"]); sels["ls"] = a["sel"]
    if "mip" in by:
        a = by["mip"][0]
        row.update(mip_solver=a["solver"], mip_value=a["value"], mip_bound=a["bound"], mip_status=a["status"], mip_time=a["time"],
                   mip_formulation=a["formulation"], mip_wall_clock=a.get("wall_clock"), mip_solver_runtime=a.get("solver_runtime"))
        if a.get("f_check") is not None:
            cands.append(a["f_check"])
    best = max(cands) if cands else float("nan")
    row["best_known"] = best
    if best == best and best != 0:
        for key in ("aadmm_ls", "pal_ls", "madmm_first_ls", "madmm_best_ls", "madmm_mean_ls", "ls_best", "aadmm_round", "pal_round", "madmm_first_raw"):
            if key in row:
                row[key + "_gap_pct"] = 100.0 * (best - row[key]) / abs(best)
        if "bound_U" in row:
            row["certified_gap_pct"] = 100.0 * (row["bound_U"] - best) / abs(row["bound_U"])
    return row, sels


def summary_line(row):
    parts = [f"{row['instance']:<28s} n={row['n']:<6d} m={row['m']:<5d}"]
    if "bound_U" in row: parts.append(f"U={row['bound_U']:.1f}")
    for k, lab in (("aadmm_ls", "AADMM+LS"), ("pal_ls", "PAL+LS"), ("madmm_first_ls", "MADMM+LS"), ("ls_best", "LS")):
        if k in row:
            g = row.get(k + "_gap_pct")
            parts.append(f"{lab}={row[k]:.1f}" + (f"({g:.2f}%)" if g is not None else ""))
    if "mip_value" in row:
        parts.append(f"{row['mip_solver']}={fmt(row['mip_value'],1)}[{row['mip_status']},{row['mip_time']:.0f}s]")
    parts.append(f"best={fmt(row['best_known'],1)}")
    if row.get("errors"): parts.append("ERRORS: " + row["errors"])
    return "  ".join(parts)


def build_tasks(args, meta, shm_name, inst):
    base = dict(shm=shm_name, n=inst.n, p=inst.p, gcat=np.concatenate(inst.groups), gsizes=inst.gsize.tolist(),
                lam=meta["lam"], norm2=meta["norm2"], delta=meta["delta"], tmax=args.time_limit)
    tasks = []
    ms = args.methods
    if "aadmm" in ms: tasks.append(dict(base, method="aadmm", tag="aadmm"))
    if "pal" in ms: tasks.append(dict(base, method="pal", tag="pal"))
    if "madmm" in ms:
        for s in range(1, args.madmm_starts + 1):
            tasks.append(dict(base, method="madmm", tag=f"madmm{s}", seed=s))
    if "ls" in ms: tasks.append(dict(base, method="ls", tag="ls", starts=args.ls_starts, seed=1, seconds=args.ls_seconds))
    if "mip" in ms and args.mip != "none" and inst.n <= args.mip_max_n:
        tasks.append(dict(base, method="mip", tag="mip", solver=args.mip, mip_time=args.mip_time, mip_threads=args.mip_threads))
    return tasks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", help="instance files (.txt / .npz)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2, help="worker processes (default: all cores)")
    ap.add_argument("--methods", default="aadmm,pal,madmm,ls,mip", help="comma list from aadmm,pal,madmm,ls,mip")
    ap.add_argument("--time-limit", type=float, default=600.0, help="wall-clock cap (s) per AADMM / PAL / MADMM-ALU run; best-so-far is returned")
    ap.add_argument("--madmm-starts", type=int, default=1, help="number of random starts of MADMM-ALU (each is a parallel task)")
    ap.add_argument("--ls-starts", type=int, default=10, help="minimum number of local-search restarts")
    ap.add_argument("--ls-seconds", type=float, default=0.0, help="keep restarting local search for this many seconds")
    ap.add_argument("--mip", default="highs", choices=["none", "highs", "cplex", "gurobi"], help="benchmark MIP solver")
    ap.add_argument("--mip-time", type=float, default=600.0, help="MIP time limit (s)")
    ap.add_argument("--mip-max-n", type=int, default=300, help="skip the MIP benchmark above this n (HiGHS linearisation has O(n^2) variables)")
    ap.add_argument("--mip-threads", type=int, default=0, help="CPLEX/Gurobi threads (0 = solver default)")
    ap.add_argument("--delta", default="lanczos", choices=["lanczos", "rigorous"],
                    help="shift for the certified bound: lambda_N from an eigensolver (+tiny margin) or ||R||_inf (unconditionally valid, weaker)")
    ap.add_argument("--npz-perm-mode", default="groups", choices=["groups", "inverse"],
                    help="npz: group k = perm[block k] ('groups') or inverse(perm)[block k] ('inverse')")
    ap.add_argument("--mem-gb", type=float, default=8.0, help="max total size of R matrices kept in shared memory at once")
    ap.add_argument("--max-inflight", type=int, default=4, help="max number of instances processed concurrently")
    ap.add_argument("--out", default="bqpgub_results", help="output prefix (writes <out>.csv, <out>.json, <out>_solutions/)")
    ap.add_argument("--blas-threads", default="1", help="BLAS threads per process (default 1)")
    ap.add_argument("--selftest", action="store_true", help="run a quick correctness test and exit")
    args = ap.parse_args()
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]
    if args.selftest:
        return selftest()
    if not args.files:
        ap.error("no input files")

    infos = sorted(((peek_n(f), f) for f in args.files))              # small instances first
    pending = deque(f for n, f in infos)
    sizes = {f: n for n, f in infos}
    budget = args.mem_gb * 1e9
    os.makedirs(args.out + "_solutions", exist_ok=True)
    ctx = get_context("spawn")
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx)
    inflight, fut2key = {}, {}
    rows, all_json = [], []
    t_all = time.perf_counter()
    print(f"{len(infos)} instance(s); workers={args.workers}; methods={','.join(args.methods)}; time-limit={args.time_limit}s", flush=True)
    try:
        while pending or inflight:
            used = sum(v["mem"] for v in inflight.values())
            while pending and len(inflight) < args.max_inflight and (not inflight or used + 8.0 * sizes[pending[0]] ** 2 <= budget):
                path = pending.popleft()
                t0 = time.perf_counter()
                try:
                    p, R, groups = load_npz(path, args.npz_perm_mode) if path.endswith(".npz") else load_txt(path)
                    n = p.size
                    allv = np.concatenate(groups)
                    assert allv.size == n and np.array_equal(np.sort(allv), np.arange(n)), "groups do not partition the variables"
                    shm = shared_memory.SharedMemory(create=True, size=R.nbytes)
                    Rs = np.ndarray(R.shape, dtype=np.float64, buffer=shm.buf)
                    Rs[:] = R
                    del R
                    asym = float(np.max(np.abs(Rs[:min(n, 2000), :min(n, 2000)] - Rs[:min(n, 2000), :min(n, 2000)].T)))
                    _symmetrize_inplace(Rs)                                   # x'Rx is unchanged by (R+R')/2
                    load_s = time.perf_counter() - t0
                    inst = GInstance(Rs, p, groups, name=os.path.basename(path))
                    t1 = time.perf_counter()
                    lam, norm2, delta = spectral_data(inst, rigorous=(args.delta == "rigorous"))
                    inst.lam, inst.norm2 = lam, norm2
                    meta = dict(name=os.path.basename(path), n=n, m=len(groups), lam=lam, norm2=norm2, delta=delta,
                                delta_mode=args.delta, load_s=load_s, spec_s=time.perf_counter() - t1, asym=asym)
                    tasks = build_tasks(args, meta, shm.name, inst)
                    futs = [pool.submit(worker, t) for t in tasks]
                    key = path
                    inflight[key] = dict(meta=meta, shm=shm, Rs=Rs, futs=futs, res=[], mem=R_bytes(n))
                    for f in futs:
                        fut2key[f] = key
                    used += R_bytes(n)
                    print(f"[load] {meta['name']}: n={n} m={len(groups)} group sizes {int(inst.gsize.min())}-{int(inst.gsize.max())} "
                          f"R asymmetry {asym:.3g} (symmetrised)  lambda_N={lam:.4g}  delta={delta:.4g}  "
                          f"load {load_s:.1f}s eig {meta['spec_s']:.1f}s; {len(tasks)} task(s)", flush=True)
                except Exception as e:
                    print(f"[skip] {path}: {type(e).__name__}: {e}", flush=True)
            if not inflight:
                continue
            # instances without any applicable task are finished immediately
            for key in [k for k, v in inflight.items() if not v["futs"]]:
                st = inflight.pop(key)
                row, sels = finalize(st["meta"], [], args); rows.append(row); all_json.append(dict(row=row, tasks=[]))
                print("[done] " + summary_line(row) + "  (no applicable task)", flush=True)
                del st["Rs"]; st["shm"].close(); st["shm"].unlink()
            if not fut2key:
                continue
            done, _ = wait(list(fut2key), return_when=FIRST_COMPLETED)
            for f in done:
                key = fut2key.pop(f)
                st = inflight[key]
                try:
                    st["res"].append(f.result())
                except Exception as e:                                         # worker crash (e.g. out of memory)
                    st["res"].append(dict(method="?", tag="?", ok=False, error=f"worker failed: {e}"))
                if len(st["res"]) == len(st["futs"]):
                    row, sels = finalize(st["meta"], st["res"], args)
                    rows.append(row)
                    all_json.append(dict(row=row, tasks=[{k: v for k, v in r.items() if k != "sel"} for r in st["res"]]))
                    if sels:
                        np.savez(os.path.join(args.out + "_solutions", st["meta"]["name"] + ".npz"), **{k: np.array(v) for k, v in sels.items()})
                    print("[done] " + summary_line(row), flush=True)
                    del st["Rs"]
                    try:
                        st["shm"].close(); st["shm"].unlink()
                    except Exception:
                        pass
                    del inflight[key]
                    write_outputs(args, rows, all_json)
    except KeyboardInterrupt:
        print("interrupted - cleaning up", flush=True)
        for f in list(fut2key):
            f.cancel()
    finally:
        for st in inflight.values():
            try:
                del st["Rs"]; st["shm"].close(); st["shm"].unlink()
            except Exception:
                pass
        pool.shutdown(wait=False, cancel_futures=True)
    write_outputs(args, rows, all_json)
    print(f"finished in {time.perf_counter() - t_all:.1f}s -> {args.out}.csv / {args.out}.json", flush=True)


def R_bytes(n):
    return 8.0 * n * n


def write_outputs(args, rows, all_json):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in sorted(rows, key=lambda r: r["n"]):
            w.writerow(r)
    json.dump(all_json, open(args.out + ".json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# =============================================================================
def selftest():
    """quick checks: symmetrisation, lambda_N, certified bound vs exact optimum, loaders, all methods"""
    import tempfile
    rng = np.random.default_rng(0)
    n, sizes = 18, [3, 4, 5, 6]
    perm = rng.permutation(n)
    groups = [perm[o:o + s] for o, s in zip(np.r_[0, np.cumsum(sizes)[:-1]], sizes)]
    B = rng.integers(-50, 51, size=(n, n)).astype(float)
    p = rng.integers(-10, 11, size=n).astype(float)
    R = B.copy(); _symmetrize_inplace(R)
    inst = GInstance(R, p, groups)
    lam, norm2, delta = spectral_data(inst)
    inst.lam, inst.norm2 = lam, norm2
    Z = np.linalg.svd(np.vstack([(np.arange(n) == g[:, None]).any(0).astype(float) for g in groups]))[2][len(groups):].T   # null-space basis
    lam_ref = float(np.linalg.eigvalsh(Z.T @ R @ Z).max())
    assert abs(lam - lam_ref) < 1e-6 * max(1, abs(lam_ref)), (lam, lam_ref)
    best = -np.inf
    for sel in itertools.product(*groups):
        idx = np.array(sel); v = p[idx].sum() + R[np.ix_(idx, idx)].sum()
        best = max(best, v)
    xP, info = run_aadmm(inst, delta, 60)
    assert info["U"] >= best - 1e-6, (info["U"], best)
    xl = local_search_swap(inst, round_groups(inst, xP))
    xb, _ = run_madmm(inst, 1, 60); xp, ip = run_pal(inst, 60)
    ls, _ = run_ls(inst, 10, 1, 0)
    assert inst.f(xl) <= best + 1e-9 and inst.f(xb) <= best + 1e-9 and inst.f(local_search_swap(inst, round_groups(inst, xp))) <= best + 1e-9
    assert ip["ascent_ok"]
    # loaders round trip (txt with 1-based groups, npz with sizes/perm)
    with tempfile.TemporaryDirectory() as d:
        f1 = os.path.join(d, "t.txt")
        with open(f1, "w") as fh:
            fh.write(f"n = {n}\nm = {len(groups)}\np = [{';'.join(repr(float(v)) for v in p)}]\nR = [{';'.join(' '.join(repr(float(v)) for v in r) for r in B)}]\n")
            for k, g in enumerate(groups):
                fh.write(f"C{k + 1} = [{' '.join(str(int(v) + 1) for v in g)}]\n")
        p2, R2, g2 = load_txt(f1)
        assert np.allclose(p2, p) and np.allclose(R2, B) and all(set(a) == set(b) for a, b in zip(g2, groups))
        f2 = os.path.join(d, "t.npz")
        order = np.concatenate(groups)
        np.savez(f2, R=B, p=p, sizes=np.array(sizes), perm=order, seed=1, bs=0)
        p3, R3, g3 = load_npz(f2)
        assert all(np.array_equal(a, b) for a, b in zip(g3, groups))
    print(f"selftest OK  (exact optimum {best:.3f}; AADMM bound {info['U']:.3f}; AADMM+LS {inst.f(xl):.3f}; "
          f"MADMM-ALU {inst.f(xb):.3f}; PAL {inst.f(xp):.3f}; lambda_N {lam:.4f})")


if __name__ == "__main__":
    main()
