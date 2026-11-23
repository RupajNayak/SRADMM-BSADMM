"""Sanity checks for FISTA on the shifted relaxation (R_delta), and comparison with SR-ADMM.
Usage: python fista_check.py CMISP-20-1.txt [more .txt instances]
Needs cmisp_runner.py in the same folder. To test YOUR implementation, replace my_fista() below
(it must return the relaxed point in P)."""
import sys, numpy as np
sys.path.insert(0, '.')
import cmisp_runner as c

def fdelta(I, x, delta):                       # f_delta(x) = f(x) + delta*(m - ||x||^2)
    return I.f(x) + delta * (I.m - x @ x)

def my_fista(I, delta, norm2, kmax=20000, eps=1e-12, trace=None):
    """Accelerated projected gradient, step 1/L, L = 2(|delta|+||R||_2); records f_delta(x_k) in `trace`."""
    L = 2.0 * (abs(delta) + norm2)
    x = I.x0.copy(); y = x.copy(); t = 1.0
    for k in range(1, kmax + 1):
        g = I.p + 2 * (I.Rs @ y - delta * y)
        xn = I.proj_P(y + g / L)
        tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = xn + ((t - 1) / tn) * (xn - x)
        step = np.linalg.norm(xn - x); x, t = xn, tn
        if trace is not None: trace.append(fdelta(I, x, delta))
        if step <= eps * max(1.0, np.linalg.norm(x)): break
    return x, L

def check(I):
    lam, delta, norm2 = c.lambda_N(I)
    print(f"\n== {I.name}: n={I.n}, m={I.m}, lambda_N={lam:.4f}, delta={delta:.4f}")
    # 1. projection sanity
    rng = np.random.default_rng(0); ok = True
    for _ in range(20):
        v = rng.normal(size=I.n) * 5; p1 = I.proj_P(v); p2 = I.proj_P(p1)
        ok &= bool(np.all(p1 >= -1e-12)) and np.allclose(np.bincount(I.gid, weights=p1, minlength=I.m), 1.0) and np.allclose(p1, p2)
    print("1. projection onto P: feasible and idempotent:", ok)
    # 2. long FISTA run as reference (and check function-value monotone trend)
    tr = []; xref, L = my_fista(I, delta, norm2, kmax=200000, eps=1e-14, trace=tr)
    Ustar = max(tr); x0 = I.x0
    print(f"2. reference run: {len(tr)} iterations, f_delta* = {Ustar:.8f}")
    # 3. O(1/k^2) bound: (U* - f(x_k)) (k+1)^2 / (2 L ||x0-x*||^2) should be <= 1
    d0 = np.linalg.norm(x0 - xref) ** 2
    if d0 > 1e-12:
        ratios = [(Ustar - fk) * (k + 1) ** 2 / (2 * L * d0) for k, fk in enumerate(tr[:2000], start=0)]
        print(f"3. max of (U*-f_k)(k+1)^2 / (2L||x0-x*||^2) over first {len(ratios)} iterations: {max(ratios):.3g}  (should be <= 1)")
    # 4. compare with SR-ADMM and with the stopping rule of c.fista
    xa, it_a, conv_a = c.sradmm(I, delta, norm2)
    xf, it_f, conv_f = c.fista(I, delta, norm2)
    for name, x in [("SR-ADMM", xa), ("FISTA (runner)", xf), ("reference", xref)]:
        U = c.certified_bound(I, x, delta)
        print(f"4. {name:15s} f_delta={fdelta(I, x, delta):.6f}  certified bound={U:.6f}  FW gap={U - fdelta(I, x, delta):.2e}")
    print(f"5. distances: ||x_FISTA - x_SRADMM|| = {np.linalg.norm(xf - xa):.3e},  ||x_FISTA - x_ref|| = {np.linalg.norm(xf - xref):.3e},  ||x_SRADMM - x_ref|| = {np.linalg.norm(xa - xref):.3e}")
    print(f"   relative gap in f_delta: SR-ADMM {(Ustar-fdelta(I,xa,delta))/abs(Ustar):.2e},  FISTA {(Ustar-fdelta(I,xf,delta))/abs(Ustar):.2e}")
    # 6. rounded and polished values
    _, fa = c.local_search(I, c.round_groupwise(I, xa)); _, ff = c.local_search(I, c.round_groupwise(I, xf))
    print(f"6. rounded + LS: SR-ADMM {fa:.4f}, FISTA {ff:.4f}, equal: {abs(fa-ff) < 1e-9}")

if __name__ == '__main__':
    for pth in sys.argv[1:]:
        check(c.Inst(c.load_instance(pth)))
