# bqpgub_bench.py — parallel AADMM / MADMM-ALU / PAL benchmark for BQP-GUB

Solves `max p'x + x'Rx` s.t. one variable per group, `x` binary, with

| method | what it is |
|---|---|
| **AADMM** | ADMM for the shifted convex relaxation (fixed penalty, CG on the null space of the group constraints); also returns the **certified upper bound U** |
| **MADMM-ALU** | ADMM with binary-set splitting (heuristic, no convergence guarantee); `--madmm-starts S` random starts, each a parallel task |
| **PAL** | dual-free proximal ascent with adaptive local updates |
| **ls** | multistart one-swap local search (baseline) |
| **mip** | HiGHS, CPLEX or Gurobi, *benchmark only* (`--mip highs|cplex|gurobi`; include `mip` in `--methods`) |

AADMM, PAL and MADMM-ALU are each reported raw (rounded) and polished with swap local search (`+LS`). Parameters are those of the manuscript.

## Install (macOS, Apple Silicon)
```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy scipy            # NumPy >= 2 wheels use Apple's Accelerate BLAS
pip install cplex                  # optional (CPLEX benchmark)
python3 bqpgub_bench.py --selftest # should print "selftest OK"
```

## Run
```bash
python3 bqpgub_bench.py data/*.txt data/*.npz --workers 4
python3 bqpgub_bench.py bqpgubs-250-1.npz --mip highs --mip-time 600 --madmm-starts 5
python3 bqpgub_bench.py data/*.npz --methods aadmm,pal --time-limit 900 --workers 3
python3 bqpgub_bench.py data/*.txt --mip cplex --mip-max-n 1000       # CPLEX benchmark
```
Output: `<out>.csv` (one row per instance), `<out>.json` (per-task details), `<out>_solutions/<instance>.npz` (selected variable per group, for each method). A one-line summary per instance is printed as soon as it finishes.

Key options: `--workers` (default all cores) · `--time-limit` (seconds per AADMM/PAL/MADMM run; the best point found so far is returned and flagged `*_timed_out`) · `--mip-max-n` (skip MIP above this n) · `--delta rigorous` (certified bound with δ=‖R‖∞, unconditionally valid, weaker) · `--mem-gb` / `--max-inflight` (how many instances are held in memory at once) · `--blas-threads` (default 1 per process; for n ≥ 5000 with few workers try 4 — it speeds up the λ_N eigensolve).

## Input formats (please check these assumptions)
* `.txt`: `n = ..`, `m = ..`, `p = [a;b;..]`, `R = [row;row;..]`, `C1 = [i j ..]` … `Cm` — **1-based** indices, groups of any size.
* `.npz`: keys `R`, `p`, `sizes`, `perm` (`seed`, `bs` ignored). Group *k* = `perm[off[k]:off[k+1]]`, `off = cumsum(sizes)`. In the supplied n=250 file `perm` is the identity, so this reading could not be verified; if your generator meant the opposite convention use `--npz-perm-mode inverse`.
* `R` need not be symmetric: it is replaced by (R+Rᵀ)/2, which leaves `x'Rx` (hence every objective value) unchanged. The script prints the asymmetry it found.

## Memory (16 GB)
One dense copy of `R` (8·n² bytes: 0.8 GB at n=10000) lives in shared memory and is used by all workers without copying; loading briefly needs about twice that. In a test with n=10000 the whole machine peaked at 1.8 GB. Instances are processed smallest first; at most `--max-inflight` (default 4) are held at once and never more than `--mem-gb` (default 8) of `R` data.

## Notes
* Times are wall-clock while other tasks run, so they depend on `--workers`; for clean timings use `--workers 1`. On an M-series chip choose `--workers` ≤ number of performance cores.
* **HiGHS** uses the Adams–Sherali linearisation (≈ n²/2 extra variables): sensible for n ≲ 100–300 only (default `--mip-max-n 300`). I did not obtain a proven result for n=250 within several minutes; expect the time limit.
* **CPLEX** solves the nonconvex binary QP directly (`optimalitytarget=3`, global). The pip community edition is limited (about 1000 variables/constraints); larger n needs an academic or commercial licence. The CPLEX path was only tested on n=20.
* The certified bound `bound_U` and `certified_gap_pct = 100(U − best_known)/U` are in the CSV. With `--delta lanczos` (default) δ = λ_N plus a tiny margin computed by an eigensolver, which is not a rigorous guarantee for n > 400; use `--delta rigorous` when a mathematically unconditional bound is needed.
* Tested (Linux, Python 3.13): self-test; your two files (n=20 → optimum 450.2235 found by all methods and by HiGHS/CPLEX; n=250); synthetic n=2000 `.txt` (67 s) and n=10000 `.npz`. Not tested on macOS itself (uses the `spawn` start method and `multiprocessing.shared_memory`, both available on macOS).

* **Gurobi** (`--mip gurobi`, `pip install gurobipy`) solves the nonconvex binary QP with `NonConvex=2`. Tested only on n=20 (optimal, matches). The pip licence is size-limited and refused n=250 ("Model too large"), so larger instances need a full or academic licence.
