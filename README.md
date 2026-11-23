# SRADMM-BSADMM: shifted convex relaxation and ADMM-type methods for BQP-GUB

Code, results and tables for

> H. Sharma, R. K. Nayak, N. K. Mohanty, *A shifted convex relaxation with certified bounds and computational alternatives for Boolean quadratic programming with generalized upper-bound constraints*, submitted to *Optimization Methods and Software*.

**Problem (BQP-GUB).** `max p'x + x'Rx`, exactly one variable chosen from each group, `x` binary (contains the CMISP).

| name in paper | what it is |
|---|---|
| **SR-ADMM** | ADMM on the shifted concave relaxation, CG on the null space of the group constraints, certified upper bound |
| **FISTA** | accelerated projected gradient on the same relaxation |
| **PAL** | dual-free proximal ascent |
| **BS-ADMM** | ADMM with binary-set splitting (heuristic) |
| LS, LS-ms, ILS, Greedy+LS | swap local search, time-matched multistart, iterated local search, greedy start |
| HiGHS / CPLEX / Gurobi | exact solvers, for the references only |

> **Naming.** The code and some CSV columns use the earlier names: `aadmm` = SR-ADMM, `madmm` = BS-ADMM (MADMM-ALU), `lsms` = LS-ms.

Gap to reference: `100 (f_ref - f)/f_ref`; certified gap: `100 (U - f_ref)/U`. `f_ref` is the proven optimum for n <= 100, otherwise the best value found by any compared method (not an optimum).

## Layout
| path | content |
|---|---|
| `paper_experiments/bqpgub_paper.py` | driver of the main run (version `v5-match-to`); reads `.txt` (Matlab-style) and `.npz` instances |
| `paper_experiments/bqpgub_paper_v4_main_run.py` | version `v4-identity-perm` that produced the first 98 instances |
| `paper_experiments/bqpgub.py`, `bqpgub_bin.py`, `cmisp_runner.py` | core implementations and the generators of the small random studies |
| `benchmark/` | `bqpgub_bench.py`, parallel benchmark driver with HiGHS/CPLEX/Gurobi (own README) |
| `scripts/` | `dedup_results.py`, `max_cert_gap.py`, `final_analysis.py`, `make_tables*.py`, `fista_check.py` |
| `results/main_run/` | CSVs and logs of the main run (124 instances: 101 dense, 23 large CMISP), `cmisp_small.csv` (21 small CMISP), LS-sweep files |
| `results/benchmark/` | CSVs of the benchmark driver (HiGHS runs, LS runs) |
| `results/solver_checks/` | Gurobi log and CPLEX CSV (Section 8.2) |
| `results/random_studies/` | JSON results of the shift, ablation and tuning studies |
| `tables/` | the LaTeX tables of the paper |
| `notebooks/` | Jupyter notebooks used to launch the runs |

### Main-run files
* `paper_results_dedup.csv`: first run, 98 instances (75 dense, 23 CMISP), driver v4.
* `paper_run_rest12.csv`: 12 dense instances (n = 2000-4000, instances 3-5), driver v5.
* `paper_run_missing14.csv`: 14 dense instances (n = 5000-10000), driver v5.
* `paper_results_main124.csv`: the three files merged (124 distinct instances).

Driver v5 differs from v4 only by the option `--match-to {slowest,aadmm}`; the default `slowest` is the behaviour of v4.

## Quick start
```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python3 benchmark/bqpgub_bench.py --selftest
python3 paper_experiments/bqpgub_paper.py --help
```
Gurobi and CPLEX are optional and need their own licences; the free pip editions are size-limited.

## Reproducing the main run
```bash
python3 paper_experiments/bqpgub_paper.py <instance files> --time-limit 300 --match-cap 120 --threads 1 --npz-groups perm --out paper_results.csv
python3 scripts/dedup_results.py paper_results.csv --out paper_results_dedup.csv --summarize
python3 scripts/max_cert_gap.py results/main_run/paper_results_main124.csv results/main_run/cmisp_small.csv
```
* `--time-limit 300`: wall-clock limit of SR-ADMM, PAL and BS-ADMM; `--match-cap 120`: cap of the time-matched LS-ms and ILS.
* `.npz` files store `sizes` and `perm`; `--npz-groups perm` reads group k as `perm[e_{k-1}:e_k]` (the convention of the paper); with an identity `perm` all options agree.
* Duplicate rows appeared when two jobs wrote the same file; the deduplicated data are used for the tables.
* LS-ms and ILS are time-budgeted and not bit-reproducible (two runs of `cmisp-2000-2` gave 6605 and 6623). Times were measured on one MacBook (M5) and are indicative.

## Instances
**TODO authors:** state here where the instance files are (the `.npz` files reach ~780 MB and cannot be stored on GitHub) and how the benchmark instances were generated. `.txt` format: `n = …`, `p = […]`, `R = […]`, `C1 = […]`, … (1-based).

## Citation and licence
Citation: to be added after acceptance. MIT licence, see `LICENSE`.
