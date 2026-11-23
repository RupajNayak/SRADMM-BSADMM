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

## Files (flat layout: all files are in the repository root)

The same files, arranged in folders (`paper_experiments/`, `benchmark/`, `scripts/`, `results/`, `tables/`, `notebooks/`), are also available as `repo_final.zip`.
| files | content |
|---|---|
| `bqpgub_paper.py` | driver of the main run (version `v5-match-to`); reads `.txt` (Matlab-style) and `.npz` instances |
| `bqpgub_paper_v4_main_run.py` | version `v4-identity-perm` that produced the first 98 instances |
| `bqpgub.py`, `bqpgub_bin.py`, `cmisp_runner.py` | core implementations and the generators of the small random studies |
| `bqpgub_bench.py`, `benchmark_README.md`, `notebook_commands.md`, `run_example.sh` | parallel benchmark driver with HiGHS/CPLEX/Gurobi and its instructions |
| `dedup_results.py`, `max_cert_gap.py`, `final_analysis.py`, `make_benchmark_tables.py`, `make_tables*.py`, `fista_check.py` | scripts for deduplication, statistics and tables |
| `paper_results_dedup.csv`, `paper_run_rest12.csv`, `paper_run_missing14.csv`, `paper_results_main124.csv`, `cmisp_small.csv`, `paper_run_*.log` | main run: 124 instances (101 dense, 23 large CMISP) and 21 small CMISP |
| `lssweep_*.csv`, `ls_at_aadmm*.csv/.log` | local-search budget sweep and equal-time runs |
| `resA.csv`, `resB.csv`, `resC1.csv`, `resC2.csv`, `resLS.csv` | benchmark-driver results (HiGHS and local-search runs) |
| `resCPLEX.csv`, `resGurobi.log` | CPLEX and Gurobi runs (Section 8.2) |
| `ablation.json`, `delta.json`, `exact.json`, `highs*.json`, `large.json`, `bin_*.json`, `tuning_bin.json`, `summary*.json` | results of the random studies (shift, ablation, tuning) |
| `final_tables_main.tex`, `final_tables_headline.tex`, `table_*.tex` | the LaTeX tables of the paper |
| `bqpgub_paper_runs.ipynb`, `bqpgub_reference_runs.ipynb` | notebooks used to launch the runs |

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
python3 bqpgub_bench.py --selftest
python3 bqpgub_paper.py --help
```
Gurobi and CPLEX are optional and need their own licences; the free pip editions are size-limited.

## Reproducing the main run
```bash
python3 bqpgub_paper.py <instance files> --time-limit 300 --match-cap 120 --threads 1 --npz-groups perm --out paper_results.csv
python3 dedup_results.py paper_results.csv --out paper_results_dedup.csv --summarize
python3 max_cert_gap.py paper_results_main124.csv cmisp_small.csv
```
* `--time-limit 300`: wall-clock limit of SR-ADMM, PAL and BS-ADMM; `--match-cap 120`: cap of the time-matched LS-ms and ILS.
* `.npz` files store `sizes` and `perm`; `--npz-groups perm` reads group k as `perm[e_{k-1}:e_k]` (the convention of the paper); with an identity `perm` all options agree.
* Duplicate rows appeared when two jobs wrote the same file; the deduplicated data are used for the tables.
* LS-ms and ILS are time-budgeted and not bit-reproducible (two runs of `cmisp-2000-2` gave 6605 and 6623). Times were measured on one MacBook (M5) and are indicative.

## Instances
**TODO authors:** state here where the instance files are (the `.npz` files reach ~780 MB and cannot be stored on GitHub) and how the benchmark instances were generated. `.txt` format: `n = …`, `p = […]`, `R = […]`, `C1 = […]`, … (1-based).

## Citation and licence
Citation: to be added after acceptance. MIT licence, see `LICENSE`.
