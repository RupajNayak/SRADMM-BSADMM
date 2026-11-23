#!/usr/bin/env python3
"""Remove duplicate instance rows from bqpgub_paper result CSVs (e.g. after two jobs wrote the same file).

  python dedup_results.py paper_results.csv [more.csv ...] --out paper_results_dedup.csv [--keep first|last] [--summarize]

* All input files are concatenated; one row per instance name is kept (--keep first by default).
* Every instance that occurred more than once is reported with the columns whose values differ
  (relative difference > 1e-9), so that non-reproducible columns (time-budgeted LS-ms / ILS, timings) are visible.
* --summarize prints the per-size table of bqpgub_paper.py on the de-duplicated data (the numbers for the paper).
"""
import argparse, os, sys
import numpy as np, pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+"); ap.add_argument("--out", required=True)
    ap.add_argument("--keep", choices=["first", "last"], default="first")
    ap.add_argument("--summarize", action="store_true"); ap.add_argument("--latex")
    a = ap.parse_args()
    D = pd.concat([pd.read_csv(f).assign(_src=os.path.basename(f), _row=range(len(pd.read_csv(f)))) for f in a.csv], ignore_index=True)
    n0 = len(D); key = "name"
    dup = D[D.duplicated(key, keep=False)]
    print(f"{n0} rows, {D[key].nunique()} distinct instances, {n0 - D[key].nunique()} duplicate rows")
    num = [c for c in D.columns if c not in (key, "_src", "_row") and pd.api.types.is_numeric_dtype(D[c])]
    for nm, g in dup.groupby(key, sort=False):
        diff = []
        for c in num:
            v = g[c].dropna().to_numpy(float)
            if len(v) > 1 and (v.max() - v.min()) > 1e-9 * max(1.0, np.abs(v).max()): diff.append(c)
        val = [c for c in diff if not c.endswith(("_t", "_tls", "_it", "_cg", "_inner", "setup_t", "budget", "_starts"))]
        print(f"  {nm}: {len(g)} copies; differing value columns: {val if val else 'none'}; differing timing/count columns: {len(diff) - len(val)}")
    U = D.drop_duplicates(key, keep=a.keep).drop(columns=["_src", "_row"]).reset_index(drop=True)
    U.to_csv(a.out, index=False); print(f"wrote {len(U)} rows to {a.out}")
    if a.summarize:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bqpgub_paper
        bqpgub_paper.summarize(a.out, a.latex)


if __name__ == "__main__":
    main()
