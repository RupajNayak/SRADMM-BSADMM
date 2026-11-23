#!/usr/bin/env python3
"""Maximum / mean certified gap over all instances.
  python max_cert_gap.py paper_results_dedup.csv paper_run_rest12.csv cmisp_small.csv [...]
Certified gap = 100*(U - ref)/U, U = aadmm_U, ref = best value of all compared methods (column 'ref' if present,
else the maximum over the *_fls / lsms_f / ils_f / ls10 / exact columns that exist)."""
import sys, numpy as np, pandas as pd
D = pd.concat([pd.read_csv(f) for f in sys.argv[1:]], ignore_index=True).drop_duplicates("name", keep="first")
vals = [c for c in D.columns if c.endswith("_fls") or c in ("lsms_f", "ils_f", "ls10", "ls_same", "exact", "greedy_f")]
D["ref_calc"] = D[vals].apply(pd.to_numeric, errors="coerce").max(axis=1)
if "ref" in D: D["ref_calc"] = np.fmax(D["ref_calc"], pd.to_numeric(D["ref"], errors="coerce"))
D["cg"] = 100 * (D["aadmm_U"] - D["ref_calc"]) / D["aadmm_U"]
print(len(D), "distinct instances; columns used for the reference:", vals)
print("MAX certified gap: %.2f%%  (%s)" % (D.cg.max(), D.loc[D.cg.idxmax(), "name"]))
print("mean %.2f%%, median %.2f%%" % (D.cg.mean(), D.cg.median()))
print("\nlargest five:\n", D.nlargest(5, "cg")[["name", "n", "cg"]].to_string(index=False))
print("\nby size (n, count, mean, max):")
print(D.groupby("n").cg.agg(["count", "mean", "max"]).round(2).to_string())
print("\nrange of the size means: %.1f-%.1f%%" % (D.groupby("n").cg.mean().min(), D.groupby("n").cg.mean().max()))
