"""Tables 2 and 3 of the paper (benchmark results) from results/*.csv.
Usage: python scripts/make_benchmark_tables.py   (writes tables/table_bench.tex, tables/table_benchtime.tex and prints the band means)
Gurobi incumbents at n=200 are read from results/resGurobi_summary.txt (printed table of the Gurobi run)."""
import os, re
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "..", "results"); OUT = os.path.join(HERE, "..", "tables")
os.makedirs(OUT, exist_ok=True)
rd = lambda k: pd.read_csv(os.path.join(RES, k + ".csv"))
res = pd.concat([rd("resA"), rd("resB"), rd("resC1"), rd("resC2")], ignore_index=True).drop_duplicates("instance").set_index("instance")
res["ls20"] = rd("resLS").set_index("instance").ls_best
gur = {}
for line in open(os.path.join(RES, "resGurobi_summary.txt")).read().splitlines()[1:]:
    p = line.split()
    if "Gurobi" in p:
        i = p.index("Gurobi")
        try: gur[p[1]] = float(p[i + 1])
        except ValueError: pass
res["gur"] = pd.Series(gur)
vals = ["aadmm_ls", "pal_ls", "madmm_best_ls", "madmm_first_ls", "ls_best", "ls20", "gur"]
res["ref"] = res[vals + ["best_known"]].max(axis=1)      # best known value (optimum for n <= 100)
G = {"AADMM": "aadmm_ls", "MADMM5": "madmm_best_ls", "MADMM1": "madmm_first_ls", "PAL": "pal_ls", "LS": "ls_best", "LS20": "ls20"}
for k, c in G.items(): res["g_" + k] = 100 * (res.ref - res[c]) / res.ref
res = res.reset_index(); res["n"] = res.n.astype(int)
res["band"] = pd.cut(res.n, [0, 100, 500, 2000, 10000], labels=["20-100", "200-500", "1000-2000", "2500-10000"])
cols = ["g_AADMM", "g_MADMM5", "g_MADMM1", "g_PAL", "g_LS", "g_LS20"]
t = res.groupby("n")[cols].mean(); t["cnt"] = res.groupby("n").size()
def cell(v, bold):
    if pd.isna(v): return "--"
    return f"\\textbf{{{v:.2f}}}" if bold else f"{v:.2f}"
rows = []
for n, r in t.iterrows():
    cand = {k: r[k] for k in ["g_AADMM", "g_MADMM5", "g_PAL", "g_LS", "g_LS20"] if not pd.isna(r[k])}
    m = min(cand.values())
    cells = [cell(r[k], k in cand and abs(r[k] - m) < 1e-9) for k in cols]
    cells[2] = cell(r["g_MADMM1"], False)
    rows.append(f"{int(n)} & {int(r.cnt)} & " + " & ".join(cells) + " \\\\")
head = open(os.path.join(HERE, "..", "tables", "table_bench.tex")).read().split("\\midrule")[0] if os.path.exists(os.path.join(OUT, "table_bench.tex")) else None
tab = (head + "\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n") if head else None
if tab: open(os.path.join(OUT, "table_bench.tex"), "w").write(tab)
b = res.groupby("band", observed=True)
tb = pd.DataFrame({"cnt": b.size(), "it_min": b.aadmm_iters.min(), "it_max": b.aadmm_iters.max(), "a_t": b.aadmm_time.mean(),
                   "m_t": b.madmm_time_mean.mean(), "p_t": b.pal_time.mean(), "p_to": b.pal_timed_out.sum(), "l_t": b.ls_time.mean()})
print(res.groupby("band", observed=True)[cols].mean().round(2)); print(tb.round(2))
print("certified gap (%):", (100 * (res.bound_U - res.ref) / res.bound_U).agg(["min", "mean", "max"]).round(1).to_dict(),
      "| U >= best known on all instances:", bool((res.bound_U >= res.ref - 1e-6).all()))
