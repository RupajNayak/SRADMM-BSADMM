#!/usr/bin/env python3
"""
Final analysis: turns the result CSVs into the paper's tables (final_tables.tex) and prints the key statistics.

Run in Jupyter:   !python final_analysis.py "/Users/rupajnayak/Desktop/bqpgub_merged/code"
Input files (all optional except the first; missing ones are skipped):
  paper_results.csv       main run, all methods, instances with n <= 10000 (duplicates are removed)
  ls_at_aadmm_time.csv    AADMM+LS and multistart LS given exactly AADMM's time (n > 1000)
  extra_main.csv          the same for the extra dense instances (also FISTA)
  extra_ls120.csv         multistart LS, 120 s, for the extra dense instances
  lssweep_{10,30,120,300}s.csv   multistart LS with budgets of 10/30/120/300 s
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None


def rd(name):
    p = DATA / name
    if not p.exists():
        print(f"[missing] {name}"); return None
    d = pd.read_csv(p).drop_duplicates("name", keep="first")
    print(f"[ok] {name}: {len(d)} instances"); return d


M = rd("paper_results.csv")
A = rd("ls_at_aadmm_time.csv")
X = rd("extra_main.csv")
L2 = rd("extra_ls120.csv")
SW = {b: rd(f"lssweep_{b}s.csv") for b in (10, 30, 120, 300)}
if M is None:
    sys.exit("paper_results.csv is required")
fam = lambda d: np.where(d.name.str.startswith("cmisp"), "CMISP", "dense")
fmt = lambda v, k=2: "--" if (v is None or (isinstance(v, float) and np.isnan(v))) else f"{v:.{k}f}"
out = []

# ------------------------------------------------------------------ Table 1/2: main run (all methods)
M = M.copy(); M["family"] = fam(M)
vc = {"aadmm_fls": "AADMM+LS", "pal_fls": "PAL+LS", "madmm_fls": "MADMM-ALU+LS", "fista_fls": "FISTA+LS",
      "lsms_f": "LS-ms", "greedy_fls": "Greedy+LS", "ils_f": "ILS"}
M["best"] = M[list(vc)].max(axis=1)
for k, v in vc.items():
    M["gap_" + v] = (M.best - M[k]) / M.best.abs() * 100
M["cert"] = (M.aadmm_U - M.best) / M.aadmm_U.abs() * 100
for m in ("aadmm", "pal", "madmm"):
    M[m + "_tot"] = M[m + "_t"] + M[m + "_tls"]
M["ls_tot"] = M.lsms_t
cols_gap = ["AADMM+LS", "PAL+LS", "MADMM-ALU+LS", "LS-ms", "ILS"]
G = M.groupby(["family", "n"])
print("\n=== MAIN RUN: %d instances; FISTA+LS identical to AADMM+LS on %d of %d" %
      (len(M), int((abs(M.fista_fls - M.aadmm_fls) < 1e-6).sum()), len(M)))


def block(rows_fn, ncol):
    lines = []
    for f in ("dense", "CMISP"):
        if f not in M.family.values: continue
        lines.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{'dense (\\texttt{bqpgubs})' if f == 'dense' else 'CMISP'}}}}}\\\\")
        for (ff, n), g in G:
            if ff == f: lines.append(rows_fn(n, g))
        lines.append("\\midrule")
    return "\n".join(lines[:-1]) if lines else ""


def row_gap(n, g):
    return (f"{n} & {len(g)} & " + " & ".join(fmt(g["gap_" + c].mean()) for c in cols_gap) + f" & {fmt(g.cert.mean(), 1)} \\\\")


def row_time(n, g):
    return (f"{n} & {len(g)} & {fmt(g.aadmm_tot.mean(), 1)} & {fmt(g.fista_t.mean(), 1)} & {fmt(g.pal_tot.mean(), 1)} & "
            f"{fmt(g.madmm_tot.mean(), 1)} & {fmt(g.ls_tot.mean(), 1)} \\\\")


out.append("% ====== Table: mean gap to best known (main run) ======\n\\begin{table}[H]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\caption{Mean gap (\\%) to the best known value (best of all methods in the main run), by size. AADMM, PAL and MADMM-ALU are followed by swap local search. "
           "LS-ms and ILS were given the running time of PAL+LS, capped at 120\\,s. Cert.: certified gap $100(U-f_{\\rm best})/U$. FISTA+LS equals AADMM+LS on every instance and is omitted.}\n"
           "\\label{tab:main}\n\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{rrrrrrrr}\n\\toprule\n"
           "$n$ & \\# & AADMM+LS & PAL+LS & MADMM-ALU+LS & LS-ms & ILS & Cert.\\\\\n\\midrule\n" + block(row_gap, 8) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
out.append("% ====== Table: mean times (main run) ======\n\\begin{table}[H]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
           "\\caption{Mean wall-clock time (s) per instance, including rounding and local search. PAL was limited to 300\\,s, LS-ms to the time of the slowest of the others, at most 120\\,s.}\n"
           "\\label{tab:times}\n\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}{rrrrrrr}\n\\toprule\n"
           "$n$ & \\# & AADMM+LS & FISTA+LS & PAL+LS & MADMM-ALU+LS & LS-ms\\\\\n\\midrule\n" + block(row_time, 7) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
big = M[M.n > 1000]
print("PAL stopped at its time limit on %d of %d instances with n > 1000" % (int(big.pal_hit.astype(str).str.upper().eq("TRUE").sum()), len(big)))
print("PAL ascent inequality held on all runs:", bool(M.pal_ascent_ok.astype(str).str.upper().eq("TRUE").all()),
      "| AADMM converged on %d of %d" % (int(M.aadmm_conv.astype(str).str.upper().eq("TRUE").sum()), len(M)))
for lab, h in (("n<=500", M[M.n <= 500]), ("1000<=n<=2000", M[(M.n >= 1000) & (M.n <= 2000)]), ("n>=2500", M[M.n >= 2500])):
    print(f"  {lab:14s} ({len(h):3d}): " + ", ".join(f"{c} {h['gap_' + c].mean():.2f}" for c in cols_gap) + f" | cert {h.cert.mean():.1f}")
sp = (M.aadmm_tot / M.fista_t).replace([np.inf], np.nan)
print("FISTA+LS is faster than AADMM+LS by a factor %.1f (median), range %.1f-%.1f" % (sp.median(), sp.min(), sp.max()))

# ------------------------------------------------------------------ headline: AADMM+LS vs LS at the same time and at 120 s
rows = []
src = [d for d in (A, X) if d is not None]
if src:
    H = pd.concat(src, ignore_index=True).drop_duplicates("name", keep="first")
    H["family"] = fam(H)
    H["aadmm_ls"] = H.aadmm_fls; H["t_aadmm"] = H.aadmm_t + H.aadmm_tls; H["ls_same"] = H.lsms_f
    fi = pd.concat([d.set_index("name").fista_t for d in (M, X) if d is not None and "fista_t" in d]).groupby(level=0).first()
    H["t_fista"] = H.name.map(fi)
    l120 = pd.Series(dtype=float)
    Mb = M[M.budget >= 119.9].set_index("name").lsms_f
    l120 = pd.concat([Mb] + ([L2.set_index("name").lsms_f] if L2 is not None else []))
    l120 = l120[~l120.index.duplicated()]
    H["ls120"] = H.name.map(l120)
    H["ref"] = H[["aadmm_ls", "ls_same", "ls120"]].max(axis=1)
    for c in ("aadmm_ls", "ls_same", "ls120"):
        H["g_" + c] = (H.ref - H[c]) / H.ref.abs() * 100
    H["rel_same"] = (H.aadmm_ls - H.ls_same) / H.ls_same.abs() * 100
    H["rel_120"] = (H.aadmm_ls - H.ls120) / H.ls120.abs() * 100
    print("\n=== AADMM+LS vs multistart LS: %d instances with n > 1000 ===" % len(H))

    def wl(x, tol=1e-9):
        x = x.dropna(); return f"{int((x > tol).sum())}/{int((abs(x) <= tol).sum())}/{int((x < -tol).sum())}"

    def pval(x):
        x = x.dropna()
        if wilcoxon is None or len(x) < 6 or (x == 0).all(): return float("nan")
        try: return wilcoxon(x).pvalue
        except Exception: return float("nan")

    regions = [("1000<n<5000", (H.n > 1000) & (H.n < 5000)), ("n>=5000", H.n >= 5000), ("n>=6000", H.n >= 6000),
               ("n>=6000 excl. n=7500", (H.n >= 6000) & (H.n != 7500)), ("dense n>=5000", (H.n >= 5000) & (H.family == "dense")),
               ("CMISP n>=5000", (H.n >= 5000) & (H.family == "CMISP"))]
    print("AADMM+LS better/equal/worse than LS ... (mean relative difference %, Wilcoxon p)")
    for lab, mk in regions:
        h = H[mk]
        if len(h) == 0: continue
        print(f"  {lab:22s} {len(h):3d} inst | same time: {wl(h.rel_same)} ({h.rel_same.mean():+.2f}, p={pval(h.rel_same):.3g})"
              f" | 120 s: {wl(h.rel_120)} ({h.rel_120.mean():+.2f}, p={pval(h.rel_120):.3g}; n={int(h.rel_120.notna().sum())})")
    HG = H.groupby(["family", "n"])
    lines = []
    for f in ("dense", "CMISP"):
        lines.append(f"\\multicolumn{{9}}{{l}}{{\\emph{{{'dense (\\texttt{bqpgubs})' if f == 'dense' else 'CMISP'}}}}}\\\\")
        for (ff, n), g in HG:
            if ff != f: continue
            lines.append(f"{n} & {len(g)} & {fmt(g.g_aadmm_ls.mean())} & {fmt(g.g_ls_same.mean())} & {fmt(g.g_ls120.mean())} & {wl(g.rel_same)} & "
                         f"{fmt(g.t_aadmm.mean(), 1)} & {fmt(g.t_fista.mean(), 1)} & {fmt(g.ls120.notna().sum() and 120.0, 0)} \\\\")
        lines.append("\\midrule")
    out.append("% ====== Table: AADMM+LS vs local search ======\n\\begin{table}[H]\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n"
               "\\caption{AADMM+LS against multistart local search on instances with $n>1000$. Gap (\\%): mean below the best of the three compared values. "
               "LS (same time): LS given the time AADMM+LS used; LS (120\\,s): LS given 120\\,s (\\texttt{--} where not run). W/T/L: instances where AADMM+LS is better/equal/worse than LS (same time). "
               "Times in seconds; FISTA+LS is identical to AADMM+LS in quality.}\n\\label{tab:headline}\n\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n"
               "\\begin{tabular}{rrrrrrrrr}\n\\toprule\n$n$ & \\# & AADMM+LS & LS (same) & LS (120\\,s) & W/T/L & $t_{\\rm AADMM}$ & $t_{\\rm FISTA}$ & $t_{\\rm LS,120}$\\\\\n\\midrule\n"
               + "\n".join(lines[:-1]) + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n")

# ------------------------------------------------------------------ sweep
if all(SW[b] is not None for b in SW) and A is not None:
    names = SW[10].name.tolist()
    T = pd.DataFrame({"name": names, "n": SW[10].n.values})
    base = pd.concat([d for d in (A, X) if d is not None]).drop_duplicates("name").set_index("name")
    T["AADMM+LS"] = T.name.map(base.aadmm_fls)
    for b in (10, 30, 120, 300): T[f"LS {b}"] = T.name.map(SW[b].set_index("name").lsms_f)
    vals = [c for c in T.columns if c not in ("name", "n")]
    best = T[vals].max(axis=1)
    for c in vals: T[c] = (best - T[c]) / best * 100
    print("\n=== LS BUDGET SWEEP (dense, %d instances): mean gap (%%) to best of the five ===" % len(T))
    print(T.groupby("n")[vals].mean().round(2).to_string())
    lines = [f"{n} & {len(g)} & " + " & ".join(fmt(g[c].mean()) for c in vals) + " \\\\" for n, g in T.groupby("n")]
    out.append("% ====== Table: local-search budget sweep ======\n\\begin{table}[H]\n\\centering\n\\footnotesize\n"
               "\\caption{Multistart local search with budgets of 10, 30, 120 and 300\\,s against AADMM+LS (a few seconds) on dense instances: mean gap (\\%) to the best of the five values.}\n"
               "\\label{tab:sweep}\n\\begin{tabular}{rrrrrrr}\n\\toprule\n$n$ & \\# & AADMM+LS & LS 10\\,s & LS 30\\,s & LS 120\\,s & LS 300\\,s\\\\\n\\midrule\n"
               + "\n".join(lines) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")

(DATA / "final_tables.tex").write_text("\n".join(out))
print("\nLaTeX tables written to", DATA / "final_tables.tex")
