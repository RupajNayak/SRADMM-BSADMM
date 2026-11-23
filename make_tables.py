import json, os, sys
import numpy as np

R = "/home/claude/work/results"
OUT = "/home/claude/work/paper"
os.makedirs(OUT, exist_ok=True)


def gap(z, v):
    return 100.0 * (z - v) / abs(z)


def fmt_int(v):
    return f"{v:,.0f}".replace(",", "\\,")


def fmt_t(t):
    if t < 0.01:
        return "$<$0.01"
    if t < 10:
        return f"{t:.2f}"
    if t < 100:
        return f"{t:.1f}"
    return f"{t:.0f}"


def write(name, s):
    # scale every tabular to the text width (never enlarge)
    s = s.replace("\\begin{tabular}", "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}")
    s = s.replace("\\end{tabular}", "\\end{tabular}}")
    open(f"{OUT}/{name}", "w").write(s)


summary = {}

# ---------------------------------------------------------------- exact (n = 20)
E = json.load(open(f"{R}/exact.json"))
meths = [("AADMM", "aadmm", "f_round"), ("AADMM+LS", "aadmm", "f_ls"),
         ("MADMM-ALU", "madmm", "f_round"), ("MADMM-ALU+LS", "madmm", "f_ls"),
         ("LS (10 starts)", "ls", "value")]
rows = [r for r in E if (r["m"], r["s"]) == (5, 4)]
t = ["\\begin{table}[t]", "\\centering",
     "\\caption{Instances with $n=20$ ($m=5$ groups of size $4$): optimality gaps in \\% relative to the exact optimum $z^\\star$ "
     "obtained by enumeration of all $4^5=1024$ feasible points. 'U gap' is the gap of the certified upper bound, $(U-z^\\star)/|z^\\star|$.}",
     "\\label{tab:exact}", "\\small", "\\begin{tabular}{rrrrrrrrr}", "\\toprule",
     "Seed & $z^\\star$ & $U$ & U gap & AADMM & AADMM+LS & MADMM-ALU & MADMM-ALU+LS & LS (10 starts) \\\\", "\\midrule"]
for r in rows:
    z = r["z"]
    g = [gap(z, r[a][b]) for _, a, b in meths]
    t.append(f"{r['seed']} & {fmt_int(z)} & {fmt_int(r['aadmm']['U'])} & {gap(z, r['aadmm']['U']) * -1:.1f} & "
             + " & ".join(f"{x:.1f}" for x in g) + " \\\\")
t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("table_exact.tex", "\n".join(t) + "\n")

# summary over all 30 instances
t = ["\\begin{table}[t]", "\\centering",
     "\\caption{Summary over all 30 instances with $n=20$ (shapes $m\\times s\\in\\{5\\times4,\\ 4\\times5,\\ 10\\times2\\}$, ten seeds each). "
     "Gaps are relative to the exact optimum.}",
     "\\label{tab:exact-summary}", "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
     "Method & mean gap (\\%) & max gap (\\%) & \\# optimal (of 30) \\\\", "\\midrule"]
for name, a, b in meths:
    g = np.array([gap(r["z"], r[a][b]) for r in E])
    t.append(f"{name} & {g.mean():.2f} & {g.max():.1f} & {(g < 1e-9).sum()} \\\\")
ub = np.array([-gap(r["z"], r["aadmm"]["U"]) for r in E])
t += ["\\midrule", f"Certified bound $U$ (gap above $z^\\star$) & {ub.mean():.1f} (mean) & {ub.max():.1f} (max) & "
      f"min {ub.min():.1f} \\\\", "\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("table_exact_summary.tex", "\n".join(t) + "\n")
summary["exact"] = dict(
    valid_all=all(r["aadmm"]["U"] >= r["z"] - 1e-6 for r in E),
    conv_all=all(r["aadmm"]["converged"] for r in E),
    ascent_all=all(r["madmm"]["ascent_ok"] for r in E),
    ub_mean=float(ub.mean()), ub_median=float(np.median(ub)), ub_min=float(ub.min()), ub_max=float(ub.max()),
    stats={n: dict(mean=float(np.mean([gap(r["z"], r[a][b]) for r in E])),
                   max=float(np.max([gap(r["z"], r[a][b]) for r in E])),
                   nopt=int(sum(gap(r["z"], r[a][b]) < 1e-9 for r in E)))
           for n, a, b in meths})

# ---------------------------------------------------------------- HiGHS reference
H = json.load(open(f"{R}/highs_final.json"))
t = ["\\begin{table}[t]", "\\centering",
     "\\caption{Dense instances with a MIP reference (HiGHS, QSAP linearisation, 100\\,s limit). 'opt' = proven optimal, 'TL' = time limit "
     "(incumbent and MIP dual bound reported). $U$ is our certified bound; LS-ms is multistart swap local search run for the same time as MADMM-ALU+LS (at least 10 starts). Times in seconds.}",
     "\\label{tab:highs}", "\\small", "\\begin{tabular}{rr rrrr r rr rr r}", "\\toprule",
     "$n$ & seed & \\multicolumn{4}{c}{HiGHS} & $U$ & \\multicolumn{2}{c}{AADMM+LS} & \\multicolumn{2}{c}{MADMM-ALU+LS} & LS-ms \\\\",
     "\\cmidrule(lr){3-6}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
     " & & value & status & dual bd. & time & & value & time & value & time & value \\\\", "\\midrule"]
for r in H:
    h = r["highs"]
    st = "opt" if h["status"] == 0 else "TL"
    a, m_ = r["aadmm"], r["madmm"]
    t.append(f"{r['n']} & {r['seed']} & {fmt_int(h['value'])} & {st} & {fmt_int(h['dual_bound'])} & {fmt_t(h['time'])} & "
             f"{fmt_int(a['U'])} & {fmt_int(a['f_ls'])} & {fmt_t(a['time'] + a['time_ls'])} & {fmt_int(m_['f_ls'])} & {fmt_t(m_['time'] + m_['time_ls'])} & {fmt_int(r['ls_matched']['value'])} \\\\")
t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
write("table_highs.tex", "\n".join(t) + "\n")
summary["highs"] = [dict(n=r["n"], seed=r["seed"], hv=r["highs"]["value"], hd=r["highs"]["dual_bound"],
                         st=r["highs"]["status"], U=r["aadmm"]["U"], a=r["aadmm"]["f_ls"], m=r["madmm"]["f_ls"],
                         ls=r["ls"]["value"], lsm=r["ls_matched"]["value"]) for r in H]

# ---------------------------------------------------------------- large
if os.path.exists(f"{R}/large.json"):
    L = json.load(open(f"{R}/large.json"))
    t = ["\\begin{table}[t]", "\\centering",
         "\\caption{Larger instances without exact reference. $U$ is the certified upper bound; 'cert.\\ gap' $=100(U-f)/U$ for the best feasible value $f$ found by any method. "
         "'LS-ms' is multistart swap local search run for the same wall-clock time as MADMM-ALU+LS. Times in seconds (relaxation + rounding + local search). "
         "dense: mixed-sign random BQP-GUB; cmisp: sparse CMISP-type instances with average degree $20$.}",
         "\\label{tab:large}", "\\scriptsize", "\\setlength{\\tabcolsep}{3.5pt}",
         "\\begin{tabular}{llr r rr rr r r}", "\\toprule",
         "Type & $n$ & seed & $U$ & \\multicolumn{2}{c}{AADMM+LS} & \\multicolumn{2}{c}{MADMM-ALU+LS} & LS-ms & cert.\\ gap (\\%) \\\\",
         "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", " & & & & value & time & value & time & value & \\\\", "\\midrule"]
    for r in L:
        a, m_ = r["aadmm"], r["madmm"]
        best = max(a["f_ls"], m_["f_ls"], r["ls_matched"]["value"], r["ls"]["value"])
        cg = 100.0 * (a["U"] - best) / abs(a["U"])
        t.append(f"{r['kind']} & {r['n']} & {r['seed']} & {fmt_int(a['U'])} & {fmt_int(a['f_ls'])} & {fmt_t(a['time'] + a['time_ls'])} & "
                 f"{fmt_int(m_['f_ls'])} & {fmt_t(m_['time'] + m_['time_ls'])} & {fmt_int(r['ls_matched']['value'])} & {cg:.1f} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_large.tex", "\n".join(t) + "\n")
    summary["large"] = [dict(kind=r["kind"], n=r["n"], seed=r["seed"], U=r["aadmm"]["U"],
                             a_rel=r["aadmm"]["f_relaxed"], a_r=r["aadmm"]["f_round"], a=r["aadmm"]["f_ls"],
                             a_t=r["aadmm"]["time"] + r["aadmm"]["time_ls"], a_it=r["aadmm"]["iters"], a_conv=r["aadmm"]["converged"],
                             m_rel=r["madmm"]["f_relaxed"], m_r=r["madmm"]["f_round"], m=r["madmm"]["f_ls"],
                             m_t=r["madmm"]["time"] + r["madmm"]["time_ls"], m_it=r["madmm"]["iters"],
                             ls=r["ls"]["value"], lsm=r["ls_matched"]["value"], lsm_t=r["ls_matched"]["time"],
                             lsm_starts=r["ls_matched"]["starts"], ascent=r["madmm"]["ascent_ok"]) for r in L]

# ---------------------------------------------------------------- delta
if os.path.exists(f"{R}/delta.json"):
    D = json.load(open(f"{R}/delta.json"))
    t = ["\\begin{table}[t]", "\\centering",
         "\\caption{Effect of the shift $\\delta\\ge\\lambda_N$ on the certified bound and on the rounded solutions. "
         "Row 'mult.\\ 0' is the minimal admissible shift $\\delta=\\lambda_N$; larger shifts are $\\delta=\\lambda_N+\\mathrm{mult}\\cdot\\max\\{|\\lambda_N|,1\\}$.}",
         "\\label{tab:delta}", "\\small", "\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Type & $n$ & mult. & $\\delta$ & $U_\\delta$ (certified) & $f$(relaxed) & $f$(rounded) & $f$(rounded+LS) \\\\", "\\midrule"]
    for r in D:
        t.append(f"{r['kind']} & {r['n']} & {r['mult']:g} & {r['delta']:.1f} & {fmt_int(r['U'])} & {fmt_int(r['f_relaxed'])} & {fmt_int(r['f_round'])} & {fmt_int(r['f_ls'])} \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_delta.tex", "\n".join(t) + "\n")

# ---------------------------------------------------------------- ablation
if os.path.exists(f"{R}/ablation.json"):
    A = json.load(open(f"{R}/ablation.json"))
    variants = ["fixed rho, no accel.", "adaptive rho", "fixed rho + accel.", "adaptive rho + accel."]
    names = {"fixed rho, no accel.": "fixed $\\rho$", "adaptive rho": "adaptive $\\rho$",
             "fixed rho + accel.": "fixed $\\rho$ + accel.", "adaptive rho + accel.": "adaptive $\\rho$ + accel."}
    cases = []
    for a in A:
        key = (a["kind"], a["n"])
        if key not in cases:
            cases.append(key)
    t = ["\\begin{table}[t]", "\\centering",
         "\\caption{AADMM ablation on instances not used in the main experiments (seed 101): iterations to the stopping criterion "
         "($\\epsilon_{\\rm abs}=10^{-7}$, $\\epsilon_{\\rm rel}=10^{-6}$, at most 3000 iterations) and wall-clock time in seconds. "
         "An asterisk marks runs that hit the iteration limit.}",
         "\\label{tab:ablation}", "\\small", "\\begin{tabular}{llrrrr}", "\\toprule",
         "Type & $n$ & " + " & ".join(names[v] for v in variants) + " \\\\", "\\midrule"]
    for key in cases:
        cells = []
        for v in variants:
            a = [x for x in A if (x["kind"], x["n"]) == key and x["variant"] == v][0]
            cells.append(f"{a['iters']}{'' if a['converged'] else '*'} ({fmt_t(a['time'])})")
        t.append(f"{key[0]} & {key[1]} & " + " & ".join(cells) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_ablation.tex", "\n".join(t) + "\n")
    summary["ablation"] = A

json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
print("tables written")
