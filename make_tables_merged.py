import json, os
import numpy as np
R="/home/claude/work/results"; OUT="/home/claude/work/merged"
os.makedirs(OUT,exist_ok=True)
J=lambda n: json.load(open(f"{R}/{n}.json"))
gap=lambda z,v:100.0*(z-v)/abs(z)
def fi(v): return f"{v:,.0f}".replace(",","\\,")
def ft(t):
    if t<0.01: return "$<$0.01"
    if t<10: return f"{t:.2f}"
    if t<100: return f"{t:.1f}"
    return f"{t:.0f}"
def write(name,s):
    s=s.replace("\\begin{tabular}","\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}").replace("\\end{tabular}","\\end{tabular}}")
    open(f"{OUT}/{name}","w").write(s)
S={}

# ---------------- tuning
T=J("tuning_bin")
rows=sorted({(r["rho_min"]) for r in T},key=lambda v:(v!="convex",-(v if v!="convex" else 0)))
t=["\\begin{table}[t]","\\centering",
 "\\caption{Tuning of MADMM-ALU on held-out instances (dense, $m=5$, $s=4$, seeds 101--120; none of them is used elsewhere). "
 "Entries: mean gap\\,\\% (median gap\\,\\%), \\# optimal of 20. Penalties are in units of $\\norm R_2$: $\\rho_{\\min}=c_{\\min}\\norm R_2$, "
 "$\\rho^0=\\max\\{\\rho_{\\min},c_0\\norm R_2\\}$ (so columns coincide when $c_{\\min}\\ge c_0$). "
 "`convex' means $\\rho_{\\min}=2\\max\\{\\lambda_{\\Nset},0\\}$, the smallest value for which the $x$-subproblem is convex.}",
 "\\label{tab:tuning}","\\small","\\begin{tabular}{lccc}","\\toprule","$c_{\\min}$ & $c_0=0.02$ & $c_0=0.1$ & $c_0=0.5$ \\\\","\\midrule"]
for cm in rows:
    cells=[]
    for c0 in [0.02,0.1,0.5]:
        r=[x for x in T if x["rho_min"]==cm and x["rho0"]==c0][0]
        cells.append(f"{r['mean_gap']:.1f} ({r['median_gap']:.1f}), {r['nopt']}")
    t.append(("convex" if cm=="convex" else f"{cm:g}")+" & "+" & ".join(cells)+" \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
write("table_tuning.tex","\n".join(t)+"\n")
S["tuning"]=T

# ---------------- exact
E=J("exact"); B=J("bin_exact")
meths=[("AADMM",lambda e,b:e["aadmm"]["f_round"]),("AADMM+LS",lambda e,b:e["aadmm"]["f_ls"]),
       ("MADMM-ALU",lambda e,b:b["value"]),("MADMM-ALU+LS",lambda e,b:b["f_ls"]),
       ("PAL",lambda e,b:e["madmm"]["f_round"]),("PAL+LS",lambda e,b:e["madmm"]["f_ls"]),
       ("LS (10 starts)",lambda e,b:e["ls"]["value"])]
rows5=[(e,b) for e,b in zip(E,B) if (e["m"],e["s"])==(5,4)]
t=["\\begin{table}[t]","\\centering",
 "\\caption{Instances with $n=20$ ($m=5$, $s=4$): gaps in \\% to the exact optimum $z^\\star$ (enumeration of all $1024$ feasible points) of the feasible binary solution returned by each method, without local search. "
 "`U gap' is $100(U-z^\\star)/|z^\\star|$ for the certified bound.}",
 "\\label{tab:exact}","\\small","\\begin{tabular}{rrrrrrrr}","\\toprule",
 "Seed & $z^\\star$ & U gap & AADMM & MADMM-ALU & PAL & LS (10 starts) \\\\","\\midrule"]
for e,b in rows5:
    z=e["z"]
    t.append(f"{e['seed']} & {fi(z)} & {-gap(z,e['aadmm']['U']):.1f} & {gap(z,e['aadmm']['f_round']):.1f} & {gap(z,b['value']):.1f} & {gap(z,e['madmm']['f_round']):.1f} & {gap(z,e['ls']['value']):.1f} \\\\")
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
write("table_exact.tex","\n".join(t)+"\n")
t=["\\begin{table}[t]","\\centering",
 "\\caption{Summary over all $30$ instances with $n=20$ (shapes $5\\times4$, $4\\times5$, $10\\times2$; ten seeds each); gaps in \\% to the exact optimum. "
 "`+LS' adds swap local search to the returned binary solution.}",
 "\\label{tab:exact-summary}","\\small","\\begin{tabular}{lrrrr}","\\toprule",
 "Method & mean & median & max & \\# optimal (of 30) \\\\","\\midrule"]
S["exact"]={}
for name,fn in meths:
    g=np.array([gap(e["z"],fn(e,b)) for e,b in zip(E,B)])
    S["exact"][name]=dict(mean=g.mean(),median=float(np.median(g)),max=g.max(),nopt=int((g<1e-9).sum()))
    t.append(f"{name} & {g.mean():.2f} & {np.median(g):.2f} & {g.max():.1f} & {(g<1e-9).sum()} \\\\")
ub=np.array([-gap(e["z"],e["aadmm"]["U"]) for e in E])
t+=["\\midrule",f"Certified bound $U$ (excess over $z^\\star$) & {ub.mean():.1f} & {np.median(ub):.1f} & {ub.max():.1f} & min {ub.min():.1f} \\\\","\\bottomrule","\\end{tabular}","\\end{table}"]
write("table_exact_summary.tex","\n".join(t)+"\n")
S["exact_ub"]=dict(mean=ub.mean(),median=float(np.median(ub)),min=ub.min(),max=ub.max())
S["exact_flags"]=dict(U_valid=all(e["aadmm"]["U"]>=e["z"]-1e-6 for e in E),aadmm_conv=all(e["aadmm"]["converged"] for e in E),
   pal_asc=all(e["madmm"]["ascent_ok"] for e in E),bin_conv=all(b["converged"] for b in B),
   aadmm_it=(min(e["aadmm"]["iters"] for e in E),max(e["aadmm"]["iters"] for e in E)),
   bin_it=(min(b["iters"] for b in B),max(b["iters"] for b in B)))
S["exact_shape"]={}
for sh in [(5,4),(4,5),(10,2)]:
    S["exact_shape"][str(sh)]={n:float(np.mean([gap(e["z"],fn(e,b)) for e,b in zip(E,B) if (e["m"],e["s"])==sh])) for n,fn in meths}
    S["exact_shape"][str(sh)]["ub"]=float(np.mean([-gap(e["z"],e["aadmm"]["U"]) for e in E if (e["m"],e["s"])==sh]))

# ---------------- highs
H=J("highs_final"); BH=J("bin_highs")
t=["\\begin{table}[t]","\\centering",
 "\\caption{Dense instances with a MIP reference (HiGHS on the Adams--Sherali linearisation, $100$\\,s limit; `opt': proven optimal, `TL': time limit, incumbent and dual bound reported). "
 "$U$ is the certified bound. Values of the ADMM-type methods include local search (+LS); LS-ms is multistart swap local search run for the wall-clock time of PAL+LS. "
 "All ADMM-type methods needed under one second.}",
 "\\label{tab:highs}","\\small","\\begin{tabular}{rr rrrr r rrr r}","\\toprule",
 "$n$ & seed & \\multicolumn{4}{c}{HiGHS} & $U$ & AADMM & MADMM-ALU & PAL & LS-ms \\\\","\\cmidrule(lr){3-6}",
 " & & value & status & dual bd. & time & & \\multicolumn{3}{c}{value (+LS)} & value \\\\","\\midrule"]
S["highs"]=[]
for h,b in zip(H,BH):
    hh=h["highs"]; st="opt" if hh["status"]==0 else "TL"
    t.append(f"{h['n']} & {h['seed']} & {fi(hh['value'])} & {st} & {fi(hh['dual_bound'])} & {ft(hh['time'])} & {fi(h['aadmm']['U'])} & {fi(h['aadmm']['f_ls'])} & {fi(b['f_ls'])} & {fi(h['madmm']['f_ls'])} & {fi(h['ls_matched']['value'])} \\\\")
    S["highs"].append(dict(n=h["n"],seed=h["seed"],hv=hh["value"],hd=hh["dual_bound"],st=hh["status"],U=h["aadmm"]["U"],a=h["aadmm"]["f_ls"],b=b["f_ls"],p=h["madmm"]["f_ls"],lsm=h["ls_matched"]["value"],ls10=h["ls"]["value"],
        tmax=max(h["aadmm"]["time"]+h["aadmm"]["time_ls"],b["time"]+b["time_ls"],h["madmm"]["time"]+h["madmm"]["time_ls"])))
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
write("table_highs.tex","\n".join(t)+"\n")

# ---------------- large
L=J("large"); BL=J("bin_large")
t=["\\begin{table}[t]","\\centering",
 "\\caption{Larger instances. $U$: certified bound; `cert.\\ gap' $=100(U-f)/U$ for the best feasible value $f$ over all methods. "
 "Each ADMM-type method is followed by swap local search (value and total time in seconds). LS-ms: multistart swap local search run for the wall-clock time of PAL+LS. "
 "dense: mixed-sign random BQP-GUB; cmisp: sparse CMISP-type instances with average degree $20$.}",
 "\\label{tab:large}","\\scriptsize","\\setlength{\\tabcolsep}{3pt}","\\begin{tabular}{llr r rr rr rr r r}","\\toprule",
 "Type & $n$ & seed & $U$ & \\multicolumn{2}{c}{AADMM+LS} & \\multicolumn{2}{c}{MADMM-ALU+LS} & \\multicolumn{2}{c}{PAL+LS} & LS-ms & cert.\\ gap (\\%) \\\\",
 "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}"," & & & & value & time & value & time & value & time & value & \\\\","\\midrule"]
S["large"]=[]
for l,b in zip(L,BL):
    a,p=l["aadmm"],l["madmm"]
    best=max(a["f_ls"],b["f_ls"],p["f_ls"],l["ls_matched"]["value"],l["ls"]["value"])
    cg=100*(a["U"]-best)/a["U"]
    ta,tb,tp=a["time"]+a["time_ls"],b["time"]+b["time_ls"],p["time"]+p["time_ls"]
    t.append(f"{l['kind']} & {l['n']} & {l['seed']} & {fi(a['U'])} & {fi(a['f_ls'])} & {ft(ta)} & {fi(b['f_ls'])} & {ft(tb)} & {fi(p['f_ls'])} & {ft(tp)} & {fi(l['ls_matched']['value'])} & {cg:.1f} \\\\")
    S["large"].append(dict(kind=l["kind"],n=l["n"],seed=l["seed"],U=a["U"],a=a["f_ls"],b=b["f_ls"],p=p["f_ls"],lsm=l["ls_matched"]["value"],ta=ta,tb=tb,tp=tp,
        a_rel=a["f_relaxed"],a_r=a["f_round"],p_rel=p["f_relaxed"],p_r=p["f_round"],b_raw=b["value"],cg=cg,a_it=a["iters"],a_conv=a["converged"],b_it=b["iters"],b_conv=b["converged"],lsm_t=l["ls_matched"]["time"],ascent=p["ascent_ok"],p_it=p["iters"]))
t+=["\\bottomrule","\\end{tabular}","\\end{table}"]
write("table_large.tex","\n".join(t)+"\n")
json.dump(S,open(f"{R}/summary_merged.json","w"),indent=1,default=float)
print("ok")
