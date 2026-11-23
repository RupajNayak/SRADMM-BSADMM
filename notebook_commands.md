# Commands used for the paper's benchmark runs (Jupyter on a MacBook)

```python
import glob, os, re, subprocess, sys
code = "/path/to/repo/benchmark"; script = code + "/bqpgub_bench.py"
d = "/path/to/instances"
files = sorted(glob.glob(d + "/bqpgubs-*.txt") + glob.glob(d + "/bqpgubs-*.npz"))
def nof(f): return int(re.search(r"bqpgubs-(\d+)-", os.path.basename(f)).group(1))
A = [f for f in files if nof(f) <= 500]
B = [f for f in files if 1000 <= nof(f) <= 2000]
C = [f for f in files if nof(f) >= 2500]
pilot = [f for f in C if f.endswith("-1.npz")]; rest = [f for f in C if f not in pilot]

def launch(name, fl, extra):            # background run; caffeinate keeps the Mac awake
    log = open(f"{code}/{name}.log", "w")
    p = subprocess.Popen([sys.executable, script, *fl, "--madmm-starts", "5",
            "--out", f"{code}/{name}", *extra], stdout=log, stderr=subprocess.STDOUT)
    subprocess.Popen(["caffeinate", "-i", "-s", "-w", str(p.pid)])
    return p

launch("resA",  A,     ["--workers", "4", "--time-limit", "60",  "--mip", "highs", "--mip-time", "300", "--mip-max-n", "100"])
launch("resB",  B,     ["--workers", "4", "--time-limit", "300", "--mip", "none"])
launch("resC1", pilot, ["--workers", "3", "--blas-threads", "4", "--time-limit", "600", "--mip", "none", "--max-inflight", "2"])
launch("resC2", rest,  ["--workers", "3", "--blas-threads", "4", "--time-limit", "600", "--mip", "none", "--max-inflight", "2"])
launch("resLS", C,     ["--methods", "ls", "--ls-seconds", "20", "--workers", "3", "--mip", "none"])
small = [f for f in files if nof(f) <= 1000]
launch("resGurobi", small, ["--workers", "1", "--methods", "ls,mip", "--mip", "gurobi", "--mip-max-n", "1000", "--mip-time", "600", "--time-limit", "60"])
launch("resCPLEX",  [f for f in files if nof(f) <= 250], ["--workers", "1", "--methods", "ls,mip", "--mip", "cplex", "--mip-max-n", "250", "--mip-time", "600", "--time-limit", "60"])
```
Run the stages one after another, not at the same time. `--methods` must contain `mip` for the exact solvers to run.
