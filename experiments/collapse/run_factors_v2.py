"""Factor sweep for toy v2: one rising lever per factor, other factors at BASE."""
import json, time, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from toy_v2 import run

SEEDS = range(10)
T = 800
BASE = dict(N=20, c=0.5, sigma="unknown_first", B=5, tau_recv=20,   # agent
            M=50, lam_mean=0.003, lam_disp=0.0, s_env=0.0,          # environment
            s_ask=0.0, mean_degree=None, peer="uniform",            # communication
            alpha=0.01, n_obs_frac=1.0)                             # observation
# Baseline chosen so that communication is load-bearing at the system level:
# memory (tau_recv=20) < evidence validity (theta = ln2/0.003 = 231) and the
# per-question observation interval M/(alpha*B*N) = 50 steps > tau_recv, with
# R0 = m*tau/M = 4.95*20/50 ~ 2 on the full mesh.

FACTORS = {
    "agent":         dict(x="c", xs=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95],
                          color="N", colors=[5, 20, 100],
                          style="sigma", styles=["unknown_first", "oldest"]),
    "communication": dict(x="s_ask", xs=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                          color="mean_degree", colors=[2, 6, None],
                          style="peer", styles=["uniform", "freshest"]),
    "environment":   dict(x="lam_mean", xs=[0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
                          color="lam_disp", colors=[0.0, 1.0, 2.0],
                          style="s_env", styles=[0.0, 1.0]),
    "observation":   dict(x="alpha", xs=[0.1, 0.03, 0.01, 0.003, 0.001],
                          color="B", colors=[2, 5, 10],
                          style="n_obs_frac", styles=[1.0, 0.5]),
}


def cell(kw):
    p = {k: v for k, v in kw.items() if k != "factor"}
    r = run(T=T, record_every=5, **p)
    last = slice(-40, None)
    return {"kw": kw, "factor": kw["factor"],
            **{k: float(r[k][last].mean()) for k in ("idk", "idk_demand", "correct", "stale", "dead_q")}}


def main():
    jobs = []
    for fname, f in FACTORS.items():
        for x in f["xs"]:
            for c in f["colors"]:
                for s in f["styles"]:
                    for seed in SEEDS:
                        kw = dict(BASE); kw.update({f["x"]: x, f["color"]: c, f["style"]: s, "seed": seed, "factor": fname})
                        jobs.append(kw)
    print(len(jobs), "runs", flush=True); t0 = time.time()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=6))
    print(f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"base": BASE, "factors": FACTORS, "results": res}, open("factor_v2_results.json", "w"))


if __name__ == "__main__":
    main()
