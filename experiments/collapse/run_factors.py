"""Factor-structured sweep: one panel per factor, other factors at BASE."""
import json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from toy_collapse import run

SEEDS = range(12)
T = 1000
BASE = dict(N=20, tau_mem=30, capacity=None,                 # agent
            Q=100, p_env=0.003, frac_temporal=1.0,           # environment
            obs_zipf=0.0,                                    # flat environment at baseline
            K=5, mean_degree=None, strategy="unknown_first", # communication
            eps=0.03, obs_frac=1.0, obs_mode="random")       # observation

FACTORS = {
    "agent":         dict(x="tau_mem", xs=[5, 10, 20, 40, 80, 160],
                          color="N", colors=[5, 20, 100],
                          style="capacity", styles=[None, 25]),
    "environment":   dict(x="obs_zipf", xs=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
                          color="Q", colors=[50, 100, 125],
                          style="p_env", styles=[0.003, 0.03]),
    "communication": dict(x="K", xs=[1, 2, 5, 10, 20],
                          color="mean_degree", colors=[2, 6, None],
                          style="strategy", styles=["unknown_first", "oldest"]),
    "observation":   dict(x="eps", xs=[0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                          color="obs_frac", colors=[0.05, 0.25, 1.0],
                          style="obs_mode", styles=["random", "assigned"]),
}


def cell(kw):
    p = dict(kw)
    n_temporal = int(round(p.pop("frac_temporal") * p["Q"]))
    r = run(n_temporal=n_temporal, T=T, record_every=5, **p)
    last = slice(-40, None)   # last 200 steps
    return {"kw": kw,
            "idk": float(r["idk"][last].mean()),
            "stale": float(r["stale"][last].mean()),
            "correct": float(r["correct"][last].mean()),
            "dead_q": float(1 - r["alive_q"][last].mean()),
            "collapsed_agents": float(r["n_collapsed_agents"][last].mean() / kw["N"])}


def main():
    jobs = []
    for fname, f in FACTORS.items():
        for x in f["xs"]:
            for c in f["colors"]:
                for s in f["styles"]:
                    for seed in SEEDS:
                        kw = dict(BASE); kw.update({f["x"]: x, f["color"]: c, f["style"]: s})
                        kw["seed"] = seed; kw["factor"] = fname
                        jobs.append(kw)
    print(len(jobs), "runs")
    t0 = time.time()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, [{k: v for k, v in j.items() if k != "factor"} for j in jobs], chunksize=8))
    for j, r in zip(jobs, res):
        r["factor"] = j["factor"]
    print(f"{time.time()-t0:.0f}s")
    json.dump({"base": BASE, "factors": FACTORS, "results": res}, open("factor_results.json", "w"))


if __name__ == "__main__":
    main()
