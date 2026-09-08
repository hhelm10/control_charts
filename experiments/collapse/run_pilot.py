import json, time, itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from toy_collapse import run, time_to_collapse

SEEDS = range(12)


def cell(kw):
    r = run(**kw)
    return {"kw": kw, "final_idk": float(r["idk"][-200:].mean()),
            "final_correct": float(r["correct"][-200:].mean()),
            "final_stale": float(r["stale"][-200:].mean()),
            "alive_q": float(r["alive_q"][-200:].mean()),
            "ttc": float(time_to_collapse(r)),
            "ttc_sys": float(time_to_collapse(r, 0.05, "alive_q") if False else np.inf),
            "idk_series": r["idk"][::10].tolist(), "alive_series": r["alive_q"][::10].tolist(),
            "collapsed_agents": r["n_collapsed_agents"][::10].tolist()}


def main():
    jobs = {}
    # 1. R0 sweep x N
    jobs["r0"] = [dict(N=N, Q=100, K=5, tau_mem=tau, T=800, seed=s)
                  for N in (5, 20, 100) for tau in (5, 10, 15, 20, 25, 30, 40, 60, 100) for s in SEEDS]
    # 2. eps sweep at R0 = 0.5 and 2
    jobs["eps"] = [dict(N=20, Q=100, K=5, tau_mem=tau, eps=e, T=800, seed=s)
                   for tau in (10, 40) for e in (0, 0.01, 0.03, 0.1, 0.3, 1.0) for s in SEEDS]
    # 3. p_env x tau (all questions temporal, env obs eps=0.1)
    jobs["penv"] = [dict(N=20, Q=100, K=5, tau_mem=tau, eps=0.1, p_env=p, n_temporal=100, T=800, seed=s)
                    for tau in (5, 10, 20, 40, 80, 160) for p in (0.001, 0.003, 0.01, 0.03, 0.1) for s in SEEDS[:6]]
    # 4. strategy at several R0
    jobs["strategy"] = [dict(N=20, Q=100, K=5, tau_mem=tau, T=800, seed=s, strategy=st)
                        for st in ("uniform", "unknown_first", "oldest") for tau in (10, 15, 20, 30, 40) for s in SEEDS]
    # 5. degree at R0 = 2 (tau=40) and R0=1.5 (tau=30)
    jobs["degree"] = [dict(N=100, Q=100, K=5, tau_mem=tau, T=800, seed=s, mean_degree=d)
                      for tau in (30, 40) for d in (2, 4, 8, 16, 99) for s in SEEDS]
    t0 = time.time()
    results = {}
    with ProcessPoolExecutor() as ex:
        for name, kws in jobs.items():
            results[name] = list(ex.map(cell, kws, chunksize=4))
            print(name, len(kws), f"{time.time()-t0:.0f}s", flush=True)
    with open("pilot_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
