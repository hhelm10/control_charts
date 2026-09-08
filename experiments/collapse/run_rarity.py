"""Rarity sweep: Zipf-skewed distributions over which questions the environment
reveals (observation) and which questions agents ask about (demand)."""
import json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from toy_collapse import run
from run_factors import BASE, SEEDS, T

S_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0]
CONDS = {"environment": lambda s: dict(obs_zipf=s, ask_zipf=0.0),
         "demand": lambda s: dict(obs_zipf=0.0, ask_zipf=s),
         "both": lambda s: dict(obs_zipf=s, ask_zipf=s)}
COUPLINGS = ["same", "independent", "reversed"]


def cell(kw):
    p = dict(kw)
    n_temporal = int(round(p.pop("frac_temporal") * p["Q"]))
    tag = {k: p.pop(k) for k in ("cond", "s", "coupling_arm")}
    r = run(n_temporal=n_temporal, T=T, record_every=5, **p)
    last = slice(-40, None)
    # per-question IDK ordered by environment rank (0 = most common) and by demand rank
    idk_q = r["idk_by_q"]
    by_env = np.zeros_like(idk_q); by_env[r["obs_rank"]] = idk_q
    by_ask = np.zeros_like(idk_q); by_ask[r["ask_rank"]] = idk_q
    return {"kw": kw, **tag, "idk": float(r["idk"][last].mean()), "idk_demand": float(r["idk_demand"][last].mean()),
            "dead_q": float(1 - r["alive_q"][last].mean()), "correct": float(r["correct"][last].mean()),
            "idk_by_env_rank": by_env.tolist(), "idk_by_ask_rank": by_ask.tolist()}


def main():
    jobs = []
    for cname, f in CONDS.items():
        for s in S_VALUES:
            for seed in SEEDS:
                kw = dict(BASE); kw.update(f(s)); kw.update(seed=seed, rank_coupling="same", cond=cname, s=s, coupling_arm=False)
                jobs.append(kw)
    for coupling in COUPLINGS:
        for s in S_VALUES:
            for seed in SEEDS:
                kw = dict(BASE); kw.update(obs_zipf=s, ask_zipf=s, seed=seed, rank_coupling=coupling, cond=coupling, s=s, coupling_arm=True)
                jobs.append(kw)
    print(len(jobs), "runs"); t0 = time.time()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=8))
    print(f"{time.time()-t0:.0f}s")
    json.dump({"base": BASE, "results": res}, open("rarity_results.json", "w"))


if __name__ == "__main__":
    main()
