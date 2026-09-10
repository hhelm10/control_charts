"""How can environment speed reach P(IDK) in the decay model (v3)?

Without a coupling mechanism it cannot: retrievability never consults the truth
process (verified: identical trajectories across lambda at fixed seed). The
repo-faithful coupling is contradiction-induced abstention (conflict=True in
toy_v3): the database never deletes, so a superseded copy stays retrievable until
its own relevance decays; retrieved context containing disagreeing values makes
the agent abstain. A faster world makes successive copies disagree more often.

Sweep: lambda x tau (decay lifetimes), conflict on, plus a conflict-off
reference; produces fig_v3_speed_conflict.png and speed_conflict_results.json.
"""
import json, time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from toy_v3 import run

SEEDS = range(10)
T = 800
LN2 = float(np.log(2.0))
LAMS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
TAUS = [15, 20, 40]
BASE = dict(N=20, theta_ret=0.5, sigma="unknown_first", B=5, M=50,
            lam_disp=0.0, s_env=0.0, s_ask=0.0, mean_degree=None, peer="uniform",
            alpha=0.01, n_obs_frac=1.0, merge="evidence")


def cell(kw):
    r = run(T=T, record_every=5, **kw)
    last = slice(-40, None)
    return {"kw": kw, **{k: float(r[k][last].mean())
                         for k in ("idk", "correct", "stale", "dead_q", "conflicted")}}


def main():
    jobs = []
    for lam in LAMS:
        for tau in TAUS:
            for seed in SEEDS:
                jobs.append(dict(BASE, c_dec=round(LN2 / tau, 4), lam_mean=lam,
                                 conflict=True, seed=seed))
        for seed in SEEDS:  # conflict-off reference at tau = 20
            jobs.append(dict(BASE, c_dec=round(LN2 / 20, 4), lam_mean=lam,
                             conflict=False, seed=seed))
    print(len(jobs), "runs", flush=True); t0 = time.time()
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=4))
    print(f"{time.time()-t0:.0f}s", flush=True)
    json.dump({"base": BASE, "results": res}, open("speed_conflict_results.json", "w"))


if __name__ == "__main__":
    main()
