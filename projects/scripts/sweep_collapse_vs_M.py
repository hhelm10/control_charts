"""Collapse vs environment size M (Figure 1, right panel).

Cold-start toy runs (T=600, firsthand policy, Figure-1 baseline otherwise).
Over the steady-state window (final 200 steps, every 5th step) records per
snapshot: agent-level P(IDK), the fraction of questions in epistemic collapse
(no agent can answer), and from those the probability of chi-collapse,
P(fraction collapsed >= chi), pooled over snapshots and seeds.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_v4

CS = (0.05, 0.5)
STRATEGY = sys.argv[1] if len(sys.argv) > 1 else "none"   # none | mimesis | books
KW = {"none": dict(answer_policy="firsthand"),
      "mimesis": dict(answer_policy="open"),
      "books": dict(answer_policy="firsthand", books=True, book_frac=1.0,
                    book_write_p=0.05, book_write_W=5),
      "books_reground": dict(answer_policy="firsthand", books=True, book_frac=1.0,
                             book_write_p=0.05, book_write_W=5, book_reground=True)}[STRATEGY]
MS = (25, 35, 50, 70, 100, 140, 200, 280, 400)
CHIS = (0.25, 0.5, 0.75)
SEEDS = range(16)
WINDOW = 40  # snapshots (record_every=5 -> final 200 steps)


def one(job):
    c, M, s = job
    r = toy_v4.run(N=10, M=M, B=11, alpha=3 / 11, c_dec=c, sim_floor=0.5, k_ctx=3,
                   lam_mean=0.01, lam_disp=1.0, T=600, seed=s, **KW)
    return (c, M, r["idk"][-WINDOW:].tolist(), r["dead_q"][-WINDOW:].tolist(),
            float(np.mean(r["book_cov"][-WINDOW:])))


def main():
    jobs = [(c, M, s) for c in CS for M in MS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(one, jobs))
    res = {}
    for c in CS:
        for M in MS:
            rows = [o for o in out if o[0] == c and o[1] == M]
            agent = np.array([np.mean(o[2]) for o in rows])          # per seed
            frac = np.array([np.mean(o[3]) for o in rows])           # per seed
            snaps = np.concatenate([o[3] for o in rows])             # all snapshots
            seed_snaps = [np.array(o[3]) for o in rows]
            key = f"{c}-{M}"
            res[key] = {
                "agent_idk": [float(agent.mean()), float(agent.std() / np.sqrt(len(agent)))],
                "frac_collapsed": [float(frac.mean()), float(frac.std() / np.sqrt(len(frac)))],
                "book_cov": float(np.mean([o[4] for o in rows])),
            }
            for chi in CHIS:
                per_seed = np.array([np.mean(ss >= chi) for ss in seed_snaps])
                res[key][f"p_collapse_{chi}"] = [float(np.mean(snaps >= chi)),
                                                 float(per_seed.std() / np.sqrt(len(per_seed)))]
            print(f"c={c} M={M}: agent {agent.mean():.2f} frac {frac.mean():.2f} "
                  + " ".join(f"P(chi={x})={res[key][f'p_collapse_{x}'][0]:.2f}" for x in CHIS),
                  flush=True)
    json.dump(res, open("projects/data/collapse_vs_M.json" if STRATEGY == "none"
               else f"projects/data/collapse_vs_M_{STRATEGY}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
