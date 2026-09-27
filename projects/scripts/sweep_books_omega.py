"""Internal: certify omega = p/W as the sufficient statistic of costly writing.
Hold omega fixed and vary p (the attempt probability, which is also the budget
tax: an attempt burns the writer's whole step). Two omega levels:
  omega = 1 write / 100 steps:  (p,W) in (0.02,2) (0.05,5) (0.10,10) (0.20,20)
  omega = 5 writes / 100 steps: (p,W) in (0.05,1) (0.10,2) (0.20,4)
If curves within an omega group coincide, p has no effect beyond the rate."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_v4 import run

LAMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
CONDS = [("no books", False, 0.0, 1)]
CONDS += [(f"om1_p{p:g}", True, p, w) for p, w in ((0.02, 2), (0.05, 5), (0.10, 10), (0.20, 20))]
CONDS += [(f"om5_p{p:g}", True, p, w) for p, w in ((0.05, 1), (0.10, 2), (0.20, 4))]
SEEDS = range(10)
TAIL = 200  # steps of the T=600 run averaged (recorded every 5)


def one(job):
    label, books, p, W, lam, seed = job
    r = run(N=10, M=200, B=11, alpha=1 / 11, c_dec=0.05, chi=0.5, k_ctx=3,
            lam_mean=lam, lam_disp=1.0, answer_policy="firsthand",
            books=books, book_frac=1.0, book_write_p=p, book_write_W=W,
            T=600, seed=seed)
    n = TAIL // 5
    return {"label": label, "lam": lam, "seed": seed,
            "idk": float(np.mean(r["idk"][-n:])),
            "correct": float(np.mean(r["correct"][-n:])),
            "stale": float(np.mean(r["stale"][-n:]))}


def main():
    jobs = [(lb, b, p, w, lam, s) for lb, b, p, w in CONDS for lam in LAMS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, jobs))
    json.dump(res, open("projects/data/books_omega_results.json", "w"), indent=1)
    print(f"done: {len(res)} runs")


if __name__ == "__main__":
    main()
