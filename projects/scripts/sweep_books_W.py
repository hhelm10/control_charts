"""Internal: hold the write-attempt probability p constant, vary the number of
attempts W a commit requires. Under p fixed, the per-step budget tax is
identical across conditions, so any difference is purely the write RATE
omega = p/W. Same regime as books_costly_results.json."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_v4 import run

LAMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
P = 0.05
CONDS = [("no books", False, 0.0, 1)] + [(f"W={w}", True, P, w) for w in (1, 2, 5, 10, 20)]
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
    json.dump(res, open("projects/data/books_W_results.json", "w"), indent=1)
    print(f"done: {len(res)} runs")


if __name__ == "__main__":
    main()
