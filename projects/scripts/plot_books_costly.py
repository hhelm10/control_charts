"""Costly-writing books, two panels: P(correct) left, P(IDK) right, vs
environment speed. Conditions labeled by expected write rate omega = p/W
(completed writes per agent per 100 steps). Society: no mimesis + book reads;
the book is fed only by costly writes (attempt w.p. p, whole budget per
attempt, commit on the W-th attempt)."""
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GREY = "#0b0b0b", "#8a8984", "#9a9a9a"
CONDS = {  # results-file label -> (display label, color); omega = 100*p/W
    "easy writing (lag ~10)":    ("10 writes / 100 steps", "#2a78d6"),
    "medium writing (lag ~100)": ("1 write / 100 steps", "#6ea4e4"),
    "hard writing (lag ~500)":   ("0.2 writes / 100 steps", "#a9c8ef"),
    "no books":                  ("no books", "#9a9a9a"),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 13,
                     "legend.fontsize": 9, "legend.frameon": False})

LAMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def main():
    res = json.load(open("projects/data/books_costly_results.json"))
    agg = defaultdict(list)
    for r in res:
        agg[(r["label"], r["lam"])].append((r["correct"], r["idk"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for ax, (idx, ylab) in zip(axes, ((0, "P(correct)"), (1, "P(“I don’t know”)"))):
        for cond, (label, col) in CONDS.items():
            m = [np.mean([v[idx] for v in agg[(cond, l)]]) for l in LAMS]
            s = [np.std([v[idx] for v in agg[(cond, l)]]) / np.sqrt(len(agg[(cond, l)])) for l in LAMS]
            ax.errorbar(LAMS, m, yerr=s, color=col, lw=2.0, marker="o", ms=5.5,
                        capsize=2.5, label=label)
        ax.set_xscale("log"); ax.set_xticks(LAMS)
        ax.set_xticklabels([f"{l:g}" for l in LAMS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("environment speed")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper right", title="expected write rate", title_fontsize=9)
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_books_costly.png", dpi=170)
    print("saved fig_books_costly.png")


if __name__ == "__main__":
    main()
