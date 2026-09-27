"""Internal: omega-invariance certification figure. Curves share a color per
omega level; within a level, marker/linestyle distinguishes p. If omega = p/W
is the sufficient statistic, same-color curves coincide."""
import json
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLES = ["-", "--", ":", "-."]
MARKS = ["o", "s", "^", "D"]
GROUPS = {  # omega label -> (color, [(cond label, p)])
    "1 write / 100 steps": ("#6ea4e4", [("om1_p0.02", 0.02), ("om1_p0.05", 0.05),
                                        ("om1_p0.1", 0.10), ("om1_p0.2", 0.20)]),
    "5 writes / 100 steps": ("#1a5fb4", [("om5_p0.05", 0.05), ("om5_p0.1", 0.10),
                                         ("om5_p0.2", 0.20)]),
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 13,
                     "legend.fontsize": 8.5, "legend.frameon": False})

LAMS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]


def main():
    res = json.load(open("projects/data/books_omega_results.json"))
    agg = defaultdict(list)
    for r in res:
        agg[(r["label"], r["lam"])].append((r["correct"], r["idk"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for ax, (idx, ylab) in zip(axes, ((0, "P(correct)"), (1, "P(“I don’t know”)"))):
        for gname, (col, conds) in GROUPS.items():
            for k, (cond, p) in enumerate(conds):
                m = [np.mean([v[idx] for v in agg[(cond, l)]]) for l in LAMS]
                s = [np.std([v[idx] for v in agg[(cond, l)]]) / np.sqrt(len(agg[(cond, l)])) for l in LAMS]
                ax.errorbar(LAMS, m, yerr=s, color=col, lw=1.8, ls=STYLES[k],
                            marker=MARKS[k], ms=5, capsize=2.5,
                            label=f"{gname},  $p$={p:g}")
        m = [np.mean([v[idx] for v in agg[("no books", l)]]) for l in LAMS]
        ax.plot(LAMS, m, color="#9a9a9a", lw=1.8, marker="o", ms=5, label="no books")
        ax.set_xscale("log"); ax.set_xticks(LAMS)
        ax.set_xticklabels([f"{l:g}" for l in LAMS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("environment speed")
        ax.set_ylabel(ylab)
    axes[0].legend(loc="upper right", title="expected write rate ($\\omega=p/W$), attempt prob. $p$",
                   title_fontsize=8.5, ncol=1)
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_books_omega.png", dpi=170)
    print("saved fig_books_omega.png")


if __name__ == "__main__":
    main()
