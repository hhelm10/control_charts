"""Candidate Figure 1: epistemic collapse as a function of question speed.

REAL data, no new runs: within each real gpt-4o-mini run, the probe questions
have heterogeneous revision rates (lambda_q log-normal, dispersion 1). Each
(question, seed) contributes one point: x = realized revision rate over the
500-step run, y = steady-state response decomposition over the final 100
steps x 10 agents. Panels: P(correct), P(IDK), P(stale) (sum to 1). Hue =
memory decay (slow c=0.05 / baseline c=0.2 / fast c=1.0); dots = points,
lines = geometric-bin means. Questions with zero realized revisions plot in
the "static" bin at the left edge."""
import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#0b0b0b", "#8a8984"
CELLS = {  # cell -> (label, color, lw)
    "c0.05": ("slow forgetting ($c$=0.05)", "#2a78d6", 3.0),
    "base": ("baseline ($c$=0.2)", "#8a8984", 2.2),
    "c1.0": ("fast forgetting ($c$=1.0)", "#c94f3d", 1.8),
}
IDK = ("i don't know",)
STATIC_X = 1 / 1500          # plotting position for zero-revision questions
BINS = np.geomspace(1 / 600, 0.3, 7)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 12, "axes.labelsize": 12,
                     "legend.fontsize": 9, "legend.frameon": False})


def perq(cell):
    pts = []
    for d in glob.glob(f"experiments/results/f1v6-real-{cell}-s*"):
        fs = sorted(glob.glob(d + "/snapshots/*meta.json"),
                    key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
        if len(fs) < 20:
            continue
        last = json.load(open(fs[-1]))
        revs = {q: (int(m.group(1)) if (m := re.search(r"\[rev (\d+)\]$", t)) else 0)
                for q, t in last["truths"].items()}
        T = int(re.search(r"step_(\d+)", fs[-1]).group(1))
        agg = {q: [0.0, 0.0, 0] for q in last["questions"]}
        for f in fs[-10:]:
            m = json.load(open(f))
            for j, q in enumerate(m["questions"]):
                col = [row[j] for row in m["responses"]]
                agg[q][0] += np.mean([r == m["truths"].get(q) for r in col])
                agg[q][1] += np.mean([any(p in r.lower() for p in IDK) for r in col])
                agg[q][2] += 1
        for q, (c, i, n) in agg.items():
            pts.append((revs[q] / T, c / n, i / n))
    return np.array(pts)


def binned(x, y):
    """means over geometric bins, with a leading static bin for x == 0."""
    bx, by = [], []
    if (x == 0).any():
        bx.append(STATIC_X); by.append(y[x == 0].mean())
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (x > 0) & (x >= lo) & (x < hi)
        if m.sum() >= 3:
            bx.append(np.sqrt(lo * hi)); by.append(y[m].mean())
    return np.array(bx), np.array(by)


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), sharex=True, sharey=True)
    for cell, (label, col, lw) in CELLS.items():
        p = perq(cell)
        lam, cor, idk = p[:, 0], p[:, 1], p[:, 2]
        xs = np.where(lam == 0, STATIC_X, lam)
        for ax, y in zip(axes, (cor, idk, 1 - cor - idk)):
            jitter = xs * np.exp(0.06 * np.random.default_rng(0).standard_normal(len(xs)))
            ax.scatter(jitter, y, s=14, color=col, alpha=0.30, linewidths=0, zorder=2)
            bx, by = binned(lam, y)
            ax.plot(bx, by, color=col, lw=lw, marker="o", ms=5, zorder=4,
                    label=label if ax is axes[0] else None)
    for ax, name in zip(axes, ("P(correct)", "P(“I don’t know”)", "P(stale)")):
        ax.set_title(name, loc="left")
        ax.set_xscale("log")
        ax.set_xlabel("question revision rate $\\hat\\lambda_q$ (per step)")
        ax.set_ylim(-0.03, 1.03)
        ticks = [STATIC_X, 1e-2, 1e-1]
        ax.set_xticks(ticks)
        ax.set_xticklabels(["static", "$10^{-2}$", "$10^{-1}$"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axes[0].set_ylabel("steady-state fraction of agents")
    axes[0].legend(loc="center left", title="memory decay")
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_collapse_vs_speed.png", dpi=170)
    print("saved fig_collapse_vs_speed.png")


if __name__ == "__main__":
    main()
