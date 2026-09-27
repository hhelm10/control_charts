"""Figure 1 (final): epistemic collapse in a REAL no-mimesis agent network.

2x4: rows = P(correct) / P(IDK); columns = agent / communication / environment /
observation; solid = agent level (mean member), dashed = system level (any
member); real gpt-4o-mini data only (f1v6-real-* runs), no shading.
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#0b0b0b", "#8a8984"
HUE = {"agent": "#2a78d6", "communication": "#1baf7a", "environment": "#eb6834", "observation": "#4a3aa7"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10.5,
                     "legend.fontsize": 7.5, "legend.frameon": False})
IDK = ("i don't know",)


def four(d):
    """(agent-IDK, agent-acc, system-IDK, system-acc) over the final 10 snapshots."""
    fs = sorted(glob.glob(d + "/snapshots/snapshot_step_*_meta.json"))[-10:]
    if len(fs) < 5:
        return None
    a_i = a_a = s_i = s_a = c = 0
    for f in fs:
        m = json.load(open(f))
        truths = m.get("truths", {})
        qs, rows = m["questions"], m["responses"]
        for j, q in enumerate(qs):
            col = [rows[i][j] for i in range(len(rows))]
            idks = [any(p in r.lower() for p in IDK) for r in col]
            accs = [r == truths.get(q) for r in col]
            a_i += np.mean(idks); a_a += np.mean(accs)
            s_i += all(idks); s_a += any(accs); c += 1
    return np.array([a_i, a_a, s_i, s_a]) / c


def main():
    real = defaultdict(list)
    for d in glob.glob("experiments/results/f1v6-real-*"):
        v = four(d)
        if v is not None:
            real[re.match(r".*/f1v6-real-(.+)-s\d+_", d).group(1)].append(v)

    PANELS = [
        ("agent", "Agent — memory decay $c$", ["c0.05", "base", "c1.0"], [0.05, 0.2, 1.0], "log"),
        ("communication", "Communication — mean degree", ["deg2", "base", "mesh"], [2, 6, 9], "linear"),
        ("environment", "Environment — mean speed $\\bar\\lambda$", ["lam.001", "base", "lam.1"], [0.001, 0.01, 0.1], "log"),
        ("observation", "Observation — env queries $E$ (of $B$=11)", ["E1", "base", "E6"], [1, 3, 6], "linear"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 6.8), sharey=True)
    for ci, (fac, title, keys, xs, scale) in enumerate(PANELS):
        hue = HUE[fac]
        for ri, (ia, isys, ylab) in enumerate(((1, 3, "P(correct)"), (0, 2, "P(“I don’t know”)"))):
            ax = axes[ri][ci]
            am = [np.mean([v[ia] for v in real[k]]) for k in keys]
            sm = [np.mean([v[isys] for v in real[k]]) for k in keys]
            ax.plot(xs, am, color=hue, lw=1.9, marker="o", ms=6, zorder=4)
            ax.plot(xs, sm, color=hue, lw=1.9, ls="--", marker="s", ms=6, markerfacecolor="white", zorder=4)
            for k, x in zip(keys, xs):
                for v in real[k]:
                    ax.scatter([x], [v[ia]], color=hue, s=14, alpha=0.55, zorder=3)
                    ax.scatter([x], [v[isys]], color=hue, s=14, alpha=0.55, zorder=3,
                               facecolor="white", linewidths=1.0)
            if scale == "log":
                ax.set_xscale("log")
            ax.set_xticks(xs)
            ax.set_xticklabels([("mesh" if (fac == "communication" and x == 9) else f"{x:g}") for x in xs])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_ylim(-0.03, 1.03)
            ax.axvline(xs[1], color=MUTED, ls=":", lw=1.0)
            if ri == 0:
                ax.set_title(title, loc="left", color=hue, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ylab)
    h = [Line2D([], [], color=INK, lw=1.9, marker="o", ms=6, label="agent level (mean member)"),
         Line2D([], [], color=INK, lw=1.9, ls="--", marker="s", ms=6, markerfacecolor="white",
                label="system level (any member)")]
    axes[0][0].legend(handles=h, loc="lower left")
    fig.suptitle("Figure 1 — epistemic collapse in a real multi-agent network (gpt-4o-mini), by factor; "
                 "agent and system levels share each axis",
                 fontsize=11, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005,
             "N=10 gpt-4o-mini agents, nomic+FAISS retrieval, no mimesis: only environment-derived entries ground an answer. "
             "Truth lives in the environment; budget $B$=11 = $K$ peer asks + $E$ env queries; all questions temporal "
             "($\\lambda_q$ log-normal, dispersion 1); verbatim relay.\n"
             "Baseline (dotted) at the median of each explored range: $c$=0.2, degree 6, $\\bar\\lambda$=0.01, $E$=3. "
             "Lines: means; small points: individual seeds (1–3 per cell). Steady state: final 100 of 500 steps, 20-question probe panel.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.065, 1, 0.93))
    fig.savefig("projects/artifacts/figure1_final.png", dpi=170)
    print("saved figure1_final.png")


if __name__ == "__main__":
    main()
