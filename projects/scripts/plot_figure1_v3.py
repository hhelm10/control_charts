"""Figure 1 (v3 candidate): all panels from the same (toy) system.

(a) P(correct) over time, warm start, M = 50.
(b) collapse proportion chi(t) over time, warm start, for several environment
    sizes M (lightness); curves level off at their steady-state value.
(c) steady-state collapse proportion vs environment size (the capacity curve):
    theory lines + simulation markers; the M's shown in (b) are marked with the
    same shading, so each plateau in (b) is a point in (c).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_v4
from analytic_collapse import collapse

INK, MUTED = "#0b0b0b", "#8a8984"
SYSTEMS = {0.05: ("slow forgetting ($c$=0.05)", "#2a78d6"),
           0.5: ("fast forgetting ($c$=0.5)", "#c94f3d")}
M_TIME = {0.05: [100, 400], 0.5: [100, 400]}   # sizes shown in (b), shared
M_DEMO = 50
SEEDS = range(16)
T = 200
M_TH = np.unique(np.geomspace(15, 3000, 60).astype(int))
MS_SIM = [25, 35, 50, 70, 100, 140, 200, 280, 400]
BASE = dict(N=10, B=11, alpha=3 / 11, sim_floor=0.5, k_ctx=3, lam_mean=0.01, lam_disp=1.0,
            answer_policy="firsthand", warm_start=True, record_every=2, T=T)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11, "legend.frameon": False})


def shade(col, i, n):
    """darker = larger M."""
    r, g, b = to_rgb(col)
    w = 0.55 * (1 - i / max(n - 1, 1))
    return (r + (1 - r) * w, g + (1 - g) * w, b + (1 - b) * w)


def run(job):
    c, M, s = job
    r = toy_v4.run(M=M, c_dec=c, seed=s, **BASE)
    return job, np.array(r["t"]), np.array(r["correct"]), np.array(r["dead_q"])


def main():
    jobs = [(c, M, s) for c in SYSTEMS for M in sorted(set(M_TIME[c] + [M_DEMO])) for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(run, jobs))
    res = {}
    for (c, M, s), t, cor, dq in out:
        res.setdefault((c, M), []).append((cor, dq))
        ts = np.concatenate([[0], t])
    sim = json.load(open("projects/data/collapse_vs_M.json"))

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.6, 4.4))
    labels = []
    for c, (label, col) in SYSTEMS.items():
        cor = np.mean([r[0] for r in res[(c, M_DEMO)]], axis=0)
        axA.plot(ts, np.concatenate([[1.0], cor]), color=col, lw=2.6, label=label)
        for i, M in enumerate(M_TIME[c]):
            dq = np.mean([r[1] for r in res[(c, M)]], axis=0)
            sh = shade(col, i, len(M_TIME[c]))
            axB.plot(ts, np.concatenate([[0.0], dq]), color=sh, lw=2.4)
            labels.append([float(np.mean(dq[-10:])), f"$M$={M}", sh])
            # the same M on the capacity curve
            axC.plot([M], [collapse(M, c=c)["pi0"]], marker="o", ms=10, color=sh,
                     markeredgecolor=INK, markeredgewidth=1.0, zorder=6)
        axC.plot(M_TH, [collapse(M, c=c)["pi0"] for M in M_TH], color=col, lw=2.6, zorder=3)
        axC.plot(MS_SIM, [sim[f"{c}-{M}"]["frac_collapsed"][0] for M in MS_SIM], ls="",
                 marker="s", ms=5, color="white", markeredgecolor=col, markeredgewidth=1.4, zorder=5)
    labels.sort(key=lambda l: l[0])              # de-overlap end labels
    for i in range(1, len(labels)):
        labels[i][0] = max(labels[i][0], labels[i - 1][0] + 0.055)
    for y, txt, sh in labels:
        axB.text(T * 1.01, y, txt, color=sh, fontsize=10, va="center")
    axA.axhline(1 / M_DEMO, color=MUTED, ls=":", lw=1.4, zorder=1)
    axA.text(0.98, 1 / M_DEMO + 0.02, "chance ($1/M$)", transform=axA.get_yaxis_transform(),
             ha="right", fontsize=11, color=MUTED)
    axA.set_title(f"memory decay affects accuracy", loc="left")
    axA.set_xlabel("step ($t$)"); axA.set_ylabel(f"P(correct), $M$ = {M_DEMO}")
    axA.legend(loc="upper right")
    axB.set_title("collapse over time", loc="left")
    axB.set_xlabel("step ($t$)"); axB.set_ylabel("collapse proportion $\\chi$")
    axB.set_xlim(0, T * 1.18)
    axC.set_title("epistemic capacity", loc="left")
    axC.set_xscale("log")
    axC.set_xlabel("environment size ($M$)"); axC.set_ylabel("collapse proportion $\\chi$")
    axC.legend(handles=[Line2D([], [], color=MUTED, lw=2.6, label="theory"),
                        Line2D([], [], color=MUTED, marker="s", ls="", ms=5,
                               markerfacecolor="white", label="simulation"),
                        Line2D([], [], color=MUTED, marker="o", ls="", ms=9,
                               markeredgecolor=INK, label="sizes in (b)")], loc="upper left")
    for ax in (axA, axB, axC):
        ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig1_v3.png", dpi=170)
    print("saved fig1_v3.png")


if __name__ == "__main__":
    main()
