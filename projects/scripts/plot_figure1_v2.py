"""Figure 1 (v2 candidate): P(correct) | P(IDK) | M*(chi) vs chi.

(a) P(correct) over time, warm-started proxy runs (M=50), chance line 1/M.
(b) P(IDK) over time: agent level (solid) and system level = fraction of
    probe questions in epistemic collapse (dashed).
(c) Epistemic capacity M*(chi): the largest environment the system can
    sustain with less than a proportion chi of its questions in collapse.
    Lines: exact analytic model; markers: toy simulation (collapse_vs_M.json,
    interpolated crossing of the collapsed fraction through chi). Dotted: M=50,
    the environment of panels (a)/(b).
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_collapse import collapse

INK, MUTED = "#0b0b0b", "#8a8984"
PREFIX = os.environ.get("WS_PREFIX", "ws-surr")
SYSTEMS = {  # label -> (color, run glob, c)
    "slow forgetting ($c$=0.05)": ("#2a78d6", f"experiments/results/{PREFIX}-c0.05-s*", 0.05),
    "fast forgetting ($c$=0.5)": ("#c94f3d", f"experiments/results/{PREFIX}-c0.5-s*", 0.5),
}
IDK = ("i don't know",)
M_DEMO = 50
CHIS = np.linspace(0.02, 0.95, 40)
MS_SIM = [25, 35, 50, 70, 100, 140, 200, 280, 400]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11, "legend.frameon": False})


def run_curves(d):
    fs = sorted(glob.glob(d + "/snapshots/*meta.json"),
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    if len(fs) < 5:
        return None
    ts, rows = [], []
    for f in fs:
        m = json.load(open(f))
        truths = m.get("truths", {})
        C = np.array([[r == truths.get(q) for q, r in zip(m["questions"], row)]
                      for row in m["responses"]])
        I = np.array([[any(p in r.lower() for p in IDK) for r in row] for row in m["responses"]])
        ts.append(m["step"])
        rows.append((C.mean(), I.mean(), I.all(axis=0).mean()))
    return np.array(ts), np.array(rows).T


def mstar(c, chi):
    return brentq(lambda M: collapse(M, c=c)["pi0"] - chi, 3, 100000, xtol=0.1)


def sim_mstar(sim, c, chi):
    """environment size where the simulated collapsed fraction crosses chi (log-interp)."""
    fr = np.array([sim[f"{c}-{M}"]["frac_collapsed"][0] for M in MS_SIM])
    idx = np.flatnonzero((fr[:-1] < chi) & (fr[1:] >= chi))
    if len(idx) == 0:
        return None
    i = idx[0]
    lm = np.interp(chi, [fr[i], fr[i + 1]], np.log([MS_SIM[i], MS_SIM[i + 1]]))
    return float(np.exp(lm))


def main():
    sim = json.load(open("projects/data/collapse_vs_M.json"))
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.6, 4.4))
    for label, (col, pat, c) in SYSTEMS.items():
        per = [r for d in sorted(glob.glob(pat)) if (r := run_curves(d)) is not None]
        ts = per[0][0]
        mean = np.mean([r[1] for r in per], axis=0)
        axA.plot(ts, mean[0], color=col, lw=2.6)
        axB.plot(ts, mean[1], color=col, lw=2.6)
        axB.plot(ts, mean[2], color=col, lw=2.2, ls="--")
        axC.plot([mstar(c, x) for x in CHIS], CHIS, color=col, lw=2.6)
        for x in (0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
            m = sim_mstar(sim, c, x)
            if m is not None:
                axC.plot([m], [x], marker="o", color=col, ls="", ms=6.5,
                         markeredgecolor=INK, markeredgewidth=0.6)
    axA.axhline(1 / M_DEMO, color=MUTED, ls=":", lw=1.4, zorder=1)
    axA.text(0.98, 1 / M_DEMO + 0.02, "chance ($1/M$)", transform=axA.get_yaxis_transform(),
             ha="right", fontsize=11, color=MUTED)
    axA.set_title("memory decay affects accuracy", loc="left")
    axA.set_xlabel("step ($t$)"); axA.set_ylabel("P(correct)")
    axA.legend(handles=[Line2D([], [], color=c, lw=2.6, label=l)
                        for l, (c, _, _) in SYSTEMS.items()], loc="upper right")
    axB.set_title("forgetting leads to collapse", loc="left")
    axB.set_xlabel("step ($t$)"); axB.set_ylabel("P(“I don’t know”)")
    axB.legend(handles=[Line2D([], [], color=INK, lw=2.6, label="agent"),
                        Line2D([], [], color=INK, lw=2.2, ls="--",
                               label="system (all agents)")], loc="center right")
    for ax in (axA, axB):
        ax.set_ylim(-0.03, 1.03)
    axC.axvline(M_DEMO, color=MUTED, ls=":", lw=1.4)
    axC.text(M_DEMO * 1.06, 0.97, "$M$ = 50\n(left panels)", transform=axC.get_xaxis_transform(),
             ha="left", va="top", fontsize=11, color=MUTED)
    axC.set_xscale("log")
    axC.set_title("epistemic capacity", loc="left")
    axC.set_xlabel("environment size ($M$)")
    axC.set_ylabel("collapse proportion $\\chi$")
    axC.set_ylim(0, 1)
    axC.legend(handles=[Line2D([], [], color=MUTED, lw=2.6, label="theory"),
                        Line2D([], [], color=MUTED, marker="o", ls="", ms=6.5,
                               label="simulation")], loc="lower right")
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig1_v2.png", dpi=170)
    print("saved fig1_v2.png")


if __name__ == "__main__":
    main()
