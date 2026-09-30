"""Figure 1, three panels; each shows aggregate + fastest question + slowest
question (line style), for two memory-decay systems (color).

(a) P(correct) over time -- warm-started runs (every agent seeded at t=0 with
    the current truth of all 50 questions), 20-question probe panel.
(b) The failure mass broken down: P(IDK) vs P(stale) over time (marker/metric),
    same runs. correct + idk + stale = 1 within each scope.
(c) Steady-state P(IDK), agent level (solid) and system level = fraction of
    questions in epistemic collapse (dashed), as a function of the environment's breadth
    M (questions to track); toy platform, 8 seeds, cold-start T=600.

Data source for (a)/(b): WS_PREFIX env var -- ws-surr (measured-gate proxy,
default) or ws-real2 (post-fix real rerun, when available).
"""
import glob
import json
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#0b0b0b", "#8a8984"
PREFIX = os.environ.get("WS_PREFIX", "ws-surr")
SYSTEMS = {  # label -> (color, run glob, toy key prefix)
    "slow forgetting ($c$=0.05)": ("#2a78d6", f"experiments/results/{PREFIX}-c0.05-s*", "0.05"),
    "fast forgetting ($c$=0.5)": ("#c94f3d", f"experiments/results/{PREFIX}-c0.5-s*", "0.5"),
}
SCOPES = {"agg": ("all questions", "-", 2.6)}
IDK = ("i don't know",)
MS = [25, 50, 100, 200, 400]
CHI_STYLES = {0.25: ("-", "o"), 0.5: ("--", "s")}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 11, "legend.frameon": False})


def run_curves(d):
    """ts + {scope: (correct, idk) arrays} for one run."""
    fs = sorted(glob.glob(d + "/snapshots/*meta.json"),
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    if len(fs) < 5:
        return None
    ts, out = [], {k: [] for k in SCOPES}
    for f in fs:
        m = json.load(open(f))
        truths = m.get("truths", {})
        ts.append(m["step"])
        C = np.array([[r == truths.get(q) for q, r in zip(m["questions"], row)]
                      for row in m["responses"]])
        I = np.array([[any(p in r.lower() for p in IDK) for r in row] for row in m["responses"]])
        out["agg"].append((C.mean(), I.mean()))
    return np.array(ts), {k: np.array(v).T for k, v in out.items()}


def sm(y, w=3):
    """centered moving average, edges shrunk; identity for w<=1."""
    y = np.asarray(y, dtype=float)
    return np.array([y[max(0, i - w // 2):i + w // 2 + 1].mean() for i in range(len(y))])


def main():
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    axA, axB, axC = axes
    for label, (col, pat, ckey) in SYSTEMS.items():
        per = [r for d in sorted(glob.glob(pat)) if (r := run_curves(d)) is not None]
        ts = per[0][0]
        for scope, (sl, ls, lw) in SCOPES.items():
            w = 1
            cor = sm(np.mean([r[1][scope][0] for r in per], axis=0), w)
            idk = sm(np.mean([r[1][scope][1] for r in per], axis=0), w)
            axA.plot(ts, cor, color=col, lw=lw, ls=ls)
            wrong = 1 - cor
            ok = wrong > 0.02                      # conditional undefined when nothing is wrong
            axB.plot(ts[ok], idk[ok] / wrong[ok], color=col, lw=lw, ls=ls)
            axB.plot(ts[ok], (wrong[ok] - idk[ok]) / wrong[ok], color=col, lw=lw * 0.7, ls="--")
    cvm = json.load(open("projects/data/collapse_vs_M.json"))
    MSC = [25, 35, 50, 70, 100, 140, 200, 280, 400]
    for label, (col, pat, ckey) in SYSTEMS.items():
        for chi, (ls, mk) in CHI_STYLES.items():
            m = np.array([cvm[f"{ckey}-{M}"][f"p_collapse_{chi}"][0] for M in MSC])
            e = np.array([cvm[f"{ckey}-{M}"][f"p_collapse_{chi}"][1] for M in MSC])
            axC.errorbar(MSC, m, yerr=e, color=col, lw=2.4, ls=ls, marker=mk, ms=5,
                         markerfacecolor="white" if mk == "s" else col, capsize=2)
    axA.axhline(1 / 50, color=MUTED, ls=":", lw=1.4, zorder=1)
    axA.text(0.98, 1 / 50 + 0.02, "chance ($1/M$)", transform=axA.get_yaxis_transform(),
             ha="right", fontsize=11, color=MUTED)
    axA.set_title("memory decay affects accuracy", loc="left")
    axA.set_xlabel("step"); axA.set_ylabel("P(correct)")
    hsys = [Line2D([], [], color=c, lw=2.4, label=l) for l, (c, _, _) in SYSTEMS.items()]
    axA.legend(handles=hsys, loc="upper right")
    axB.set_title("staleness versus forgetting", loc="left")
    axB.set_xlabel("step"); axB.set_ylabel("share of wrong responses")
    hB = [Line2D([], [], color=INK, lw=1.8, label="P(“I don’t know”$\,|\,$wrong)"),
          Line2D([], [], color=INK, lw=1.3, ls="--", label="P(stale$\,|\,$wrong)")]
    axB.legend(handles=hB, loc="center left")
    axC.set_title("environment size causes collapse", loc="left")
    axC.set_xscale("log"); axC.set_xticks(MS)
    axC.set_xticklabels([str(M) for M in MS])
    axC.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axC.set_xlabel("questions in the environment ($M$)")
    axC.set_ylabel("P($\\chi$-epistemic collapse)")
    hC = [Line2D([], [], color=INK, lw=2.4, ls=ls, marker=mk, ms=5,
                 markerfacecolor="white" if mk == "s" else INK, label=f"$\\chi$ = {chi}")
          for chi, (ls, mk) in CHI_STYLES.items()]
    axC.legend(handles=hC, loc="upper left")
    for ax in axes:
        ax.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig1_three_panel.png", dpi=170)
    print("saved fig1_three_panel.png")


if __name__ == "__main__":
    main()
