"""Analytic vs simulated epistemic collapse under three strategies.

Lines: analytic_collapse.py (no fitted parameters). Markers: toy_v4 sweeps
(sweep_collapse_vs_M.py; 16 seeds, steady window = final 200 of 600 steps).
Left: P(question in collapse). Right: P(chi-collapse), chi = 0.25.
Books use the book_reground rule and fully analytic coverage b(t).
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_collapse import collapse, book_coverage, p_chi_collapse

INK, MUTED = "#0b0b0b", "#8a8984"
STRATS = {  # key -> (label, color, sim file)
    "none": ("no mitigation", "#3b3b3b", "projects/data/collapse_vs_M.json"),
    "mimesis": ("sharing secondhand (mimesis)", "#1baf7a", "projects/data/collapse_vs_M_mimesis.json"),
    "books": ("books ($\\omega$ = 1 write / 100 steps)", "#7b4fc9",
              "projects/data/collapse_vs_M_books_reground.json"),
}
CS = {0.05: ("-", "o"), 0.5: ("--", "s")}
MS_SIM = [25, 35, 50, 70, 100, 140, 200, 280, 400]
MS_TH = np.unique(np.geomspace(20, 450, 40).astype(int))
CHI = 0.25
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 10.5, "legend.frameon": False})


def theory(strategy, c, M):
    kw = {}
    if strategy == "books":
        cov = np.mean([book_coverage(M, t, c=c) for t in range(400, 601, 40)])
        kw = dict(write_p=0.05, book_frac=1.0, book_cov=cov)
    return collapse(M, c=c, strategy=strategy, **kw)["pi0"]


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for key, (label, col, path) in STRATS.items():
        sim = json.load(open(path))
        for c, (ls, mk) in CS.items():
            pi0 = np.array([theory(key, c, M) for M in MS_TH])
            axL.plot(MS_TH, pi0, color=col, ls=ls, lw=2.2)
            axR.plot(MS_TH, [p_chi_collapse(M, p, CHI) for M, p in zip(MS_TH, pi0)],
                     color=col, ls=ls, lw=2.2)
            fr = [sim[f"{c}-{M}"]["frac_collapsed"][0] for M in MS_SIM]
            pc = [sim[f"{c}-{M}"][f"p_collapse_{CHI}"][0] for M in MS_SIM]
            kw = dict(color=col, marker=mk, ls="", ms=6.5,
                      markerfacecolor="white" if mk == "s" else col, zorder=5)
            axL.plot(MS_SIM, fr, **kw)
            axR.plot(MS_SIM, pc, **kw)
    for ax in (axL, axR):
        ax.set_xscale("log")
        ax.set_xticks([25, 50, 100, 200, 400]); ax.set_xticklabels(["25", "50", "100", "200", "400"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("questions in the environment ($M$)")
        ax.set_ylim(-0.03, 1.03)
    axL.set_title("per-question collapse", loc="left")
    axL.set_ylabel("P(question in collapse)")
    axR.set_title(f"system collapse ($\\chi$ = {CHI})", loc="left")
    axR.set_ylabel("P($\\chi$-epistemic collapse)")
    h1 = [Line2D([], [], color=col, lw=2.4, label=lab) for lab, col, _ in STRATS.values()]
    h2 = [Line2D([], [], color=INK, lw=2.2, ls=ls, marker=mk, ms=6,
                 markerfacecolor="white" if mk == "s" else INK, label=f"$c$ = {c}")
          for c, (ls, mk) in CS.items()]
    h3 = [Line2D([], [], color=MUTED, lw=2.2, label="theory"),
          Line2D([], [], color=MUTED, marker="o", ls="", ms=6, label="simulation")]
    leg1 = axL.legend(handles=h1, loc="upper left")
    axL.add_artist(leg1)
    axL.legend(handles=h2 + h3, loc="upper left", bbox_to_anchor=(0.0, 0.76))
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_analytic_collapse.png", dpi=170)
    print("saved fig_analytic_collapse.png")


if __name__ == "__main__":
    main()
