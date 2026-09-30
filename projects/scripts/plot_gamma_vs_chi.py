"""Epistemic efficiency Gamma(chi) = M*_strategy(chi) / M*_none(chi) over the full
range of the collapse proportion chi, for mimesis and (steady-state) books.

Lines: closed forms
  mimesis  Gamma = x / (1 - e^{-x}),  x = K ln(1/chi) / (N E)
  books    Gamma = (1 - p_w)(1 + f K / (E p*)) tau_b / tau,  p* = chi^{1/N}
Markers: exact birth-death model (analytic_collapse.py), M* by root finding.
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_collapse import collapse

INK, MUTED = "#0b0b0b", "#8a8984"
N, B, ALPHA, K_CTX, SF = 10, 11, 3 / 11, 3, 0.5
E, K = ALPHA * B, (1 - ALPHA) * B
PW, F = 0.05, 1.0
CS = {0.05: "#2a78d6", 0.5: "#c94f3d"}
CHI_EX = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CACHE = "projects/data/gamma_vs_chi_exact.json"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 10.5, "legend.frameon": False})


def mstar(c, chi, strat, **kw):
    return brentq(lambda M: collapse(M, c=c, strategy=strat, **kw)["pi0"] - chi,
                  3, 100000, xtol=0.1)


def exact(job):
    c, chi = job
    base = mstar(c, chi, "none")
    mim = mstar(c, chi, "mimesis")
    bk = mstar(c, chi, "books", write_p=PW, book_frac=F, book_cov=1.0)
    return f"{c}-{chi}", (mim / base, bk / base)


def gamma_mim(chi):
    x = K * math.log(1 / chi) / (N * E)
    return x / (1 - math.exp(-x))


def gamma_books(chi, c):
    D = math.floor(math.log(1 / SF) / c)
    p = chi ** (1 / N)
    tau = D + 1 + K_CTX / (B * (ALPHA + (1 - ALPHA) * (1 - p)))
    tau_b = D + 1 + K_CTX / (B * (1 - PW) * (ALPHA + (1 - ALPHA) * F))
    return (1 - PW) * (1 + F * K / (E * p)) * tau_b / tau


def main():
    if os.path.exists(CACHE):
        ex = json.load(open(CACHE))
    else:
        with ProcessPoolExecutor() as pool:
            ex = dict(pool.map(exact, [(c, chi) for c in CS for chi in CHI_EX]))
        json.dump(ex, open(CACHE, "w"), indent=1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))
    chis = np.linspace(0.02, 0.98, 200)
    axL.plot(chis, [gamma_mim(x) for x in chis], color="#1baf7a", lw=2.6)
    for c, col in CS.items():
        axR.plot(chis, [gamma_books(x, c) for x in chis], color=col, lw=2.6)
        for chi in CHI_EX:
            gm, gb = ex[f"{c}-{chi}"]
            axL.plot([chi], [gm], marker="o", color=col, ls="", ms=6.5, markeredgecolor=INK,
                     markeredgewidth=0.6)
            axR.plot([chi], [gb], marker="o", color=col, ls="", ms=6.5, markeredgecolor=INK,
                     markeredgewidth=0.6)
    for ax in (axL, axR):
        ax.axhline(1, color=MUTED, ls=":", lw=1.4)
        ax.set_xlabel("collapse proportion $\\chi$")
        ax.set_xlim(0, 1)
    axL.set_title("sharing secondhand information", loc="left")
    axL.set_ylabel("epistemic efficiency $\\Gamma(\\chi)$")
    axR.set_title("books (full coverage)", loc="left")
    axR.set_ylabel("epistemic efficiency $\\Gamma(\\chi)$")
    h = [Line2D([], [], color=MUTED, lw=2.6, label="closed form"),
         Line2D([], [], color=MUTED, marker="o", ls="", ms=6.5, label="exact model")]
    h += [Line2D([], [], color=col, lw=2.6, label=f"$c$ = {c}") for c, col in CS.items()]
    axR.legend(handles=h, loc="upper right")
    axL.legend(handles=[Line2D([], [], color="#1baf7a", lw=2.6,
                               label="$x/(1-e^{-x})$, $x = K\\ln(1/\\chi)/(NE)$")],
               loc="upper right")
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_gamma_vs_chi.png", dpi=170)
    print("saved fig_gamma_vs_chi.png")


if __name__ == "__main__":
    main()
