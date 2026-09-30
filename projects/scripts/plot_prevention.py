"""How much do mitigations prevent collapse? Epistemic efficiency Gamma, where
pi0_strategy ~= pi0_none ** Gamma and M*_strategy = Gamma * M*_none.

Left: mimesis at the collapse threshold, Gamma* = x / (1 - e^{-x}),
      x = (K / (N E)) ln(1/chi) -- lines; exact birth-death model -- markers
      (varying the ask bandwidth K at fixed B via alpha is avoided: we vary N).
Right: books over time, Gamma(t) = (1 - p_w)(tau_b/tau)(1 + f K b(t) / E), for
      three write rates; markers = exact model ln(pi0_books)/ln(pi0_none).
"""
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analytic_collapse import collapse, renewal_idk
from scipy.optimize import brentq

INK, MUTED = "#0b0b0b", "#8a8984"
B, ALPHA, K_CTX, SF = 11, 3 / 11, 3, 0.5
E, K = ALPHA * B, (1 - ALPHA) * B
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 10.5, "legend.frameon": False})


def mstar(c, chi, strat, N=10, **kw):
    return brentq(lambda M: collapse(M, N=N, c=c, strategy=strat, **kw)["pi0"] - chi,
                  3, 50000, xtol=0.1)


def coverage_path(M, c, ts, write_p, W, N=10):
    """b(t) at the requested times, integrating the coverage ODE once."""
    D = math.floor(math.log(1 / SF) / c)
    live = 1 - write_p
    mu = (1 - (1 - ALPHA / M) ** B) * live
    rate = N * write_p / W
    out, b, tt = [], 0.0, 0
    for T in ts:
        while tt < T:
            p_u = renewal_idk(mu, D, B, live * (ALPHA + (1 - ALPHA) * b), K_CTX)
            b = 1 - (1 - b) * math.exp(-rate * (1 - p_u)); tt += 1
        out.append(b)
    return np.array(out)


def books_gamma_closed(M, c, b, write_p, N=10):
    """rare-holder limit: (1 - p_w)(tau_b/tau)(1 + f K b / E)."""
    D = math.floor(math.log(1 / SF) / c)
    p = collapse(M, N=N, c=c)["agent_idk"]
    tau = D + 1 + K_CTX / (B * (ALPHA + (1 - ALPHA) * (1 - p)))
    tau_b = D + 1 + K_CTX / (B * (1 - write_p) * (ALPHA + (1 - ALPHA) * b))
    return (1 - write_p) * (tau_b / tau) * (1 + K * b / E)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.4, 4.6))

    # --- mimesis: Gamma* vs x = K ln(1/chi) / (N E)
    xs = np.linspace(1e-3, 1.4, 200)
    axL.plot(xs, xs / (1 - np.exp(-xs)), color="#1baf7a", lw=2.6, label="closed form")
    for chi, mk in ((0.25, "o"), (0.5, "s"), (0.75, "^")):
        for N in (3, 5, 10, 20):
            for c in (0.05, 0.5):
                x = K * math.log(1 / chi) / (N * E)
                g = mstar(c, chi, "mimesis", N=N) / mstar(c, chi, "none", N=N)
                axL.plot([x], [g], marker=mk, color=INK if c == 0.05 else MUTED, ls="",
                         ms=6.5, markerfacecolor="white" if c == 0.5 else INK)
    axL.set_title("sharing secondhand information", loc="left")
    axL.set_xlabel("$x = K \\ln(1/\\chi)\\,/\\,(N E)$")
    axL.set_ylabel("epistemic efficiency $\\Gamma$  ($M^\\star$ ratio)")
    hL = [Line2D([], [], color="#1baf7a", lw=2.6, label="$x/(1-e^{-x})$")]
    hL += [Line2D([], [], color=INK, marker=m, ls="", ms=6.5, label=f"exact, $\\chi$ = {chi}")
           for chi, m in ((0.25, "o"), (0.5, "s"), (0.75, "^"))]
    hL += [Line2D([], [], color=MUTED, marker="o", ls="", ms=6.5, markerfacecolor="white",
                  label="(hollow: $c$ = 0.5)")]
    axL.legend(handles=hL, loc="upper left")

    # --- books: Gamma(t) for three write rates, c=0.5, M=400 (collapse regime)
    M, c = 400, 0.5
    ts = np.arange(0, 3001, 50)
    lnp_none = math.log(collapse(M, c=c)["pi0"])
    for (pw, W, lab, col) in ((0.20, 2, "10 writes / 100 steps", "#2a78d6"),
                              (0.05, 5, "1 write / 100 steps", "#6ea4e4"),
                              (0.02, 10, "0.2 writes / 100 steps", "#a9c8ef")):
        bs = coverage_path(M, c, ts, pw, W)
        axR.plot(ts, [books_gamma_closed(M, c, b, pw) for b in bs], color=col, lw=2.6, label=lab)
        for t, b in list(zip(ts, bs))[::6]:
            exact = math.log(collapse(M, c=c, strategy="books", write_p=pw, book_frac=1.0,
                                      book_cov=b)["pi0"]) / lnp_none
            axR.plot([t], [exact], marker="o", color=col, ls="", ms=6.5,
                     markeredgecolor=INK, markeredgewidth=0.8)
    axR.axhline(1, color=MUTED, ls=":", lw=1.4)
    axR.set_title("books ($c$ = 0.5, $M$ = 400)", loc="left")
    axR.set_xlabel("step")
    axR.set_ylabel("epistemic efficiency $\\Gamma(t)$")
    axR.legend(loc="lower right", title="expected write rate", title_fontsize=10.5)
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_prevention.png", dpi=170)
    print("saved fig_prevention.png")


if __name__ == "__main__":
    main()
