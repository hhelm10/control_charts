"""Figure: contradiction-induced abstention makes environment speed visible in P(IDK)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from collections import defaultdict

D = json.load(open("speed_conflict_results.json"))
RES = D["results"]
HUE = "#eb6834"  # environment orange
INK, MUTED = "#0b0b0b", "#8a8984"
LN2 = float(np.log(2.0))
LAMS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
TAUS = [15, 20, 40]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})


def shade(hex_, level):
    c = np.array(to_rgb(hex_)); return {0: 0.5 * c + 0.5, 1: c, 2: 0.65 * c}[level]


def series(pred, metric):
    d = defaultdict(list)
    for r in RES:
        if pred(r["kw"]):
            d[r["kw"]["lam_mean"]].append(r[metric])
    ys = [np.mean(d[x]) for x in LAMS]
    se = [np.std(d[x]) / np.sqrt(len(d[x])) for x in LAMS]
    return ys, se


fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))
for ax, metric, ttl in zip(axes, ["idk", "conflicted", "dead_q"],
                           ["P(“I don’t know”)", "P(retrievable but contradictory)",
                            "P(no agent can answer)"]):
    for ti, tau in enumerate(TAUS):
        c_dec = round(LN2 / tau, 4)
        ys, se = series(lambda k, c=c_dec: k["conflict"] and k["c_dec"] == c, metric)
        col = shade(HUE, ti)
        ax.errorbar(LAMS, ys, yerr=se, color=col, lw=1.7, ms=5, marker="o", capsize=0,
                    label=f"$\\tau$ = {tau}  ($R_0$ = {4.95 * tau / 50:.1f})")
    ys, se = series(lambda k: not k["conflict"], metric)
    ax.errorbar(LAMS, ys, yerr=se, color=MUTED, lw=1.4, ls="--", marker="s", ms=4,
                markerfacecolor="white", capsize=0, label="no conflict rule ($\\tau$ = 20)")
    ax.set_xscale("log"); ax.set_xticks(LAMS); ax.set_xticklabels([f"{v:g}" for v in LAMS])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(-0.02, 0.55)
    ax.set_xlabel("Environment speed $\\bar\\lambda$ (changes / question / step)")
    ax.set_title(ttl, loc="left")
    ax.axvline(0.003, color=MUTED, ls=":", lw=1.1)
axes[0].legend(loc="upper left")
fig.suptitle("Making speed visible in “I don’t know”: contradictory memories.  The database never deletes, so a fast world fills\n"
             "retrieved context with disagreeing values and the agent abstains — and slower decay amplifies the effect.",
             fontsize=10.5, x=0.01, ha="left", y=0.995)
fig.text(0.01, 0.005,
         "Toy v3 + conflict rule: the superseded copy stays retrievable until its own relevance $e^{-c\\,\\mathrm{age}}$ falls below $\\theta$;\n"
         "if it disagrees with the current copy the agent answers IDK. Baseline otherwise ($N$=20, $M$=50, $B$=5, $\\alpha$=0.01, mesh).\n"
         "Without the rule (grey dashed) P(IDK) is exactly $\\lambda$-independent. Mean ± s.e., 10 seeds, last 200 of 800 steps.",
         ha="left", va="bottom", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.1, 1, 0.9))
fig.savefig("fig_v3_speed_conflict.png", dpi=170)
print("saved fig_v3_speed_conflict.png")
