"""Four-factor figure (1 x 4): one panel per factor, same metric everywhere.

Encoding (identical in every panel):
  hue         = factor
  lightness   = level of the factor's 2nd variable (light -> dark = small -> large)
  line/marker = level of the factor's 3rd variable (solid filled circle = baseline, dashed hollow square = alternative)
  grey dotted vertical = baseline value of the x variable (what the other panels hold it at)
  grey dashed vertical = mean-field threshold R0 = K*tau/Q = 1, where x maps onto R0
Every x-axis is a resource, so all curves fall as x grows.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from collections import defaultdict

D = json.load(open("factor_results.json"))
BASE, FACTORS, RES = D["base"], D["factors"], D["results"]

HUE = {"agent": "#2a78d6", "environment": "#eb6834", "communication": "#1baf7a", "observation": "#4a3aa7"}
INK, MUTED = "#0b0b0b", "#8a8984"
STYLE = [dict(ls="-", marker="o"), dict(ls="--", marker="s")]

TITLE = {"agent": "Agent", "environment": "Environment", "communication": "Communication", "observation": "Observation"}
XLAB = {"tau_mem": r"Memory lifetime $\tau_{mem}$ (steps)",
        "obs_zipf": "Environment flatness (normalised entropy)",
        "K": "Bandwidth $K$ (questions / agent / step)",
        "eps": r"Observation rate $\epsilon$ (per agent / step)"}
LEVEL = {"N": lambda v: f"$N$ = {v}",
         "capacity": lambda v: "unbounded memory" if v is None else f"memory of {v} entries",
         "Q": lambda v: f"$Q$ = {v}",
         "p_env": lambda v: fr"$p_{{env}}$ = {v}",
         "mean_degree": lambda v: "full mesh" if v is None else f"degree {v}",
         "strategy": lambda v: {"unknown_first": "ask unknown-first", "oldest": "ask oldest-first"}[v],
         "obs_frac": lambda v: f"{int(round(v * BASE['N']))} of {BASE['N']} observe",
         "obs_mode": lambda v: {"random": "random questions", "assigned": "assigned subset"}[v]}
# where the two legends sit, per panel and metric
LEGEND_POS = {
    "idk":        {"agent": ("center right", "lower right"), "environment": ("center right", "upper left"),
                   "communication": ("upper right", "center right"), "observation": ("upper right", "center right")},
    "notcorrect": {"agent": ("lower right", "center right"), "environment": ("lower left", "center left"),
                   "communication": ("lower right", "center right"), "observation": ("lower left", "center left")},
    "dead_q":     {"agent": ("lower right", "center right"), "environment": ("center left", "upper left"),
                   "communication": ("lower right", "center right"), "observation": ("lower left", "center left")},
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2,
                     "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})


def shade(hex_, level):
    c = np.array(to_rgb(hex_))
    return {0: 0.5 * c + 0.5, 1: c, 2: 0.65 * c}[level]


def flatness(s_zipf, Q):
    """Normalised entropy of a Zipf(s) distribution over Q questions: 1 = flat."""
    p = np.arange(1, Q + 1, dtype=float) ** (-s_zipf); p /= p.sum()
    return float(-(p * np.log(p)).sum() / np.log(Q))


def make_figure(metric, ylabel, fname, suptitle):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), sharey=True)
    for i, (ax, (fac, f)) in enumerate(zip(axes, FACTORS.items())):
        xkey, ckey, skey = f["x"], f["color"], f["style"]
        hue = HUE[fac]
        d = defaultdict(list)
        for r in RES:
            if r["factor"] == fac:
                d[(r["kw"][ckey], r["kw"][skey], r["kw"][xkey])].append(r[metric])

        def xpos(x, Q=BASE["Q"]):
            if xkey == "eps" and x == 0:
                return 0.001                      # eps = 0 drawn at the left edge of the log axis
            if xkey == "obs_zipf":
                return flatness(x, Q)
            return x

        for ci, cval in enumerate(f["colors"]):
            for si, sval in enumerate(f["styles"]):
                ys = [np.mean(d[(cval, sval, x)]) for x in f["xs"]]
                se = [np.std(d[(cval, sval, x)]) / np.sqrt(len(d[(cval, sval, x)])) for x in f["xs"]]
                xp = [xpos(x, cval if ckey == "Q" else BASE["Q"]) for x in f["xs"]]
                col = shade(hue, ci)
                ax.errorbar(xp, ys, yerr=se, color=col, lw=1.7, ms=5, capsize=0, elinewidth=1, **STYLE[si],
                            markerfacecolor=col if si == 0 else "white", markeredgecolor=col, markeredgewidth=1.3)

        # axes
        if xkey == "obs_zipf":
            ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlim(0.15, 1.05)
        else:
            ax.set_xscale("log")
            ax.set_xticks([xpos(x) for x in f["xs"]])
            ax.set_xticklabels(["0" if (xkey == "eps" and x == 0) else f"{x:g}" for x in f["xs"]])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel(XLAB[xkey]); ax.set_ylim(-0.03, 1.03)
        ax.set_title(TITLE[fac], loc="left", color=hue, fontweight="bold")

        # reference lines: baseline (dotted) and R0 = 1 (dashed)
        bx = xpos(BASE[xkey])
        ax.axvline(bx, color=MUTED, ls=":", lw=1.1)
        if i == 0:
            ax.text(bx, 1.02, " baseline", color=MUTED, fontsize=7, va="top", ha="left")
        thr = {"tau_mem": BASE["Q"] / BASE["K"], "K": BASE["Q"] / BASE["tau_mem"]}.get(xkey)
        if thr is not None:
            ax.axvline(thr, color=MUTED, ls="--", lw=0.9)
            ax.text(thr, 1.02, r"$R_0$=1 ", color=MUTED, fontsize=7, va="top", ha="right")

        # legends: colour levels (2nd variable) and line style (3rd variable)
        h1 = [Line2D([], [], color=shade(hue, k), lw=2, label=LEVEL[ckey](v)) for k, v in enumerate(f["colors"])]
        h2 = [Line2D([], [], color=INK, lw=1.4, ls=STYLE[k]["ls"], marker=STYLE[k]["marker"], ms=4.5,
                     markerfacecolor=INK if k == 0 else "white", label=LEVEL[skey](v)) for k, v in enumerate(f["styles"])]
        loc1, loc2 = LEGEND_POS[metric][fac]
        l1 = ax.legend(handles=h1, loc=loc1); ax.add_artist(l1)
        ax.legend(handles=h2, loc=loc2, bbox_to_anchor=(0, 0.88, 1, 0) if (fac == "environment" and metric == "idk") else None)

    axes[0].set_ylabel(ylabel)
    b = BASE
    fig.suptitle(suptitle, fontsize=11, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005,
             f"Other factors held at baseline (dotted line): $N$={b['N']}, $\\tau_{{mem}}$={b['tau_mem']}, unbounded memory · flat environment, "
             f"$Q$={b['Q']}, $p_{{env}}$={b['p_env']} · $K$={b['K']}, full mesh, ask unknown-first · $\\epsilon$={b['eps']}, all agents, random questions.\n"
             f"Dashed line: mean-field threshold $R_0=K\\tau_{{mem}}/Q=1$.  Mean ± s.e. over 12 seeds, last 200 of 1000 steps (toy simulator).",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), w_pad=1.2)
    fig.savefig(fname, dpi=170)
    print("saved", fname)


for r in RES:
    r["notcorrect"] = 1 - r["correct"]
make_figure("idk", "“I don’t know” fraction", "fig_factors_idk.png",
            "Epistemic collapse by factor.  Each x-axis is a resource; scarcity induces collapse.")
make_figure("notcorrect", "Not-correct fraction (IDK + stale)", "fig_factors_notcorrect.png",
            "Companion: not-correct fraction (adds confidently stale answers to “I don’t know”).")
make_figure("dead_q", "Dead-question fraction (system-level)", "fig_factors_system.png",
            "Companion: system-level collapse (questions no agent can answer).")
