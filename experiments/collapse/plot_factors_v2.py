"""1x4 factor figure for toy v2. Each x-axis is an intrinsic property of the factor
that, as it grows, raises the probability of epistemic collapse. Encoding as before:
hue = factor, lightness = 2nd variable, solid/dashed = 3rd variable, dotted = baseline."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from collections import defaultdict

D = json.load(open("factor_v2_results.json"))
BASE, FACTORS, RES = D["base"], D["factors"], D["results"]
ORDER = ["agent", "communication", "environment", "observation"]
HUE = {"agent": "#2a78d6", "communication": "#1baf7a", "environment": "#eb6834", "observation": "#4a3aa7"}
INK, MUTED = "#0b0b0b", "#8a8984"
STYLE = [dict(ls="-", marker="o"), dict(ls="--", marker="s")]
TITLE = {"agent": "Agent", "communication": "Communication", "environment": "Environment", "observation": "Observation"}
XLAB = {"c": "Caution $c$ (required P[answer still true])",
        "s_ask": "Demand concentration (Zipf exponent of $\\pi_{ask}$)",
        "lam_mean": "Environment speed $\\bar\\lambda$ (changes / question / step)",
        "alpha": "Asks per observation, $(1-\\alpha)/\\alpha$"}
LEVEL = {"N": lambda v: f"$N$ = {v}",
         "sigma": lambda v: {"unknown_first": "ask unknown-first", "oldest": "ask oldest-evidence-first"}[v],
         "mean_degree": lambda v: "full mesh" if v is None else f"degree {v}",
         "peer": lambda v: {"uniform": "random peer", "freshest": "freshest-evidence peer"}[v],
         "lam_disp": lambda v: {0.0: "all questions same speed", 1.0: "speeds spread ×e", 2.0: "speeds spread ×e²"}[v],
         "s_env": lambda v: "flat environment" if v == 0 else "skewed environment (Zipf 1)",
         "B": lambda v: f"budget $B$ = {v}",
         "n_obs_frac": lambda v: "all agents observe" if v == 1.0 else "half of agents observe"}
LEGEND_POS = {
    "idk":        {"agent": ("lower right", "center right"), "communication": ("upper left", "lower right"),
                   "environment": ("upper left", "center left"), "observation": ("center left", "center left")},
    "notcorrect": {"agent": ("lower right", "center right"), "communication": ("upper left", "lower right"),
                   "environment": ("upper left", "center left"), "observation": ("center left", "lower right")},
    "dead_q":     {"agent": ("center right", "upper left"), "communication": ("upper left", "lower right"),
                   "environment": ("upper left", "center left"), "observation": ("upper left", "lower right")},
}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})


def shade(hex_, level):
    c = np.array(to_rgb(hex_)); return {0: 0.5 * c + 0.5, 1: c, 2: 0.65 * c}[level]


def xpos(xkey, x):
    if xkey == "alpha":
        return round((1.0 - x) / x)     # asks per observation (round by construction)
    return x


def draw_row(axes, metric, ylabel, show_xlabel=True, show_title=True, row_label=None, show_legend=True):
    for i, (ax, fac) in enumerate(zip(axes, ORDER)):
        f = FACTORS[fac]; xkey, ckey, skey = f["x"], f["color"], f["style"]; hue = HUE[fac]
        d = defaultdict(list)
        for r in RES:
            if r["factor"] == fac:
                d[(r["kw"][ckey], r["kw"][skey], r["kw"][xkey])].append(r[metric])
        for ci, cval in enumerate(f["colors"]):
            for si, sval in enumerate(f["styles"]):
                ys = [np.mean(d[(cval, sval, x)]) for x in f["xs"]]
                se = [np.std(d[(cval, sval, x)]) / np.sqrt(len(d[(cval, sval, x)])) for x in f["xs"]]
                xp = [xpos(xkey, x) for x in f["xs"]]
                col = shade(hue, ci)
                ax.errorbar(xp, ys, yerr=se, color=col, lw=1.7, ms=5, capsize=0, elinewidth=1, **STYLE[si],
                            markerfacecolor=col if si == 0 else "white", markeredgecolor=col, markeredgewidth=1.3)
        xp_all = [xpos(xkey, x) for x in f["xs"]]
        if xkey in ("lam_mean", "alpha"):
            ax.set_xscale("log")
            ax.set_xticks(xp_all); ax.set_xticklabels([f"{v:g}" for v in xp_all])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(min(xp_all) / 1.6, max(xp_all) * 1.6)
        elif xkey == "c":
            ax.set_xticks([0.3, 0.5, 0.7, 0.9]); ax.set_xlim(0.25, 1.0)
        else:
            ax.set_xticks([0, 1, 2, 3]); ax.set_xlim(-0.15, 3.15)
        ax.set_ylim(-0.03, 1.03)
        if show_xlabel:
            ax.set_xlabel(XLAB[xkey])
        else:
            ax.tick_params(labelbottom=False)
        if show_title:
            ax.set_title(TITLE[fac], loc="left", color=hue, fontweight="bold")
        bx = xpos(xkey, BASE[xkey])
        ax.axvline(bx, color=MUTED, ls=":", lw=1.1)
        if i == 0 and show_title:
            ax.text(bx, -0.02, " baseline", color=MUTED, fontsize=7, va="bottom", ha="left")
        h1 = [Line2D([], [], color=shade(hue, k), lw=2, label=LEVEL[ckey](v)) for k, v in enumerate(f["colors"])]
        h2 = [Line2D([], [], color=INK, lw=1.4, ls=STYLE[k]["ls"], marker=STYLE[k]["marker"], ms=4.5,
                     markerfacecolor=INK if k == 0 else "white", label=LEVEL[skey](v)) for k, v in enumerate(f["styles"])]
        if not show_legend:
            continue
        l1 = ax.legend(handles=h1, loc="upper left", bbox_to_anchor=(0.0, 1.0),
                       frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
        ax.add_artist(l1)
        ax.legend(handles=h2, loc="upper left", bbox_to_anchor=(0.0, 1.0 - 0.075 * (len(h1) + 0.6)),
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.85)
    axes[0].set_ylabel(ylabel)
    if row_label:
        axes[0].text(-0.28, 0.5, row_label, transform=axes[0].transAxes, rotation=90, va="center", ha="center",
                     fontsize=10, fontweight="bold", color=INK)


def footer(fig):
    b = BASE
    fig.text(0.01, 0.005,
             f"Other factors at baseline (dotted line): $N$={b['N']}, $c$={b['c']}, memory $\\tau$={b['tau_recv']} steps, ask unknown-first, budget $B$={b['B']} · "
             f"full mesh, random peer, flat demand · $M$={b['M']}, $\\bar\\lambda$={b['lam_mean']} for every question, flat environment · "
             f"$\\alpha$={b['alpha']} ({(1-b['alpha'])/b['alpha']:.0f} asks per observation), all agents observe.\n"
             f"Model: provenance clock, newest evidence wins, answer iff evidence age satisfies $e^{{-\\lambda_q\\,\\mathrm{{age}}}}\\geq c$ and receipt age $\\leq\\tau$. "
             f"Mean ± s.e. over 10 seeds, last 200 of {800} steps (toy simulator v2).",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)


def make_figure(metric, ylabel, fname, suptitle):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), sharey=True)
    draw_row(axes, metric, ylabel)
    fig.suptitle(suptitle, fontsize=11, x=0.01, ha="left", y=0.995)
    footer(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), w_pad=1.2)
    fig.savefig(fname, dpi=170); print("saved", fname)


def make_figure1(fname):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True)
    draw_row(axes[0], "idk", "P(“I don’t know”)", show_xlabel=False, show_title=True, row_label="Agent level")
    draw_row(axes[1], "dead_q", "P(no agent can answer)", show_xlabel=True, show_title=False, row_label="System level",
             show_legend=False)
    fig.suptitle("Epistemic collapse by factor. Top: an agent cannot answer. Bottom: no agent can answer.",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)
    footer(fig)
    fig.tight_layout(rect=(0.01, 0.04, 1, 0.97), w_pad=1.2, h_pad=1.0)
    fig.savefig(fname, dpi=170); print("saved", fname)


for r in RES:
    r["notcorrect"] = 1 - r["correct"]
make_figure("idk", "P(“I don’t know”)", "fig_factors_v2_idk.png",
            "Epistemic collapse by factor: one intrinsic lever per factor, all rising toward collapse.")
make_figure("notcorrect", "P(not correct)  (IDK + stale)", "fig_factors_v2_notcorrect.png",
            "Companion: probability the answer is not correct (adds confidently stale answers).")
make_figure("dead_q", "P(no agent can answer)", "fig_factors_v2_system.png",
            "Companion: system-level collapse (questions no agent can answer).")
make_figure1("figure1_collapse_by_factor.png")
