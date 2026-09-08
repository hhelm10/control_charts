"""Factor figures for toy v3 (memory decay instead of caution). Same layout and
encoding as plot_factors_v2; the agent lever is the relevance decay rate c_dec.
Main figures use merge="evidence" (the decided rule); a companion figure compares
evidence vs receipt merge (staleness laundering)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from collections import defaultdict

D = json.load(open("factor_v3_results.json"))
BASE, FACTORS = D["base"], D["factors"]
RES = [r for r in D["results"] if r["kw"]["merge"] == "evidence"]
RES_ALL = D["results"]
ORDER = ["agent", "communication", "environment", "observation"]
HUE = {"agent": "#2a78d6", "communication": "#1baf7a", "environment": "#eb6834", "observation": "#4a3aa7"}
INK, MUTED = "#0b0b0b", "#8a8984"
STYLE = [dict(ls="-", marker="o"), dict(ls="--", marker="s")]
TITLE = {"agent": "Agent", "communication": "Communication", "environment": "Environment", "observation": "Observation"}
LN2 = float(np.log(2.0))
XLAB = {"c_dec": "Memory decay rate $c$  (relevance $\\langle q,k\\rangle\\,e^{-c\\cdot\\mathrm{age}}$)",
        "s_ask": "Demand concentration (Zipf exponent of $\\pi_{ask}$)",
        "lam_mean": "Environment speed $\\bar\\lambda$ (changes / question / step)",
        "alpha": "Asks per observation, $(1-\\alpha)/\\alpha$"}
LEVEL = {"N": lambda v: f"$N$ = {v}",
         "sigma": lambda v: {"unknown_first": "ask unknown-first", "oldest": "ask oldest-memory-first"}[v],
         "mean_degree": lambda v: "full mesh" if v is None else f"degree {v}",
         "peer": lambda v: {"uniform": "random peer", "freshest": "freshest-evidence peer"}[v],
         "lam_disp": lambda v: {0.0: "all questions same speed", 1.0: "speeds spread ×e", 2.0: "speeds spread ×e²"}[v],
         "s_env": lambda v: "flat environment" if v == 0 else "skewed environment (Zipf 1)",
         "B": lambda v: f"budget $B$ = {v}",
         "n_obs_frac": lambda v: "all agents observe" if v == 1.0 else "half of agents observe"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})


def shade(hex_, level):
    c = np.array(to_rgb(hex_)); return {0: 0.5 * c + 0.5, 1: c, 2: 0.65 * c}[level]


def xpos(xkey, x):
    if xkey == "alpha":
        return round((1.0 - x) / x)
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
        if xkey in ("lam_mean", "alpha", "c_dec"):
            ax.set_xscale("log")
            ax.set_xticks(xp_all)
            if xkey == "c_dec":
                ax.set_xticklabels([f"{v:g}\n$\\tau$={LN2 / v:.0f}" for v in xp_all], fontsize=7)
            else:
                ax.set_xticklabels([f"{v:g}" for v in xp_all])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlim(min(xp_all) / 1.6, max(xp_all) * 1.6)
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
        if xkey == "c_dec":  # R0 = m*tau/M = 1 at c = ln(1/theta) * m / M
            m_bw = (1 - BASE["alpha"]) * BASE["B"]
            thr = LN2 * m_bw / BASE["M"]
            ax.axvline(thr, color=MUTED, ls="--", lw=0.9)
            if show_title:
                ax.text(thr, 1.02, "$R_0$=1 ", color=MUTED, fontsize=7, va="top", ha="right")
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
             f"Other factors at baseline (dotted line): $N$={b['N']}, decay $c$={b['c_dec']:.4f} "
             f"($\\tau_{{eff}}=\\ln(1/\\theta)/c$ = {LN2 / b['c_dec']:.0f} steps at threshold $\\theta$={b['theta_ret']}), ask unknown-first, budget $B$={b['B']} · "
             f"full mesh, random peer, flat demand · $M$={b['M']}, $\\bar\\lambda$={b['lam_mean']}, flat environment · "
             f"$\\alpha$={b['alpha']} ({(1-b['alpha'])/b['alpha']:.0f} asks per observation), all agents observe.\n"
             f"Model v3: infinite memory, relevance $\\langle q,k\\rangle e^{{-c\\,\\mathrm{{age}}}}$ on the receipt clock, answer iff relevance $\\geq\\theta$; "
             f"no caution — a fast environment makes answers stale, not absent. Newest-evidence merge. "
             f"Mean ± s.e. over 10 seeds, last 200 of 800 steps (toy simulator v3).",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)


def make_figure(metric, ylabel, fname, suptitle):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), sharey=True)
    draw_row(axes, metric, ylabel)
    fig.suptitle(suptitle, fontsize=11, x=0.01, ha="left", y=0.995)
    footer(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97), w_pad=1.2)
    fig.savefig(fname, dpi=170); print("saved", fname)


def make_figure1(fname):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True)
    draw_row(axes[0], "idk", "P(“I don’t know”)", show_xlabel=False, show_title=True, row_label="Agent level")
    draw_row(axes[1], "dead_q", "P(no agent can answer)", show_xlabel=True, show_title=False, row_label="System level",
             show_legend=False)
    fig.suptitle("Epistemic collapse by factor (v3: memory decay, no caution). Top: an agent cannot answer. Bottom: no agent can answer.",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)
    footer(fig)
    fig.tight_layout(rect=(0.01, 0.05, 1, 0.97), w_pad=1.2, h_pad=1.0)
    fig.savefig(fname, dpi=170); print("saved", fname)


def make_merge_figure(fname):
    """Staleness laundering: evidence vs receipt merge across environment speed."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True)
    xs = FACTORS["environment"]["xs"]
    for ax, metric, ttl in zip(axes, ["idk", "stale", "correct"],
                               ["P(“I don’t know”)", "P(stale answer)", "P(correct answer)"]):
        for mi, mg in enumerate(["evidence", "receipt"]):
            d = defaultdict(list)
            for r in RES_ALL:
                if (r["factor"] == "environment" and r["kw"]["merge"] == mg
                        and r["kw"]["lam_disp"] == 0.0 and r["kw"]["s_env"] == 0.0):
                    d[r["kw"]["lam_mean"]].append(r[metric])
            ys = [np.mean(d[x]) for x in xs]
            se = [np.std(d[x]) / np.sqrt(len(d[x])) for x in xs]
            col = HUE["environment"] if mi == 0 else shade(HUE["environment"], 2)
            ax.errorbar(xs, ys, yerr=se, color=col, lw=1.7, ms=5, capsize=0, **STYLE[mi],
                        markerfacecolor=col if mi == 0 else "white", markeredgecolor=col,
                        label={"evidence": "newest evidence wins", "receipt": "newest receipt wins (repo)"}[mg])
        ax.set_xscale("log"); ax.set_xticks(xs); ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(-0.03, 1.03); ax.set_xlabel(XLAB["lam_mean"])
        ax.set_title(ttl, loc="left")
        ax.axvline(BASE["lam_mean"], color=MUTED, ls=":", lw=1.1)
    axes[0].legend(loc="upper left")
    fig.suptitle("Circulation launders staleness: the repo's newest-receipt merge trades wrongness for availability",
                 fontsize=11, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005,
             "Toy v3 at baseline except $\\bar\\lambda$ (x-axis). Under newest-receipt merge an old circulating copy overwrites\n"
             "newer evidence whenever it is re-received, so fresh observations spread no faster than stale lineages:\n"
             "IDK falls slightly, staleness roughly doubles. Mean ± s.e. over 10 seeds, last 200 of 800 steps.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(fname, dpi=170); print("saved", fname)


for r in RES:
    r["notcorrect"] = 1 - r["correct"]
make_figure("idk", "P(“I don’t know”)", "fig_factors_v3_idk.png",
            "Epistemic collapse by factor (v3): one intrinsic lever per factor.")
make_figure("notcorrect", "P(not correct)  (IDK + stale)", "fig_factors_v3_notcorrect.png",
            "Companion (v3): probability the answer is not correct (adds confidently stale answers).")
make_figure("dead_q", "P(no agent can answer)", "fig_factors_v3_system.png",
            "Companion (v3): system-level collapse (questions no agent can answer).")
make_figure1("figure1_v3_collapse_by_factor.png")
make_merge_figure("fig_v3_merge_laundering.png")
