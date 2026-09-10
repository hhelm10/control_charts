"""Factor figures for toy v4 (crowding retrieval). Layout and encoding as before;
the agent lever is the relevance decay rate c_dec, which under crowding sets the
similarity head start Delta = ln(1/chi)/c_dec of the exact match over
cross-question entries in the finite context."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from collections import defaultdict

D = json.load(open("factor_v4_results.json"))
BASE, FACTORS, RES = D["base"], D["factors"], D["results"]
ORDER = ["agent", "communication", "environment", "observation"]
HUE = {"agent": "#2a78d6", "communication": "#1baf7a", "environment": "#eb6834", "observation": "#4a3aa7"}
INK, MUTED = "#0b0b0b", "#8a8984"
STYLE = [dict(ls="-", marker="o"), dict(ls="--", marker="s")]
TITLE = {"agent": "Agent", "communication": "Communication", "environment": "Environment", "observation": "Observation"}
LN2 = float(np.log(2.0))
XLAB = {"c_dec": "Memory decay rate $c$  (relevance $\\langle q,k\\rangle\\,e^{-c\\cdot\\mathrm{age}}$)",
        "s_ask": "Demand concentration (Zipf exponent of $\\pi_{ask}$)",
        "lam_mean": "Environment speed $\\bar\\lambda$ (changes / question / step)",
        "alpha": "Observation share of budget $\\alpha$\n(below: steps between visits to a question, at baseline)"}
LEVEL = {"N": lambda v: f"$N$ = {v}",
         "sigma": lambda v: {"unknown_first": "ask unknown-first", "oldest": "ask oldest-memory-first"}[v],
         "mean_degree": lambda v: "full mesh" if v is None else f"degree {v}",
         "peer": lambda v: {"uniform": "random peer", "freshest": "freshest-evidence peer"}[v],
         "B_bw": lambda v: f"budget $B$ = {v} (obs. rate fixed)",
         "M": lambda v: f"$M$ = {v} questions",
         "k_ctx": lambda v: f"context size $k$ = {v}",
         "s_env": lambda v: "flat environment" if v == 0 else "skewed environment (Zipf 1)",
         "B": lambda v: f"budget $B$ = {v}",
         "n_obs_frac": lambda v: "all agents observe" if v == 1.0 else "half of agents observe"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})


def shade(hex_, level):
    c = np.array(to_rgb(hex_)); return {0: 0.5 * c + 0.5, 1: c, 2: 0.65 * c}[level]


def xpos(xkey, x):
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
                ax.set_xticklabels([f"{v:g}\n$\\Delta$={LN2 / v:.0f}" for v in xp_all], fontsize=7)
            elif xkey == "alpha":
                # top: share of budget observing; bottom: per-question revisit interval M/(alpha*B*N) at baseline
                ax.set_xticklabels([f"{v * 100:g}%\n{BASE['M'] / (v * BASE['B'] * BASE['N']):,.0f} steps"
                                    for v in xp_all], fontsize=7)
            else:
                ax.set_xticklabels([f"{v:g}" for v in xp_all])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            if xkey == "alpha":     # scarcity increases to the right
                ax.set_xlim(max(xp_all) * 1.6, min(xp_all) / 1.6)
            else:
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
        if xkey == "c_dec":  # R0 = 1 at Delta + k/r ~ M/m, i.e. c ~ ln(1/chi) * m / M (r large)
            m_bw = (1 - BASE["alpha"]) * BASE["B"]
            thr = np.log(1 / BASE["chi"]) * m_bw / BASE["M"]
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
             f"Other factors at baseline (dotted line): $N$={b['N']}, decay $c$={b['c_dec']:.4f}, cross-similarity $\\chi$={b['chi']} "
             f"(head start $\\Delta=\\ln(1/\\chi)/c$ = {np.log(1/b['chi']) / b['c_dec']:.0f} steps), context $k$={b['k_ctx']}, staleness-aware asking, budget $B$={b['B']} · "
             f"full mesh, random peer, flat demand · $M$={b['M']}, speeds log-normal($\\bar\\lambda$={b['lam_mean']}, $\\sigma_\\lambda$={b['lam_disp']:g}), flat environment · $\\alpha$={b['alpha']}, all agents observe.\n"
             f"Model v4 (crowding): the database never deletes; every received answer inserts; question $q$ is answerable iff fewer than $k$ fresher "
             f"insertions outrank its entry's decayed score in the fixed context.\n"
             f"Forgetting is relative and activity-dependent ($\\tau_{{eff}} = \\Delta + k/r_{{ins}}$); no caution. Staleness-aware asking: known questions are "
             f"re-verified with weight $(1-e^{{-\\lambda_q\\cdot\\mathrm{{evidence\\,age}}}})^2$, so a fast world drains the ask budget into re-verification.\n"
             f"Communication panel varies budget at fixed observation rate ($\\alpha B$ = 0.05). Mean ± s.e. over 10 seeds, last 200 of 800 steps (toy v4).",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)


def make_figure(metric, ylabel, fname, suptitle):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.1), sharey=True)
    draw_row(axes, metric, ylabel)
    fig.suptitle(suptitle, fontsize=11, x=0.01, ha="left", y=0.995)
    footer(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97), w_pad=1.2)
    fig.savefig(fname, dpi=170); print("saved", fname)


def make_figure1(fname):
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharey=True)
    draw_row(axes[0], "idk", "P(“I don’t know”)", show_xlabel=False, show_title=True, row_label="Agent level")
    draw_row(axes[1], "dead_q", "P(no agent can answer)", show_xlabel=True, show_title=False, row_label="System level",
             show_legend=False)
    fig.suptitle("Epistemic collapse by factor (v4: crowding — forgetting is relative). Top: an agent cannot answer. Bottom: no agent can answer.",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)
    footer(fig)
    fig.tight_layout(rect=(0.01, 0.06, 1, 0.97), w_pad=1.2, h_pad=1.0)
    fig.savefig(fname, dpi=170); print("saved", fname)


for r in RES:
    r["notcorrect"] = 1 - r["correct"]
make_figure("idk", "P(“I don’t know”)", "fig_factors_v4_idk.png",
            "Epistemic collapse by factor (v4): one intrinsic lever per factor.")
make_figure("notcorrect", "P(not correct)  (IDK + stale)", "fig_factors_v4_notcorrect.png",
            "Companion (v4): probability the answer is not correct (adds confidently stale answers).")
make_figure("dead_q", "P(no agent can answer)", "fig_factors_v4_system.png",
            "Companion (v4): system-level collapse (questions no agent can answer).")
make_figure1("figure1_v4_collapse_by_factor.png")
