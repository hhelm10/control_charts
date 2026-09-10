"""The price of epistemic hygiene: one panel, two answering policies.

Policy "open" (default): any retrievable entry grounds an answer.
Policy "firsthand" (no miming): only self-observed evidence grounds an answer,
and hearsay is never re-told (it still inserts, and crowds).

Everything else held at baseline; the only mover is the observation share alpha.
The curves must meet at alpha = 1 (no asks left), so the gap between them IS the
epistemic value of communication.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import toy_v4

LN2 = float(np.log(2.0))
INK, MUTED = "#0b0b0b", "#8a8984"
GREEN = "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 8, "legend.frameon": False, "legend.handlelength": 2.2})

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
BUDGETS = [2, 5]
SEEDS = range(10)
BASE = dict(N=20, M=50, c_dec=LN2 / 20, chi=0.5, k_ctx=3, lam_mean=0.003, T=800)
GREY = "#444444"


def shade(hex_, level):
    from matplotlib.colors import to_rgb
    c = np.array(to_rgb(hex_)); return {0: 0.55 * c + 0.45, 1: c, 2: 0.65 * c}[level]


def cell(job):
    pol, B, a, s = job
    r = toy_v4.run(seed=s, B=B, alpha=a, answer_policy=pol, **BASE)
    sl = slice(-40, None)
    return {"pol": pol, "B": B, "alpha": a,
            **{k: float(r[k][sl].mean()) for k in ("idk", "stale", "correct", "dead_q", "sys_corr")}}


def main():
    jobs = [(pol, B, a, s) for pol in ("open", "firsthand") for B in BUDGETS
            for a in ALPHAS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=4))
    json.dump(res, open("policy_results.json", "w"))

    fig, axg = plt.subplots(2, 2, figsize=(11.2, 7.6), sharex=True, sharey=True)
    (axr, ax), (ax2r, ax2) = axg
    stats = {}
    for pol, hue, style in (("open", GREEN, dict(ls="-", marker="o")),
                            ("firsthand", GREY, dict(ls="--", marker="s", markerfacecolor="white"))):
        for bi, B in enumerate(BUDGETS):
            ys, se, stale, dead, dse = [], [], [], [], []
            for a in ALPHAS:
                sel = [r for r in res if r["pol"] == pol and r["B"] == B and r["alpha"] == a]
                v = [r["idk"] for r in sel]; d = [r["dead_q"] for r in sel]
                ys.append(np.mean(v)); se.append(np.std(v) / np.sqrt(len(v)))
                dead.append(np.mean(d)); dse.append(np.std(d) / np.sqrt(len(d)))
                stale.append(np.mean([r["stale"] for r in sel]))
            stats[(pol, B)] = (ys, stale, dead)
            col = shade(hue, bi)
            acc = [np.mean([r["correct"] for r in res if r["pol"] == pol and r["B"] == B and r["alpha"] == a]) for a in ALPHAS]
            sysc = [np.mean([r["sys_corr"] for r in res if r["pol"] == pol and r["B"] == B and r["alpha"] == a]) for a in ALPHAS]
            ax.errorbar(ALPHAS, ys, yerr=se, color=col, lw=1.7, ms=5, capsize=0, **style)
            ax2.errorbar(ALPHAS, dead, yerr=dse, color=col, lw=1.7, ms=5, capsize=0, **style)
            axr.errorbar(ALPHAS, acc, color=col, lw=1.7, ms=5, capsize=0, **style)
            ax2r.errorbar(ALPHAS, sysc, color=col, lw=1.7, ms=5, capsize=0, **style)
    for a_, ylab, rowlab in ((ax, "P(“I don’t know”)", None),
                             (ax2, "P(no agent can answer)", None),
                             (axr, "P(correct answer)", "Agent level"),
                             (ax2r, "P(some agent answers correctly)", "System level")):
        a_.set_ylim(-0.03, 1.03)
        a_.axvline(0.01, color=MUTED, ls=":", lw=1.1)
        a_.set_ylabel(ylab)
        if rowlab:
            a_.text(-0.17, 0.5, rowlab, transform=a_.transAxes, rotation=90, va="center", ha="center",
                    fontsize=10, fontweight="bold", color=INK)
    for a_ in (ax2, ax2r):
        a_.set_xscale("log"); a_.set_xticks(ALPHAS)
        a_.set_xticklabels([f"{a * 100:g}%" for a in ALPHAS], rotation=25)
        a_.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        a_.set_xlabel("Share of budget spent observing the environment, $\\alpha$")
    ax.tick_params(labelleft=True); ax2.tick_params(labelleft=True)
    ax.set_title("Availability: refusing to repeat others makes agents ignorant", loc="left")
    axr.set_title("Accuracy: abstention costs more than staleness", loc="left")
    ax.text(0.01, 1.0, " baseline", color=MUTED, fontsize=7, va="top")
    # the gap annotation at B = 5: firsthand's best vs open at baseline
    fh_best = min(stats[("firsthand", 5)][0])
    ax.annotate("", xy=(1.0, fh_best), xytext=(0.01, fh_best),
                arrowprops=dict(arrowstyle="<-", color=MUTED, lw=1.1))
    ax.text(0.105, fh_best + 0.05, "at $B$=5, a no-miming collective spending its whole budget\n"
            "observing still knows less than an open one observing 1% of the time",
            fontsize=7.5, color=INK, ha="center", va="bottom")
    from matplotlib.lines import Line2D
    h1 = [Line2D([], [], color=GREEN, ls="-", marker="o", ms=5, lw=1.7,
                 label="open: any retrievable entry grounds"),
          Line2D([], [], color=GREY, ls="--", marker="s", ms=5, lw=1.7, markerfacecolor="white",
                 label="no miming: only first-hand grounds")]
    h2 = [Line2D([], [], color=shade("#777777", k), lw=2.4, label=f"budget $B$ = {B}")
          for k, B in enumerate(BUDGETS)]
    l1 = ax.legend(handles=h1, loc="center left", bbox_to_anchor=(0.02, 0.66))
    ax.add_artist(l1)
    ax.legend(handles=h2, loc="center left", bbox_to_anchor=(0.02, 0.44), title="light → dark",
              title_fontsize=7)
    fig.text(0.01, 0.005,
             "Toy v4 at baseline except $\\alpha$ and $B$; asks fill the rest of the budget under both policies.\n"
             "Hearsay still inserts and crowds under no miming — it just grounds nothing.\n"
             f"The freshness dividend at $B$=5: stale fraction at baseline {stats[('open', 5)][1][2]:.2f} (open) vs "
             f"{stats[('firsthand', 5)][1][2]:.2f} (no miming); {stats[('firsthand', 5)][1][-1]:.2f} at its $\\alpha$=1 optimum.\n"
             "Curves of a budget meet at $\\alpha$=1, where there is no communication left to forbid. "
             "Mean ± s.e. over 10 seeds, last 200 of 800 steps.",
             ha="left", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0.02, 0.1, 1, 1))
    fig.savefig("fig_v4_policies.png", dpi=170)
    print("saved fig_v4_policies.png")
    for key in stats:
        print(key, [f"{a:g}:{y:.2f}" for a, y in zip(ALPHAS, stats[key][0])])


if __name__ == "__main__":
    main()
