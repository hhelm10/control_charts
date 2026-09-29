"""Figure 3: hyperparameter effects on epistemic collapse for the three societies.

Societies: no mimesis (ground on own observation only), mimesis (standard open
interaction), book (no mimesis + all asks read the shared edition; a documented
record grounds). 2x4: rows = accuracy / P(IDK); columns = one hyperparameter each
(memory decay c, demand concentration s_ask, environment speed lambda, world size
M), everything else at baseline (alpha = 1%, B = 5, editions every 100 steps).
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor
import toy_v4

LN2 = float(np.log(2.0))
INK, MUTED = "#0b0b0b", "#8a8984"
GREEN, GREY, BLUE = "#1baf7a", "#444444", "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10.5,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})

SEEDS = range(10)
BOOK_PUB = 100
BASE = dict(N=20, M=50, B=5, alpha=0.01, c_dec=LN2 / 20, sim_floor=0.5, k_ctx=3,
            lam_mean=0.003, T=800)
SOCIETIES = {  # name -> (answer_policy, books, book_frac, style)
    "no mimesis": ("firsthand", False, 0.0, dict(color=GREY, ls="--", marker="s", markerfacecolor="white")),
    "mimesis":    ("open",      False, 0.0, dict(color=GREEN, ls="-", marker="o")),
    "book":       ("firsthand", True,  1.0, dict(color=BLUE, ls="-.", marker="^")),
}
PANELS = {
    "c_dec":    dict(xs=[round(LN2 / t, 4) for t in (99, 50, 20, 10, 5)],
                     lab="Memory decay rate $c$", log=True,
                     tick=lambda v: f"{v:g}\n$\\Delta$={LN2 / v:.0f}"),
    "s_ask":    dict(xs=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
                     lab="Demand concentration (Zipf of $\\pi_{ask}$)", log=False,
                     tick=lambda v: f"{v:g}"),
    "lam_mean": dict(xs=[0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
                     lab="Environment speed $\\bar\\lambda$", log=True,
                     tick=lambda v: f"{v:g}"),
    "M":        dict(xs=[25, 50, 100, 200, 400],
                     lab="World size $M$ (questions)", log=True,
                     tick=lambda v: f"{v:g}"),
}


def cell(job):
    soc, key, x, s = job
    pol, books, bf, _ = SOCIETIES[soc]
    kw = dict(BASE); kw[key] = x
    r = toy_v4.run(seed=s, answer_policy=pol, books=books, book_pub=BOOK_PUB, book_frac=bf, **kw)
    sl = slice(-40, None)
    return {"soc": soc, "key": key, "x": x,
            **{k: float(r[k][sl].mean()) for k in ("idk", "stale", "correct", "dead_q", "sys_corr")}}


def main():
    jobs = [(soc, key, x, s) for soc in SOCIETIES for key, p in PANELS.items()
            for x in p["xs"] for s in SEEDS]
    print(len(jobs), "runs")
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=4))
    json.dump(res, open("projects/data/figure3_results.json", "w"))

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.0), sharey="row")
    for ci, (key, p) in enumerate(PANELS.items()):
        for ri, metric in enumerate(("correct", "idk")):
            ax = axes[ri][ci]
            for soc, (pol, books, bf, sty) in SOCIETIES.items():
                ys, se = [], []
                for x in p["xs"]:
                    v = [r[metric] for r in res if r["soc"] == soc and r["key"] == key and r["x"] == x]
                    ys.append(np.mean(v)); se.append(np.std(v) / np.sqrt(len(v)))
                ax.errorbar(p["xs"], ys, yerr=se, lw=1.8, ms=5, capsize=0, **sty)
            if p["log"]:
                ax.set_xscale("log")
            ax.set_xticks(p["xs"])
            ax.set_xticklabels([p["tick"](v) for v in p["xs"]], fontsize=7)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_ylim(-0.03, 1.03)
            if key in BASE:
                ax.axvline(BASE[key], color=MUTED, ls=":", lw=1.1)
            if ri == 0:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(p["lab"])
    axes[0][0].set_ylabel("P(correct answer)")
    axes[1][0].set_ylabel("P(“I don’t know”)")
    axes[0][0].text(-0.3, 0.5, "Accuracy", transform=axes[0][0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, fontweight="bold", color=INK)
    axes[1][0].text(-0.3, 0.5, "Availability", transform=axes[1][0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, fontweight="bold", color=INK)
    handles = [Line2D([], [], lw=1.8, ms=5, **sty, label=soc)
               for soc, (_, _, _, sty) in SOCIETIES.items()]
    axes[0][0].legend(handles=handles, loc="lower left")
    fig.suptitle("Figure 3 — how each hyperparameter drives epistemic collapse, by society "
                 "(no mimesis / mimesis / book)", fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005,
             "Toy v4; one hyperparameter varies per column, all else at baseline (dotted): $N$=20, $M$=50, $B$=5, $\\alpha$=1%, "
             "$\\Delta$=20, $k$=3, $\\bar\\lambda$=0.003, mesh, staleness-aware asking.\n"
             "Societies: no mimesis = only self-observed evidence grounds; mimesis = any retrievable entry grounds; "
             "book = no mimesis whose asks all read a shared agent-written record (editions every "
             f"{BOOK_PUB} steps). Mean ± s.e. over 10 seeds, last 200 of 800 steps.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0.015, 0.06, 1, 0.95))
    fig.savefig("projects/artifacts/figure3_hyperparams_policies.png", dpi=170)
    print("saved figure3_hyperparams_policies.png")


if __name__ == "__main__":
    main()
