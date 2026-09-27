"""Figure 1 (books era): mimesis, no mimesis, and an agent-written environment.

2x3. Columns: (1) no mimesis -- only self-observed evidence grounds an answer;
(2) mimesis -- standard open interaction; (3) both policies with BOOKS: every
observation is recorded in a shared manuscript, published in editions every
book_pub steps; agents spend part (open: half) or all (no mimesis) of their ask
budget reading the last edition, and a documented record grounds under both
policies. Rows: accuracy (top), P(IDK) (bottom). Only the observation share
alpha varies; everything else is at baseline (B = 5).
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
GREEN, GREY = "#1baf7a", "#444444"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10.5,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})

ALPHAS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.6, 1.0]
SEEDS = range(10)
BOOK_PUB = 100
BASE = dict(N=20, M=50, B=5, c_dec=LN2 / 20, chi=0.5, k_ctx=3, lam_mean=0.003, T=800)
CONDS = {  # name -> (answer_policy, books, book_frac)
    "nomime":       ("firsthand", False, 0.0),
    "mime":         ("open",      False, 0.0),
    "nomime_books": ("firsthand", True,  1.0),
    "mime_books":   ("open",      True,  0.5),
}


def cell(job):
    cond, a, s = job
    pol, books, bf = CONDS[cond]
    r = toy_v4.run(seed=s, alpha=a, answer_policy=pol, books=books, book_pub=BOOK_PUB,
                   book_frac=bf, **BASE)
    sl = slice(-40, None)
    return {"cond": cond, "alpha": a,
            **{k: float(r[k][sl].mean()) for k in ("idk", "stale", "correct", "dead_q", "sys_corr")}}


def curve(res, cond, metric):
    ys, se = [], []
    for a in ALPHAS:
        v = [r[metric] for r in res if r["cond"] == cond and r["alpha"] == a]
        ys.append(np.mean(v)); se.append(np.std(v) / np.sqrt(len(v)))
    return ys, se


def main():
    jobs = [(c, a, s) for c in CONDS for a in ALPHAS for s in SEEDS]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(cell, jobs, chunksize=4))
    json.dump(res, open("projects/data/books_results.json", "w"))

    fig, axes = plt.subplots(2, 3, figsize=(12.6, 7.0), sharex=True, sharey="row")
    STY = {"firsthand": dict(color=GREY, ls="--", marker="s", markerfacecolor="white"),
           "open": dict(color=GREEN, ls="-", marker="o")}
    COLS = [("No mimesis", ["nomime"]),
            ("Mimesis", ["mime"]),
            (f"Both, with books (editions every {BOOK_PUB} steps)", ["nomime_books", "mime_books"])]
    for ci, (title, conds) in enumerate(COLS):
        for ri, metric in enumerate(("correct", "idk")):
            ax = axes[ri][ci]
            if ci == 2:   # ghosts: the book-less counterparts, for the lift
                for ref in ("nomime", "mime"):
                    ys, _ = curve(res, ref, metric)
                    pol = CONDS[ref][0]
                    ax.plot(ALPHAS, ys, color=STY[pol]["color"], ls=STY[pol]["ls"],
                            lw=1.0, alpha=0.3)
            for cond in conds:
                pol = CONDS[cond][0]
                ys, se = curve(res, cond, metric)
                ax.errorbar(ALPHAS, ys, yerr=se, lw=1.8, ms=5, capsize=0, **STY[pol])
            ax.set_xscale("log"); ax.set_ylim(-0.03, 1.03)
            ax.axvline(0.01, color=MUTED, ls=":", lw=1.1)
            if ri == 0:
                ax.set_title(title, loc="left", fontweight="bold")
            else:
                ax.set_xticks(ALPHAS)
                ax.set_xticklabels([f"{a * 100:g}%" for a in ALPHAS], rotation=25)
                ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
                ax.set_xlabel("Share of budget observing, $\\alpha$")
    axes[0][0].set_ylabel("P(correct answer)")
    axes[1][0].set_ylabel("P(“I don’t know”)")
    axes[0][0].text(-0.26, 0.5, "Accuracy", transform=axes[0][0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, fontweight="bold", color=INK)
    axes[1][0].text(-0.26, 0.5, "Availability", transform=axes[1][0].transAxes, rotation=90,
                    va="center", ha="center", fontsize=10, fontweight="bold", color=INK)
    handles = [Line2D([], [], lw=1.8, ms=5, **STY["open"], label="mimesis (open grounding)"),
               Line2D([], [], lw=1.8, ms=5, **STY["firsthand"], label="no mimesis (first-hand only)"),
               Line2D([], [], color=MUTED, lw=1.0, alpha=0.5, label="same policy without books (col. 3)")]
    axes[0][2].legend(handles=handles, loc="lower right")
    fig.suptitle("An agent-written record restores the collective: books give no-mimesis agents nearly everything mimesis provides",
                 fontsize=11, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005,
             "Toy v4 at baseline except $\\alpha$ ($N$=20, $M$=50, $B$=5, $\\Delta$=20, $k$=3, $\\bar\\lambda$=0.003, mesh).\n"
             "Books: every observation is recorded in a shared manuscript, published every "
             f"{BOOK_PUB} steps; reading the edition grounds under both policies and consumes ask budget "
             "(no mimesis: all asks read the book; mimesis: half).\n"
             "A book copy inserts and crowds like any answer. Mean ± s.e. over 10 seeds, last 200 of 800 steps.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0.015, 0.07, 1, 0.95))
    fig.savefig("projects/artifacts/figure1_mimesis_books.png", dpi=170)
    print("saved figure1_mimesis_books.png")


if __name__ == "__main__":
    main()
