"""Figure 1 analysis: real-system (gpt-4o-mini) probe-IDK trajectories per setting,
with zero-free-parameter toy-v4 overlays calibrated from measurables:
chi = 0.6 (k-th best off-target nomic cosine), c = decay coefficient,
k_ctx = retrieval_k, B = questions_per_turn, M = total_questions, N = 10.
Toy approximations: no observation channel (alpha ~ 0; the real system's only
environment input is temporal-question ownership), repo ask policy
(unknown-first with decay re-ask), evidence-merge, no books.
"""
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
import toy_v4

INK, MUTED = "#0b0b0b", "#8a8984"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10,
                     "legend.fontsize": 7.5, "legend.frameon": False})

CHI = 0.30   # fitted on the baseline arm's static IDK; all other settings out-of-sample
SETTINGS = {  # name -> (title, toy overrides)
    "baseline":  ("baseline (Q=50, c=0.05, K=5, k=3)", {}),
    "q200":      ("Q = 200", dict(M=200)),
    "q500":      ("Q = 500", dict(M=500)),
    "decay0.2":  ("decay c = 0.2", dict(c_dec=0.2)),
    "decay0.5":  ("decay c = 0.5", dict(c_dec=0.5)),
    "k1bw":      ("bandwidth K = 1", dict(B=1)),
    "ret1":      ("context k = 1", dict(k_ctx=1)),
    "deg2":      ("mean degree 2", dict(mean_degree=2)),
    "deg6":      ("mean degree 6", dict(mean_degree=6)),
}
TOY_BASE = dict(N=10, M=50, B=5, alpha=0.0, c_dec=0.05, chi=CHI, k_ctx=3,
                lam_mean=1e-9, lam_disp=0.0,
                sigma="unknown_first", answer_policy="open", T=1000)

IDK_PAT = ("i don't know", "i do not know", "i don’t know")


def best_runs():
    best = defaultdict(lambda: (0, None))
    for d in glob.glob(str(Path(__file__).parents[3] / "experiments" / "results" / "fig1-*")):
        cfg = re.match(r".*/(fig1-.+-s\d+)_", d).group(1)
        n = len(glob.glob(d + "/snapshots/snapshot_step_*_meta.json"))
        if n > best[cfg][0]:
            best[cfg] = (n, d)
    return {c: d for c, (n, d) in best.items() if n >= 85}


def idk_trajectory(run_dir):
    """Static probes only: the temporal half has an ownership channel the toy does not model."""
    steps, idk = [], []
    for f in sorted(glob.glob(run_dir + "/snapshots/snapshot_step_*_meta.json")):
        m = json.load(open(f))
        si = [j for j, t in enumerate(m["temporal_mask"]) if not t]
        rows = m["responses"]
        n = sum(any(p in rows[i][j].lower() for p in IDK_PAT)
                for i in range(len(rows)) for j in si)
        steps.append(m["step"])
        idk.append(n / (len(rows) * len(si)))
    return np.array(steps), np.array(idk)


def toy_overlay(mods, seeds=8):
    kw = dict(TOY_BASE); kw.update(mods)
    tr = [toy_v4.run(seed=s, record_every=10, **kw)["idk"] for s in range(seeds)]
    t = np.arange(1, len(tr[0]) + 1) * 10
    return t, np.mean(tr, axis=0)


def main():
    runs = best_runs()
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.2), sharex=True, sharey=True)
    summary = []
    for ax, (name, (title, mods)) in zip(axes.flat, SETTINGS.items()):
        seed_finals = []
        for cfg, d in sorted(runs.items()):
            if re.match(rf"fig1-{re.escape(name)}-s\d+$", cfg):
                s, y = idk_trajectory(d)
                ax.plot(s, y, color=BLUE, lw=1.0, alpha=0.45)
                seed_finals.append(y[-20:].mean())
        tt, ty = toy_overlay(mods)
        ax.plot(tt, ty, color=ORANGE, lw=1.8, ls="--")
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.03)
        if seed_finals:
            summary.append((name, np.mean(seed_finals), ty[-20:].mean(), len(seed_finals)))
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    for row in axes:
        row[0].set_ylabel("P(“I don’t know”), static probes")
    from matplotlib.lines import Line2D
    axes[0][0].legend(handles=[
        Line2D([], [], color=BLUE, lw=1.0, alpha=0.6, label="real system (gpt-4o-mini), per seed"),
        Line2D([], [], color=ORANGE, lw=1.8, ls="--", label="toy v4 (one parameter fitted on baseline)")],
        loc="upper right")
    fig.suptitle("Figure 1 — epistemic collapse in a real multi-agent system across settings, with calibrated simulation overlays",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005,
             "Real system: N=10 gpt-4o-mini agents, nomic+FAISS retrieval, paper mechanics; probe panel every 10 steps, static probes only (the temporal half has an ownership channel the toy does not model). "
             f"Toy v4 with ONE fitted parameter, $\\chi_{{eff}}$={CHI} (calibrated on the baseline arm; all other panels out-of-sample;\n"
             "c = decay coefficient, $k$ = retrieval_k, $B$ = questions/turn carried over directly); no observation channel, unknown-first asking. "
             "Blue: individual seeds. Orange dashed: toy mean of 8 seeds.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    out = Path(__file__).parents[2] / "artifacts" / "figure1_real_vs_sim.png"
    fig.savefig(out, dpi=170)
    print("saved", out)
    print(f"\n{'setting':>12} {'real IDK':>9} {'toy IDK':>8} {'seeds':>6}")
    for name, r, t, n in summary:
        print(f"{name:>12} {r:>9.2f} {t:>8.2f} {n:>6}")


if __name__ == "__main__":
    main()
