"""Example of a system EXPERIENCING epistemic collapse (Figure 1 of the summary).

Warm start on the validated surrogate platform: every agent begins with fresh,
current, firsthand knowledge of all M questions; then the world runs. Two
systems differing only in memory decay (slow c=0.05 vs fast c=1.0), tracked on
each seed's fastest- and slowest-revising question. Columns: P(correct),
P(IDK), P(stale) over the N=10 agents (the three sum to 1); mean over seeds.
Real-baseline parameters otherwise (N=10, M=50, B=11, E=3, lam_mean=0.01,
disp 1, firsthand policy)."""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import toy_v4

INK, MUTED = "#0b0b0b", "#8a8984"
SYSTEMS = {  # label -> (color, linewidth, c_dec); widths differ so coincident lines both show
    "slow forgetting ($c$=0.05)": ("#2a78d6", 3.4, 0.05),
    "fast forgetting ($c$=1.0)": ("#c94f3d", 1.8, 1.0),
}
SEEDS = 20
T = 150
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 12, "axes.labelsize": 12,
                     "legend.fontsize": 9, "legend.frameon": False})


def curves(c_dec):
    """Per-step (correct, idk, stale) for the fastest and slowest question,
    averaged over seeds (each seed tracks its own extreme questions)."""
    acc = {"fast": [], "slow": []}
    for s in range(SEEDS):
        # find this seed's extreme questions from the lambda draw (same rng order)
        probe = toy_v4.run(N=10, M=50, B=11, alpha=3 / 11, c_dec=c_dec, chi=0.5, k_ctx=3,
                           lam_mean=0.01, lam_disp=1.0, answer_policy="firsthand",
                           T=1, seed=s, record_every=1, warm_start=True)
        lam = probe["_state"]["lam"]
        qf, qs_ = int(np.argmax(lam)), int(np.argmin(lam))
        r = toy_v4.run(N=10, M=50, B=11, alpha=3 / 11, c_dec=c_dec, chi=0.5, k_ctx=3,
                       lam_mean=0.01, lam_disp=1.0, answer_policy="firsthand",
                       T=T, seed=s, record_every=1, warm_start=True, track=[qf, qs_])
        kn = np.array(r["track_knows"]); co = np.array(r["track_corr"])
        for i, key in enumerate(("fast", "slow")):
            acc[key].append(np.stack([co[:, i], 1 - kn[:, i], kn[:, i] - co[:, i]]))
        ts = np.array(r["t"])
    return ts, {k: np.mean(v, axis=0) for k, v in acc.items()}, lam


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.2), sharex=True, sharey=True)
    lams = {}
    for label, (col, lw, c_dec) in SYSTEMS.items():
        ts, res, lam = curves(c_dec)
        lams = lam
        for ri, key in enumerate(("fast", "slow")):
            for ci, name in enumerate(("P(correct)", "P(“I don’t know”)", "P(stale)")):
                ax = axes[ri][ci]
                ax.plot(ts, res[key][ci], color=col, lw=lw,
                        label=label if (ri, ci) == (0, 0) else None)
                if ri == 0:
                    ax.set_title(name, loc="left")
                if ri == 1:
                    ax.set_xlabel("step")
    axes[0][0].set_ylabel("fastest question\n($\\lambda_q \\approx 0.05$: revises every ~20 steps)", fontsize=11)
    axes[1][0].set_ylabel("slowest question\n($\\lambda_q \\approx 10^{-3}$: rarely revises)", fontsize=11)
    axes[0][0].set_ylim(-0.03, 1.03)
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_collapse_example.png", dpi=170)
    print("saved; typical lam range:", float(np.max(lams)), float(np.min(lams)))


if __name__ == "__main__":
    main()
