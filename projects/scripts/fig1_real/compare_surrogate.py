"""Faithfulness tier 1: the surrogate (measured gate + verbatim payload + real
FAISS/embeddings, no LLM) against the real gpt-4o-mini runs, per setting.

Static-probe IDK trajectories and run-level P[collapse] (final-window static
IDK >= 0.5). Real runs: 19 completed (old protocol). Surrogate: 10 seeds/setting.
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
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent))
from analyze_fig1 import idk_trajectory, SETTINGS

INK, MUTED = "#0b0b0b", "#8a8984"
BLUE, GREEN = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10,
                     "legend.fontsize": 7.5, "legend.frameon": False})

RESULTS = Path(__file__).parents[3] / "experiments" / "results"


def runs_for(prefix):
    best = defaultdict(lambda: (0, None))
    for d in glob.glob(str(RESULTS / f"{prefix}-*")):
        cfg = re.match(rf".*/({prefix}-.+-s\d+)_", d).group(1)
        n = len(glob.glob(d + "/snapshots/snapshot_step_*_meta.json"))
        if n > best[cfg][0]:
            best[cfg] = (n, d)
    return {c: d for c, (n, d) in best.items() if n >= 85}


def main():
    real, surr = runs_for("fig1"), runs_for("surr")
    fig, axes = plt.subplots(3, 3, figsize=(12.5, 9.2), sharex=True, sharey=True)
    print(f"{'setting':>10} {'real: mean idk':>15} {'P[col]':>7} {'n':>3} | "
          f"{'surr: mean idk':>15} {'P[col]':>7} {'n':>3}")
    for ax, (name, (title, _)) in zip(axes.flat, SETTINGS.items()):
        stats = {}
        for prefix, runs, color, lw, al in (("fig1", real, BLUE, 1.1, 0.55),
                                            ("surr", surr, GREEN, 0.8, 0.4)):
            finals = []
            for cfg, d in sorted(runs.items()):
                if re.match(rf"{prefix}-{re.escape(name)}-s\d+$", cfg):
                    s, y = idk_trajectory(d)
                    ax.plot(s, y, color=color, lw=lw, alpha=al)
                    finals.append(y[-20:].mean())
            stats[prefix] = finals
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.03)
        r, g = stats.get("fig1", []), stats.get("surr", [])
        if r or g:
            pc = lambda v: np.mean([x >= 0.5 for x in v]) if v else float("nan")
            mn = lambda v: np.mean(v) if v else float("nan")
            print(f"{name:>10} {mn(r):>15.2f} {pc(r):>7.2f} {len(r):>3} | "
                  f"{mn(g):>15.2f} {pc(g):>7.2f} {len(g):>3}")
    for ax in axes[-1]:
        ax.set_xlabel("Step")
    for row in axes:
        row[0].set_ylabel("P(“I don’t know”), static probes")
    axes[0][0].legend(handles=[
        Line2D([], [], color=BLUE, lw=1.1, label="real system (gpt-4o-mini), per seed"),
        Line2D([], [], color=GREEN, lw=0.8, label="surrogate (measured gate, no LLM), per seed")],
        loc="upper right")
    fig.suptitle("Faithfulness tier 1 — the measured-gate surrogate against the real system, per setting",
                 fontsize=11.5, x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.005,
             "Surrogate: identical pipeline (nomic embeddings, FAISS ranking, decayed retrieval, verbatim payload) with the LLM replaced "
             "by the measured gate P(IDK | present, k, position); 10 seeds/setting.\n"
             "Real: the 19 completed gpt-4o-mini runs (pre-relay protocol). Static probes only. "
             "P[collapse] = fraction of seeds with final-window static IDK ≥ 0.5.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    out = Path(__file__).parents[2] / "artifacts" / "figure1_surrogate_vs_real.png"
    fig.savefig(out, dpi=170)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
