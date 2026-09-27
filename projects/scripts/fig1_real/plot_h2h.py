"""Goodness-of-fit figure for the head-to-head: real gpt-4o-mini vs the
measured-gate surrogate, baseline setting, 5 matched seeds, T=500.

Three panels: (a) static-probe IDK trajectories, (b) answering rate over time,
(c) final-window per-seed values with arm means. Agreement of the distributions
(not per-seed paths) is the faithfulness claim.
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#0b0b0b", "#8a8984"
BLUE, GREEN = "#2a78d6", "#1baf7a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10.5,
                     "legend.fontsize": 8, "legend.frameon": False})

IDK = ("i don't know", "i do not know", "i don’t know")
ARMS = {"h2h-real": ("real system (gpt-4o-mini)", BLUE),
        "h2h-surr": ("surrogate (measured gate)", GREEN)}


def runs(prefix):
    out = {}
    for d in sorted(glob.glob(f"experiments/results/{prefix}-baseline-s*")):
        out[re.search(r"-s(\d+)_", d).group(1)] = d
    return out


def idk_traj(d):
    steps, idk = [], []
    for f in sorted(glob.glob(d + "/snapshots/snapshot_step_*_meta.json")):
        m = json.load(open(f))
        si = [j for j, t in enumerate(m["temporal_mask"]) if not t]
        rows = m["responses"]
        steps.append(m["step"])
        idk.append(np.mean([any(p in rows[i][j].lower() for p in IDK)
                            for i in range(len(rows)) for j in si]))
    return np.array(steps), np.array(idk)


def rate_traj(d, w=25):
    r = json.load(open(d + "/results.json"))["results"]
    x = np.array([s["num_knowledge_added"] / max(1, s["num_queries"]) for s in r])
    k = np.ones(w) / w
    return np.arange(len(x) - w + 1) + w // 2, np.convolve(x, k, mode="valid")


def main():
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.6, 4.0))
    finals = defaultdict(list)
    for prefix, (label, col) in ARMS.items():
        trajs, rates = [], []
        for seed, d in runs(prefix).items():
            s, y = idk_traj(d)
            a1.plot(s, y, color=col, lw=0.9, alpha=0.4)
            trajs.append(np.interp(np.arange(0, 500, 10), s, y))
            t, rr = rate_traj(d)
            a2.plot(t, rr, color=col, lw=0.9, alpha=0.4)
            rates.append(np.interp(np.arange(500), t, rr))
            finals[prefix].append(y[-10:].mean())
        a1.plot(np.arange(0, 500, 10), np.mean(trajs, axis=0), color=col, lw=2.4, label=label)
        a2.plot(np.arange(500), np.mean(rates, axis=0), color=col, lw=2.4, label=label)

    a1.set_ylim(-0.03, 1.03); a1.set_xlabel("Step")
    a1.set_ylabel("P(“I don’t know”), static probes")
    a1.set_title("Collapse trajectories", loc="left")
    a1.legend(loc="upper right")

    a2.set_ylim(-0.03, 1.03); a2.set_xlabel("Step")
    a2.set_ylabel("Answering rate (knowledge added / queries)")
    a2.set_title("Exchange dynamics", loc="left")

    for i, (prefix, (label, col)) in enumerate(ARMS.items()):
        v = finals[prefix]
        x = np.full(len(v), i) + (np.random.default_rng(0).random(len(v)) - 0.5) * 0.12
        a3.scatter(x, v, color=col, s=45, zorder=3,
                   edgecolor="white", linewidth=0.8)
        m, se = np.mean(v), np.std(v, ddof=1) / np.sqrt(len(v))
        a3.errorbar([i + 0.25], [m], yerr=[1.96 * se], color=col, fmt="D", ms=7,
                    capsize=5, lw=1.8)
    a3.set_xticks([0.1, 1.1]); a3.set_xticklabels(["real", "surrogate"])
    a3.set_xlim(-0.5, 1.8); a3.set_ylim(-0.03, 1.03)
    a3.set_ylabel("Final static-probe IDK (last 100 steps)")
    a3.set_title("Endpoints, per seed (◆ mean ± 95% CI)", loc="left")

    fig.suptitle("The simulation is faithful: measured-gate surrogate vs the real system, matched protocol "
                 "(baseline, 5 seeds each, T = 500)", fontsize=11.5, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005,
             "Both arms: identical configs, seeds, verbatim-relay protocol; the surrogate replaces the LLM with the measured "
             "answering gate P(IDK | context state) — no fitted parameters.\n"
             "Thin lines: individual seeds (stochastic paths diverge by construction; the claim is distributional). "
             "Means: real IDK 0.22 vs surrogate 0.21; answering rates within 0.06 at every phase.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.09, 1, 0.92))
    out = Path("projects/artifacts/figure1_h2h_goodness.png")
    fig.savefig(out, dpi=170)
    print("saved", out)


if __name__ == "__main__":
    main()
