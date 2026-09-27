"""Figure 2 (final): the simulation matches the real system, per factor.

1x4 calibration panels, one per component (agent / communication / environment /
observation): that factor's three settings x four metrics as real-vs-simulation
points on the diagonal, with the simulation's 10-seed IQR (horizontal) and the
real seed range (vertical). No fitted parameters anywhere in the chain.
"""
import glob
import json
import re
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#0b0b0b", "#8a8984"
HUE = {"Agent": "#2a78d6", "Communication": "#1baf7a", "Environment": "#eb6834", "Observation": "#4a3aa7"}
MET = {"agent IDK": (0, "o"), "agent correct": (1, "^"), "system IDK": (2, "s"), "system correct": (3, "D")}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 10.5,
                     "legend.fontsize": 7.5, "legend.frameon": False})
IDK = ("i don't know",)
FACTORS = {"Agent": ["c0.05", "base", "c1.0"],
           "Communication": ["deg2", "base", "mesh"],
           "Environment": ["lam.001", "base", "lam.1"],
           "Observation": ["E1", "base", "E6"]}


def four(d):
    fs = sorted(glob.glob(d + "/snapshots/snapshot_step_*_meta.json"))[-10:]
    if len(fs) < 5:
        return None
    a_i = a_a = s_i = s_a = c = 0
    for f in fs:
        m = json.load(open(f))
        truths = m.get("truths", {})
        qs, rows = m["questions"], m["responses"]
        for j, q in enumerate(qs):
            col = [rows[i][j] for i in range(len(rows))]
            idks = [any(p in r.lower() for p in IDK) for r in col]
            accs = [r == truths.get(q) for r in col]
            a_i += np.mean(idks); a_a += np.mean(accs)
            s_i += all(idks); s_a += any(accs); c += 1
    return np.array([a_i, a_a, s_i, s_a]) / c


def main():
    data = {"surr": defaultdict(list), "real": defaultdict(list)}
    for d in glob.glob("experiments/results/f1v6-*"):
        m_ = re.match(r".*/f1v6-(surr|real)-(.+)-s\d+_", d)
        v = four(d)
        if v is not None:
            data[m_.group(1)][m_.group(2)].append(v)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.7), sharey=True)
    for ax, (fac, cells) in zip(axes, FACTORS.items()):
        hue = HUE[fac]
        devs = []
        for cell in cells:
            S = np.array(data["surr"][cell]); R = np.array(data["real"][cell])
            for name, (i, mk) in MET.items():
                sx = np.median(S[:, i]); sq = np.percentile(S[:, i], [25, 75])
                ry = np.median(R[:, i]); rr = [R[:, i].min(), R[:, i].max()]
                ax.errorbar([sx], [ry],
                            xerr=[[sx - sq[0]], [sq[1] - sx]],
                            yerr=[[ry - rr[0]], [rr[1] - ry]],
                            fmt=mk, color=hue, ms=6.5, mfc=(hue if "IDK" in name else "white"),
                            mec=hue, capsize=2.5, lw=1.1, zorder=4)
                devs.append(abs(sx - ry))
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, zorder=1)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03)
        ax.set_aspect("equal")
        ax.set_title(fac, loc="left", color=hue, fontweight="bold")
        ax.set_xlabel("observed value (proxy)")
        ax.text(0.97, 0.03, f"median |Δ| = {np.median(devs):.2f}", transform=ax.transAxes,
                ha="right", fontsize=8, color=INK)
    axes[0].set_ylabel("observed value (real)")
    h = [Line2D([], [], color=INK, marker=mk, ls="", ms=6.5,
                mfc=(INK if "IDK" in n else "white"), mec=INK, label=n) for n, (i, mk) in MET.items()]
    axes[0].legend(handles=h, loc="upper left")
    fig.suptitle("Figure 2 — the simulation matches the real system, per factor: every cell × metric on the diagonal "
                 "(bars: simulation IQR over 10 seeds ↔ real seed range)",
                 fontsize=11, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005,
             "Measured answering gate + verbatim relay + mechanical retrieval; no fitted parameters. "
             "Each panel: that factor's three settings (incl. the shared baseline) × four metrics. "
             "Diagonal = perfect agreement. Same protocol and runs as Figure 1.",
             ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.12, 1, 0.9))
    fig.savefig("projects/artifacts/figure2_final.png", dpi=170)
    print("saved figure2_final.png")


if __name__ == "__main__":
    main()
