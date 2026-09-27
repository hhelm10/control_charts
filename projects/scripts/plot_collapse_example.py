"""Example of epistemic collapse in a real system (Section 2.1 figure).

Two real gpt-4o-mini runs identical except memory decay: healthy (c=0.05) vs
collapsing (c=1.0). Three panels: (a) agent trajectories in perspective space
(each agent's mean probe-response embedding per snapshot, PCA to 2D over both
runs; the "I don't know" point marked); (b) average accuracy over time;
(c) average P(IDK) over time. Embeddings via the answer-alphabet lookup."""
from pathlib import Path
import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED = "#0b0b0b", "#8a8984"
HEALTHY, COLLAPSED = "#2a78d6", "#c94f3d"
ALPHABET = str(Path(__file__).resolve().parents[1] / "data" / "answer_alphabet.npz")
RUNS = {  # label -> (color, run dir glob)
    "healthy ($c$=0.05)": (HEALTHY, "experiments/results/f1v6-real-c0.05-s42_*"),
    "collapsing ($c$=1.0)": (COLLAPSED, "experiments/results/f1v6-real-c1.0-s42_*"),
}
IDK = ("i don't know",)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 13, "axes.labelsize": 12,
                     "legend.fontsize": 9, "legend.frameon": False})


def load_alphabet():
    z = np.load(ALPHABET, allow_pickle=True)
    lut = {t: e for t, e in zip(z["texts"], z["embeddings"])}
    def emb(s):
        return lut[re.sub(r" \[rev \d+\]$", "", s)]
    return emb


def snapshots(pattern):
    d = [p for p in glob.glob(pattern) if len(glob.glob(p + "/snapshots/*meta.json")) > 10][0]
    fs = sorted(glob.glob(d + "/snapshots/snapshot_step_*_meta.json"),
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    return [(int(re.search(r"step_(\d+)", f).group(1)), json.load(open(f))) for f in fs]


def main():
    emb = load_alphabet()
    data = {}
    for label, (col, pat) in RUNS.items():
        ts, psis, accs, idks = [], [], [], []
        for t, m in snapshots(pat):
            truths, qs, rows = m.get("truths", {}), m["questions"], m["responses"]
            psis.append([np.mean([emb(r) for r in row], axis=0) for row in rows])
            accs.append([np.mean([r == truths.get(q) for q, r in zip(qs, row)]) for row in rows])
            idks.append([np.mean([any(p in r.lower() for p in IDK) for r in row]) for row in rows])
            ts.append(t)
        data[label] = (col, np.array(ts), np.array(psis), np.array(accs), np.array(idks))

    # shared PCA over all (agent, time) perspectives from both runs + the IDK point
    idk_vec = emb("I don't know")
    allpts = np.concatenate([p.reshape(-1, p.shape[-1]) for _, _, p, _, _ in data.values()]
                            + [idk_vec[None]])
    mu = allpts.mean(axis=0)
    _, _, Vt = np.linalg.svd(allpts - mu, full_matrices=False)
    proj = lambda X: (X - mu) @ Vt[:2].T

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2),
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    axA, axB, axC = axes
    W = 7  # moving-average window (snapshots) for legible trajectories
    for label, (col, ts, psis, accs, idks) in data.items():
        P = proj(psis.reshape(-1, psis.shape[-1])).reshape(len(ts), -1, 2)
        k = np.ones(W) / W
        Ps = np.stack([[np.convolve(P[:, a, d], k, mode="valid") for d in (0, 1)]
                       for a in range(P.shape[1])])  # agents x 2 x T'
        for a in range(P.shape[1]):
            axA.plot(Ps[a, 0], Ps[a, 1], color=col, lw=1.1, alpha=0.6, zorder=2)
            axA.scatter(Ps[a, 0, -1], Ps[a, 1, -1], color=col, s=26, zorder=4)
            axA.scatter(Ps[a, 0, 0], Ps[a, 1, 0], facecolor="white", edgecolor=col,
                        s=26, lw=1.2, zorder=4)
        axB.plot(ts, accs.mean(axis=1), color=col, lw=2.0, label=label)
        axC.plot(ts, idks.mean(axis=1), color=col, lw=2.0)
    ip = proj(idk_vec[None])[0]
    axA.scatter(*ip, marker="*", s=200, color=INK, zorder=5)
    axA.annotate("“I don't know”", ip, textcoords="offset points", xytext=(8, 6), fontsize=9)
    axA.set_xlabel("PC 1"); axA.set_ylabel("PC 2")
    axA.set_title("agent trajectories", loc="left")
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec=INK, label="start"),
         Line2D([], [], marker="o", ls="", color=INK, label="end")]
    axA.legend(handles=h, loc="lower right")
    axB.set_title("average accuracy", loc="left")
    axB.set_xlabel("step"); axB.set_ylabel("P(correct)")
    axB.set_ylim(-0.03, 1.03); axB.legend(loc="upper left")
    axC.set_title("average abstention", loc="left")
    axC.set_xlabel("step"); axC.set_ylabel("P(“I don’t know”)")
    axC.set_ylim(-0.03, 1.03)
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_collapse_example.png", dpi=170)
    print("saved fig_collapse_example.png")


if __name__ == "__main__":
    main()
