"""Figure 1: a REAL system experiencing epistemic collapse (warm start).

ws-real-* runs: N=10 gpt-4o-mini agents, every agent seeded at t=0 with the
current truth of all 50 questions (firsthand); T=150; two conditions differing
only in memory decay (c=0.05 vs c=1.0), 3 seeds each. Rows: each seed's
fastest- and slowest-revising probe question. Columns: P(correct), P(IDK),
P(stale) over the 10 agents (sum to 1); mean over seeds."""
import glob
import json
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#0b0b0b", "#8a8984"
SYSTEMS = {  # label -> (color, linewidth, run glob); widths differ so coincident lines both show
    "slow forgetting ($c$=0.05)": ("#2a78d6", 3.4, "experiments/results/ws-real-c0.05-s*"),
    "fast forgetting ($c$=1.0)": ("#c94f3d", 1.8, "experiments/results/ws-real-c1.0-s*"),
}
IDK = ("i don't know",)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 12, "axes.labelsize": 12,
                     "legend.fontsize": 9, "legend.frameon": False})


def run_curves(d):
    """(ts, {fast|slow: 3 x T array}) for one run; extremes by realized revisions."""
    fs = sorted(glob.glob(d + "/snapshots/*meta.json"),
                key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    if len(fs) < 5:
        return None
    last = json.load(open(fs[-1]))
    revs = {q: (int(m.group(1)) if (m := re.search(r"\[rev (\d+)\]$", t)) else 0)
            for q, t in last["truths"].items()}
    qf = max(revs, key=revs.get)
    qs_ = min(revs, key=revs.get)
    out = {"fast": [], "slow": []}
    ts = []
    for f in fs:
        m = json.load(open(f))
        ts.append(m["step"])
        for key, q in (("fast", qf), ("slow", qs_)):
            j = m["questions"].index(q)
            truth = m["truths"].get(q)
            resp = [row[j] for row in m["responses"]]
            c = np.mean([r == truth for r in resp])
            i = np.mean([any(p in r.lower() for p in IDK) for r in resp])
            out[key].append((c, i, 1 - c - i))
    return np.array(ts), {k: np.array(v).T for k, v in out.items()}, revs[qf], revs[qs_]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 6.2), sharex=True, sharey=True)
    revinfo = {}
    for label, (col, lw, pat) in SYSTEMS.items():
        per_seed = [r for d in sorted(glob.glob(pat)) if (r := run_curves(d)) is not None]
        print(label, "runs:", len(per_seed), "fast revs:", [r[2] for r in per_seed],
              "slow revs:", [r[3] for r in per_seed])
        ts = per_seed[0][0]
        for ri, key in enumerate(("fast", "slow")):
            mean = np.mean([r[1][key] for r in per_seed], axis=0)
            for ci, name in enumerate(("P(correct)", "P(“I don’t know”)", "P(stale)")):
                ax = axes[ri][ci]
                ax.plot(ts, mean[ci], color=col, lw=lw, marker="o", ms=3.5,
                        label=label if (ri, ci) == (0, 0) else None)
                if ri == 0:
                    ax.set_title(name, loc="left")
                if ri == 1:
                    ax.set_xlabel("step")
        revinfo[label] = ([r[2] for r in per_seed], [r[3] for r in per_seed])
    axes[0][0].set_ylabel("fastest question", fontsize=11)
    axes[1][0].set_ylabel("slowest question", fontsize=11)
    axes[0][0].set_ylim(-0.03, 1.03)
    axes[0][0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("projects/artifacts/fig_collapse_example_real.png", dpi=170)
    print("saved fig_collapse_example_real.png")


if __name__ == "__main__":
    main()
