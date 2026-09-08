"""E0 figure: collapse in the unmodified fastsim (Q sweep). Three panels:
(1) final knowledge per agent vs Q, (2) probe-panel IDK vs Q with the alive
fraction overlaid (system level: flat 1), (3) knowledge trajectories by Q.
Encoding follows the project figures: one hue, lightness = level, solid/dashed = N."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

HERE = Path(__file__).parent
HUE = "#2a78d6"  # agent-factor blue from the factor figures
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})


def shade(hex_color, f):
    """f in [0,1]: 0 = lightest usable tint, 1 = full hue."""
    r, g, b = to_rgb(hex_color)
    return tuple(1 - (1 - v) * (0.30 + 0.70 * f) for v in (r, g, b))


def load():
    cells = defaultdict(list)  # (N, Q) -> list of cell dicts
    for f in sorted((HERE / "summaries").glob("*.json")):
        d = json.loads(f.read_text())
        cfg = d["config"]
        cells[(cfg["agents"]["count"], cfg["data"]["total_questions"])].append(d)
    return cells


def se(a):
    a = np.asarray(a, float)
    return a.mean(), a.std(ddof=1) / np.sqrt(len(a))


def main():
    cells = load()
    QS = sorted({q for _, q in cells})
    NS = sorted({n for n, _ in cells})
    ls_for_n = {n: s for n, s in zip(NS, ["-", "--"])}
    mk_for_n = {n: m for n, m in zip(NS, ["o", "s"])}

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9))

    # Panel 1: final knowledge per agent / Q
    ax = axes[0]
    for n in NS:
        qs = [q for q in QS if (n, q) in cells]
        m, e = zip(*[se([c["mean_known_final"] / q for c in cells[(n, q)]]) for q in qs])
        ax.errorbar(qs, m, yerr=e, color=HUE if n == max(NS) else shade(HUE, 0.45),
                    ls=ls_for_n[n], marker=mk_for_n[n], ms=4.5, lw=1.6, capsize=2,
                    label=f"$N$ = {n}", mfc="white" if ls_for_n[n] == "--" else None)
    ax.set_xscale("log"); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Questions in play, $Q$")
    ax.set_ylabel("Final fraction of pool known per agent")
    ax.set_title("Knowledge at $T$ = 2000", loc="left")
    ax.axvline(5 * 17, color=MUTED, ls=":", lw=1)  # K * SAME_Q_VISIBILITY
    ax.text(5 * 17, 0.02, " predicted crowding drop\n ($K\\cdot$vis = 85): absent", color=MUTED, fontsize=7)
    ax.annotate("40% of statics\nnever seeded", xy=(200, 0.70), xytext=(90, 0.48), fontsize=7,
                color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.annotate("still learning\n(~1 q/step)", xy=(2000, 0.505), xytext=(700, 0.30), fontsize=7,
                color=MUTED, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.legend(loc="lower left")

    # Panel 2: probe IDK (agent level) and alive fraction (system level)
    ax = axes[1]
    for n in NS:
        qs = [q for q in QS if (n, q) in cells]
        col = HUE if n == max(NS) else shade(HUE, 0.45)
        m, e = zip(*[se([c["snapshots"][-1]["idk_all"] for c in cells[(n, q)]]) for q in qs])
        ax.errorbar(qs, m, yerr=e, color=col, ls=ls_for_n[n], marker=mk_for_n[n], ms=4.5,
                    lw=1.6, capsize=2, mfc="white" if ls_for_n[n] == "--" else None)
        a, ea = zip(*[se([c["snapshots"][-1]["alive_all"] for c in cells[(n, q)]]) for q in qs])
        ax.errorbar(qs, a, yerr=ea, color=col, ls=ls_for_n[n], marker=mk_for_n[n], ms=3,
                    lw=0.9, alpha=0.55, capsize=0)
    ax.set_xscale("log"); ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Questions in play, $Q$")
    ax.set_ylabel("Fraction at $t = 2000$")
    ax.set_title("No question ever dies", loc="left")
    ax.legend(handles=[Line2D([], [], color=INK, lw=1.6, label="P(IDK) on probes (agent level)"),
                       Line2D([], [], color=INK, lw=0.9, alpha=0.55, label="questions alive somewhere (system level)")],
              loc="center left")

    # Panel 3: trajectories, mean_known(t)/Q
    ax = axes[2]
    for n in NS:
        for i, q in enumerate([q for q in QS if (n, q) in cells]):
            traj = np.mean([np.asarray(c["mean_known"]) / q for c in cells[(n, q)]], axis=0)
            t = np.arange(len(traj)) * 10
            ax.plot(t, traj, color=shade(HUE, i / (len(QS) - 1)), ls=ls_for_n[n], lw=1.4,
                    label=f"$Q$ = {q}" if n == max(NS) else None)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Step")
    ax.set_ylabel("Fraction of pool known per agent")
    ax.set_title("S-curves shift right; nothing freezes", loc="left")
    ax.legend(loc="lower right", title="$N$ = 100 (dashed), light → dark", title_fontsize=7)

    fig.suptitle("E0 — the unmodified simulator does not collapse: IDK is acquisition lag or missing seeds, never loss",
                 fontsize=11, x=0.01, ha="left", y=0.99)
    fig.text(0.01, 0.005,
             "Unmodified fastsim, full mesh, $K$=5, decay strategy (0.05, additive), $n_{temporal}$=0.2$Q$, "
             "each static question seeded exactly once, $T$=2000. Solid circles = $N$=5, dashed squares = $N$=100 "
             "($N$=100 with $Q\\leq$100 omitted: questions/agent < 1). Mean ± s.e. over 30 seeds.\n"
             "Alive = answered substantively by ≥1 panel agent; the panel is all agents. "
             "Memory never expires, so knowledge only accumulates: growth per question compounds at ~$K/Q$ per step.",
             ha="left", va="bottom", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    out = HERE / "fig_e0_dilution.png"
    fig.savefig(out, dpi=170)
    print("saved", out)

    # console summary table
    print(f"\n{'N':>4} {'Q':>5} {'known/Q':>8} {'IDK':>6} {'alive':>6} {'seeds':>5}")
    for n in NS:
        for q in QS:
            if (n, q) not in cells: continue
            cs = cells[(n, q)]
            print(f"{n:>4} {q:>5} {np.mean([c['mean_known_final']/q for c in cs]):>8.3f} "
                  f"{np.mean([c['snapshots'][-1]['idk_all'] for c in cs]):>6.3f} "
                  f"{np.mean([c['snapshots'][-1]['alive_all'] for c in cs]):>6.3f} {len(cs):>5}")


if __name__ == "__main__":
    main()
