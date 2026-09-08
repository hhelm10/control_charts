import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import defaultdict

D = json.load(open("rarity_results.json")); RES = D["results"]; B = D["base"]
HUE = {"environment": "#eb6834", "demand": "#1baf7a", "both": "#4a3aa7"}
COUP = {"same": "#4a3aa7", "independent": "#2a78d6", "reversed": "#eda100"}
INK, MUTED = "#0b0b0b", "#8a8984"
plt.rcParams.update({"font.size": 9.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "legend.fontsize": 8, "legend.frameon": False})

d = defaultdict(list)
for r in RES:
    d[(r["coupling_arm"], r["cond"], r["s"])].append(r)
S = sorted({r["s"] for r in RES})

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

# (a) mean IDK vs skew, uniform-weighted (filled) and demand-weighted (hollow)
ax = axes[0]
for cond in ("environment", "demand", "both"):
    for key, st in (("idk", dict(ls="-", marker="o")), ("idk_demand", dict(ls="--", marker="s"))):
        ys = [np.mean([r[key] for r in d[(False, cond, s)]]) for s in S]
        se = [np.std([r[key] for r in d[(False, cond, s)]]) / np.sqrt(12) for s in S]
        ax.errorbar(S, ys, yerr=se, color=HUE[cond], lw=1.8, ms=5.5, elinewidth=1, capsize=0, **st,
                    markerfacecolor=HUE[cond] if key == "idk" else "white", markeredgecolor=HUE[cond], markeredgewidth=1.4)
ax.set_xlabel("Zipf exponent $s$ of the skewed distribution (0 = uniform)")
ax.set_ylabel("“I don’t know” fraction"); ax.set_ylim(-0.03, 1.03)
ax.set_title("Rarity hollows out the knowledge base;\ndemand-weighted probes barely notice", fontsize=10, loc="left")
h1 = [Line2D([], [], color=HUE[c], lw=2.2, label={"environment": "environment reveals rare/common questions",
                                                    "demand": "agents ask about rare/common questions",
                                                    "both": "both skewed, same ranking"}[c]) for c in ("environment", "demand", "both")]
h2 = [Line2D([], [], color=INK, ls="-", marker="o", ms=5, label="IDK weighted uniformly over questions"),
      Line2D([], [], color=INK, ls="--", marker="s", ms=5, markerfacecolor="white", label="IDK weighted by demand (what gets asked)")]
l1 = ax.legend(handles=h1, loc="upper left"); ax.add_artist(l1); ax.legend(handles=h2, loc="center left", bbox_to_anchor=(0, 0.55))

# (b) per-question IDK vs rank at s = 1: the knowledge frontier
ax = axes[1]
ranks = np.arange(1, B["Q"] + 1)
for cond, key in (("environment", "idk_by_env_rank"), ("demand", "idk_by_ask_rank"), ("both", "idk_by_ask_rank")):
    M = np.array([r[key] for r in d[(False, cond, 1.0)]])
    m, se = M.mean(0), M.std(0) / np.sqrt(12)
    k = np.ones(5) / 5   # 5-rank moving average for legibility
    sm = lambda v: np.convolve(np.pad(v, 2, mode="edge"), k, mode="valid")
    ax.plot(ranks, sm(m), color=HUE[cond], lw=1.8)
    ax.fill_between(ranks, sm(m - se), sm(m + se), color=HUE[cond], alpha=0.15, lw=0)
H = np.sum(1.0 / ranks)
r_star = B["K"] * B["tau_mem"] / H
ax.axvline(r_star, color=MUTED, ls="--", lw=1)
ax.text(r_star * 1.05, 0.55, f"mean-field frontier\n$K\\tau_{{mem}}\\,\\pi(q)=1$ at rank {r_star:.0f}\n(uniform asking)", color=MUTED, fontsize=7.5)
ax.set_xscale("log"); ax.set_xlabel("question rank (1 = most common); 5-rank moving average")
ax.set_ylabel("“I don’t know” fraction for that question\n(mean over agents)"); ax.set_ylim(-0.03, 1.03)
ax.set_title("The long tail dies first ($s$ = 1)\nx = environment rank (orange) or demand rank (green, violet)", fontsize=10, loc="left")
ax.legend(handles=h1, loc="upper left")

# (c) coupling between environment ranking and demand ranking (both skewed)
ax = axes[2]
for coup in ("same", "independent", "reversed"):
    ys = [np.mean([r["idk"] for r in d[(True, coup, s)]]) for s in S]
    se = [np.std([r["idk"] for r in d[(True, coup, s)]]) / np.sqrt(12) for s in S]
    ax.errorbar(S, ys, yerr=se, color=COUP[coup], lw=1.8, ms=5.5, marker="o", elinewidth=1, capsize=0,
                label={"same": "rare in environment = rare in demand", "independent": "unrelated rankings",
                       "reversed": "rare in environment = popular in demand"}[coup])
ax.set_xlabel("Zipf exponent $s$ (both distributions skewed)")
ax.set_ylabel("“I don’t know” fraction (uniform over questions)"); ax.set_ylim(-0.03, 1.03)
ax.set_title("How the two rankings relate\n(complementary coverage helps at moderate skew)", fontsize=10, loc="left")
ax.legend(loc="upper left")

fig.text(0.5, 0.005, f"Baseline held constant: N={B['N']}, τ_mem={B['tau_mem']}, Q={B['Q']}, K={B['K']}, p_env={B['p_env']}, ε={B['eps']}, full mesh, unknown-first asking. "
         "Toy simulator; mean ± s.e. over 12 seeds, last 200 of 1000 steps.", ha="center", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig("fig_rarity.png", dpi=160); print("saved")
