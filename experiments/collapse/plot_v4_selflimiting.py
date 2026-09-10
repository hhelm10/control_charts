"""Crowding makes collapse self-limiting: as knowledge dies, asks stop returning
answers, insertions stop, and old memories resurface (tau_eff = Delta + k/r_ins
grows). Hard expiry (v3) has no such feedback. Matched sub-threshold cells."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toy_v3, toy_v4

LN2 = float(np.log(2.0))
INK, MUTED = "#0b0b0b", "#8a8984"
BLUE = "#2a78d6"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False})

SEEDS = range(8)
COMMON = dict(N=20, M=50, B=5, alpha=0.01, lam_mean=0.003, T=800)
CELLS = [  # (label, effective lifetime, v3 kwargs, v4 kwargs) -- matched Delta/tau
    ("lifetime 10 ($R_0\\approx1$)", 10, dict(c_dec=LN2 / 10, theta_ret=0.5), dict(c_dec=LN2 / 20, chi=np.exp(-10 * LN2 / 20), k_ctx=3)),
    ("lifetime 6.4 ($R_0\\approx0.6$)", 6.4, dict(c_dec=LN2 / 6.4, theta_ret=0.5), dict(c_dec=LN2 / 20, chi=0.8, k_ctx=3)),
    ("lifetime 4 ($R_0\\approx0.4$)", 4, dict(c_dec=LN2 / 4, theta_ret=0.5), dict(c_dec=LN2 / 20, chi=np.exp(-4 * LN2 / 20), k_ctx=3)),
]


def traj(mod, kw):
    return np.mean([mod.run(seed=s, **COMMON, **kw)["dead_q"] for s in SEEDS], axis=0)


fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7), sharey=True)
for ax, (lbl, tau, kw3, kw4) in zip(axes, CELLS):
    t = np.arange(1, 161) * 5
    ax.plot(t, traj(toy_v3, kw3), color=MUTED, ls="--", lw=1.7, label="hard expiry (v3)")
    ax.plot(t, traj(toy_v4, kw4), color=BLUE, lw=1.7, label="crowding (v4)")
    ax.set_title(lbl, loc="left")
    ax.set_xlabel("Step"); ax.set_ylim(-0.03, 1.03)
axes[0].set_ylabel("P(no agent can answer)")
axes[0].legend(loc="center right")
fig.suptitle("Crowding makes collapse self-limiting: dying agents stop inserting, so their old memories resurface",
             fontsize=11, x=0.01, ha="left", y=0.99)
fig.text(0.01, 0.005,
         "Matched effective memory lifetimes (v3: hard cutoff $\\tau$; v4: head start $\\Delta=\\ln(1/\\chi)/c$, context $k$=3).\n"
         "Baseline otherwise ($N$=20, $M$=50, $B$=5, $\\alpha$=0.01, $\\bar\\lambda$=0.003, mesh). "
         "Under crowding, $\\tau_{eff}=\\Delta+k/r_{ins}$ grows as the insertion rate falls. Mean of 8 seeds.",
         ha="left", va="bottom", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.09, 1, 0.93))
fig.savefig("fig_v4_selflimiting.png", dpi=170)
print("saved fig_v4_selflimiting.png")
