"""The attention war: staleness-aware agents starve the knowledge that was safest.

Left: with half the questions static, static-question IDK diverges from dynamic-
question IDK as the changing half speeds up (unknown-first reference: no split).
Right: under dispersed speeds, per-question IDK vs volatility rank -- the death
frontier is inverted relative to the caution-era models: the SLOW tail dies.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toy_v4

LN2 = float(np.log(2.0))
NEG = toy_v4.NEG
INK, MUTED = "#0b0b0b", "#8a8984"
ORANGE, DARK = "#eb6834", "#8c3214"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "axes.titlesize": 11,
                     "legend.fontsize": 7.5, "legend.frameon": False, "legend.handlelength": 2.2})

BASE = dict(N=20, M=50, B=5, alpha=0.01, c_dec=LN2 / 20, chi=0.5, k_ctx=3, T=800)
SEEDS = range(8)


def final_knows(r):
    """Recompute per-(agent, question) answerability at the final step from _state."""
    st = r["_state"]; t = st["t"]; N = st["t_recv"].shape[0]
    held = st["t_recv"] > NEG
    cutoff = np.clip(np.floor(st["t_recv"] + st["delta"]).astype(np.int64), 0, t)
    n_after = st["C"][:, t][:, None] - st["C"][np.arange(N)[:, None], cutoff]
    return held & ((st["t_recv"] + st["delta"] >= t) | (n_after < st["k_ctx"])), st["lam"]


def left_panel(ax):
    lams = [0.003, 0.01, 0.03, 0.1, 0.3]
    for sig, ls, wide in (("staleness_aware", "-", True), ("unknown_first", "--", False)):
        st_m, dy_m = [], []
        for lam in lams:
            st_v, dy_v = [], []
            for s in SEEDS:
                r = toy_v4.run(seed=s, lam_mean=lam, frac_static=0.5, sigma=sig, **BASE)
                knows, lamq = final_knows(r)
                is_st = lamq == 0
                st_v.append(1 - knows[:, is_st].mean()); dy_v.append(1 - knows[:, ~is_st].mean())
            st_m.append(np.mean(st_v)); dy_m.append(np.mean(dy_v))
        lw = 1.9 if wide else 1.2
        col_s, col_d = (DARK, ORANGE) if wide else (MUTED, MUTED)
        lab = "staleness-aware" if wide else "unknown-first (reference)"
        ax.plot(lams, st_m, color=col_s, ls=ls, lw=lw, marker="o", ms=5,
                label=f"static questions — {lab}")
        ax.plot(lams, dy_m, color=col_d, ls=ls, lw=lw, marker="s", ms=4.5,
                markerfacecolor="white" if not wide else None,
                label=f"dynamic questions — {lab}")
    ax.set_xscale("log"); ax.set_xticks(lams); ax.set_xticklabels([f"{v:g}" for v in lams])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(-0.02, 0.55)
    ax.set_xlabel("Speed of the dynamic half, $\\bar\\lambda$")
    ax.set_ylabel("P(“I don’t know”)")
    ax.set_title("Half the questions never change — and they die first", loc="left")
    ax.legend(loc="upper left")


def right_panel(ax):
    n_bins = 10
    for sig, ls, col, lab in (("staleness_aware", "-", DARK, "staleness-aware"),
                              ("unknown_first", "--", MUTED, "unknown-first (reference)")):
        binned = np.zeros((len(SEEDS), n_bins))
        for si, s in enumerate(SEEDS):
            r = toy_v4.run(seed=s, lam_mean=0.03, lam_disp=2.0, sigma=sig, **BASE)
            knows, lamq = final_knows(r)
            idk_q = 1 - knows.mean(axis=0)
            order = np.argsort(lamq)                    # slow -> fast
            binned[si] = [idk_q[order[b::n_bins]].mean() if False else
                          idk_q[order].reshape(n_bins, -1).mean(axis=1)[b] for b in range(n_bins)]
        x = (np.arange(n_bins) + 0.5) / n_bins * 100
        m, se = binned.mean(axis=0), binned.std(axis=0) / np.sqrt(len(SEEDS))
        ax.errorbar(x, m, yerr=se, color=col, ls=ls, lw=1.9 if col == DARK else 1.2,
                    marker="o", ms=5, capsize=0, label=lab)
    ax.set_ylim(-0.02, 0.55)
    ax.set_xlabel("Question volatility rank (percentile; slow → fast)")
    ax.set_title("Dispersed speeds: the slow tail starves", loc="left")
    ax.legend(loc="upper right")


fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
left_panel(axes[0]); right_panel(axes[1])
fig.suptitle("The attention war: agents that re-verify at the rate the world changes starve their most stable knowledge",
             fontsize=11, x=0.01, ha="left", y=0.99)
fig.text(0.01, 0.005,
         "Toy v4 (crowding), baseline otherwise. Left: $M$=50, half static ($\\lambda_q$=0); static entries are never re-asked once known,\n"
         "so dynamic-half churn crowds them out. Right: $\\bar\\lambda$=0.03, speeds log-normal ($\\sigma_\\lambda$=2), per-question IDK by volatility decile.\n"
         "Under unknown-first asking neither split appears. Mean ± s.e. over 8 seeds, final step.",
         ha="left", va="bottom", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.12, 1, 0.93))
fig.savefig("fig_v4_attention_war.png", dpi=170)
print("saved fig_v4_attention_war.png")
