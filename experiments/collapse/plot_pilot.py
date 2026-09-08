import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "magenta": "#e87ba4", "violet": "#4a3aa7", "gray": "#8a8984"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "lines.linewidth": 2})
R = json.load(open("pilot_results.json"))


def group(name, keys):
    d = defaultdict(list)
    for c in R[name]:
        d[tuple(c["kw"].get(k) for k in keys)].append(c)
    return d


def t_sys_collapse(c, thresh=0.05):
    a = np.array(c["alive_series"])
    hit = np.flatnonzero(a <= thresh)
    return 10 * (hit[0] + 1) if len(hit) else np.inf


# ---------- Figure A: R0 threshold, N dependence, time to system collapse ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
g = group("r0", ["N", "tau_mem"])
cols = {5: C["blue"], 20: C["orange"], 100: C["aqua"]}
for N in (5, 20, 100):
    taus = sorted({k[1] for k in g if k[0] == N})
    r0 = [5 * t / 100 for t in taus]
    alive = [np.mean([c["alive_q"] for c in g[(N, t)]]) for t in taus]
    ttc = [np.median([t_sys_collapse(c) for c in g[(N, t)]]) for t in taus]
    axes[0].plot(r0, alive, "-o", ms=5, color=cols[N], label=f"N={N}")
    axes[1].plot(r0, ttc, "-o", ms=5, color=cols[N], label=f"N={N}")
axes[0].axvline(1, color=C["gray"], ls="--", lw=1)
axes[0].text(1.05, 0.5, "predicted\nthreshold", color=C["gray"], fontsize=8)
axes[0].set_xscale("log"); axes[0].set_xlabel(r"$R_0 = K\,\tau_{mem}/Q$")
axes[0].set_ylabel("fraction of questions still known\nby anyone (t=600..800)")
axes[0].set_title("System-level collapse is a threshold", fontsize=10); axes[0].legend(frameon=False)
axes[1].axvline(1, color=C["gray"], ls="--", lw=1)
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlabel(r"$R_0 = K\,\tau_{mem}/Q$")
axes[1].set_ylabel("median time to system collapse\n(<5% of questions alive; inf = none by t=800)")
axes[1].set_title("Time-to-collapse diverges near the threshold", fontsize=10)
axes[1].set_ylim(5, 1000); axes[1].legend(frameon=False)
for ax in axes[:2]:
    ax.set_xticks([0.25,0.5,1,2,5]); ax.set_xticklabels(["0.25","0.5","1","2","5"]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
# trajectories at N=20 for a few tau
g20 = {k: v for k, v in g.items() if k[0] == 20}
for tau, col in zip((15, 20, 25, 30, 40), (C["violet"], C["blue"], C["orange"], C["aqua"], C["yellow"])):
    ser = np.array([c["alive_series"] for c in g20[(20, tau)]])
    t = np.arange(ser.shape[1]) * 10
    axes[2].plot(t, ser.mean(0), color=col, label=fr"$\tau$={tau} ($R_0$={5*tau/100:.2f})")
axes[2].set_xlabel("iteration"); axes[2].set_ylabel("fraction of questions alive (N=20)")
axes[2].set_title("Collapse trajectories (mean of 12 seeds)", fontsize=10); axes[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("pilot_fig_A_threshold.png", dpi=150)

# ---------- Figure B: environment channel, strategy, topology ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
g = group("eps", ["tau_mem", "eps"])
for tau, col in ((10, C["orange"]), (40, C["blue"])):
    eps = sorted({k[1] for k in g if k[0] == tau})
    idk = [np.mean([c["final_idk"] for c in g[(tau, e)]]) for e in eps]
    alive = [np.mean([c["alive_q"] for c in g[(tau, e)]]) for e in eps]
    axes[0].plot([max(e, 0.003) for e in eps], idk, "-o", ms=5, color=col, label=fr"agent-level IDK, $\tau$={tau} ($R_0$={tau/20:.1f})")
    axes[0].plot([max(e, 0.003) for e in eps], alive, "--s", ms=5, color=col, alpha=0.6, label=fr"system: questions alive, $\tau$={tau}")
axes[0].set_xscale("log"); axes[0].set_xlabel(r"env. observation rate $\epsilon$ per agent-step ($\epsilon$=0 drawn at 0.003)")
axes[0].set_ylabel("fraction"); axes[0].set_title("Environment channel: system recovers before agents do", fontsize=10)
axes[0].legend(frameon=False, fontsize=7)

g = group("strategy", ["strategy", "tau_mem"])
for st, col in (("uniform", C["gray"]), ("unknown_first", C["blue"]), ("oldest", C["orange"])):
    taus = sorted({k[1] for k in g if k[0] == st})
    alive = [np.mean([c["alive_q"] for c in g[(st, t)]]) for t in taus]
    axes[1].plot([t / 20 for t in taus], alive, "-o", ms=5, color=col, label=st)
axes[1].set_xlabel(r"$R_0 = K\,\tau_{mem}/Q$"); axes[1].set_ylabel("fraction of questions alive (N=20)")
axes[1].set_title("Communication strategy shifts the threshold modestly", fontsize=10); axes[1].legend(frameon=False)

g = group("degree", ["tau_mem", "mean_degree"])
for tau, col in ((30, C["orange"]), (40, C["blue"])):
    degs = sorted({k[1] for k in g if k[0] == tau})
    alive = [np.mean([c["alive_q"] for c in g[(tau, d)]]) for d in degs]
    axes[2].plot(degs, alive, "-o", ms=5, color=col, label=fr"$\tau$={tau} ($R_0$={tau/20:.1f})")
axes[2].set_xscale("log"); axes[2].set_xlabel("mean degree (N=100, ER graph; 99 = full mesh)")
axes[2].set_ylabel("fraction of questions alive"); axes[2].set_title("Sparse contact graphs collapse first", fontsize=10)
axes[2].legend(frameon=False)
fig.tight_layout(); fig.savefig("pilot_fig_B_factors.png", dpi=150)

# ---------- Figure C: environment speed x memory: IDK / stale / correct ----------
g = group("penv", ["tau_mem", "p_env"])
taus = sorted({k[0] for k in g}); ps = sorted({k[1] for k in g})
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for ax, key, title, cmap in zip(axes, ("final_idk", "final_stale", "final_correct"),
                                 ("'I don't know' (collapse)", "confidently stale (wrong)", "correct"),
                                 ("Blues", "Oranges", "Greens")):
    M = np.array([[np.mean([c[key] for c in g[(t, p)]]) for p in ps] for t in taus])
    im = ax.imshow(M, origin="lower", vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(ps))); ax.set_xticklabels(ps); ax.set_yticks(range(len(taus))); ax.set_yticklabels(taus)
    ax.set_xlabel(r"environment change prob. $p_{env}$"); ax.set_ylabel(r"memory lifetime $\tau_{mem}$")
    ax.set_title(title, fontsize=10); ax.grid(False)
    for i in range(len(taus)):
        for j in range(len(ps)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] > 0.6 else "#0b0b0b")
fig.suptitle(r"All questions temporal, $\epsilon$=0.1, K=5, Q=100, N=20: forgetting converts 'wrong' into 'I don't know'; only environment sampling creates 'correct'", fontsize=9)
fig.tight_layout(); fig.savefig("pilot_fig_C_env_speed.png", dpi=150)
print("done")
