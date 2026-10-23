"""Figure 1 (v4, Option A; uniform question selection): epistemic failure = collapse + misalignment.

(a) failure over time: warm start, M = 200, lam = 0.03; stacked shares of the
    environment that is known correctly by someone / misaligned / collapsed,
    slow (top) vs fast (bottom) forgetting. Mirror simulation.
(b) failure proportion phi vs environment size M at lam = 0.03: total (solid),
    collapse (dashed), misalignment (dotted); theory lines, mirror markers.
(c) epistemic capacity M*(phi; lam): contours phi = 0.1, 0.25, 0.5 in the
    (environment speed, environment size) plane; theory lines; mirror-derived
    capacity points (interpolated crossings) at three speeds.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_mirror
from analytic_collapse import misalignment

INK, MUTED = "#0b0b0b", "#8a8984"
SYS = {0.05: ("slow forgetting ($c$=0.05)", "#2a78d6"), 0.15: ("fast forgetting ($c$=0.15)", "#c94f3d")}
KNOWN, MIS, COL = "#d9d9d6", "#e8a33d", "#5b5b5b"
LAM_B, M_A = 0.03, 400
MS = [25, 50, 100, 200, 400]
LAM_C = [0.003, 0.03, 0.3]
PHIS = {0.1: ":", 0.25: "--", 0.5: "-"}
LAMS_C = np.geomspace(1e-3, 0.3, 28)
CACHE = "projects/data/fig1_v4u_mirror.json"
TRAJ_CACHE = "projects/data/fig1_v4u_traj_M400.json"
TRAJ_SEEDS = range(300, 320)
SEEDS = range(300, 316)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25,
                     "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 10.5, "legend.frameon": False})


def _traj(job):
    c, s = job
    r = sim_mirror.run(M=M_A, c=c, lam=LAM_B, seed=s, T=150, warm_start=True, interval=3, probe=100,
                       select_rule="uniform")
    return job, r["t"].tolist(), r["collapse"].tolist(), r["misalign"].tolist()


def _steady(job):
    c, M, lam, s = job
    r = sim_mirror.run(M=M, c=c, lam=lam, seed=s, T=300, select_rule="uniform", probe=min(M, 100))
    return job, float(np.mean(r["collapse"][-10:])), float(np.mean(r["misalign"][-10:]))


def mirror_data():
    if os.path.exists(CACHE):
        return json.load(open(CACHE))
    with ProcessPoolExecutor() as ex:
        tr = list(ex.map(_traj, [(c, s) for c in SYS for s in SEEDS]))
        st = list(ex.map(_steady, [(c, M, lam, s) for c in SYS for M in MS
                                   for lam in sorted(set(LAM_C + [LAM_B])) for s in SEEDS]))
    traj, steady = {}, {}
    for (c, s), t, col, mis in tr:
        traj.setdefault(str(c), {"t": t, "col": [], "mis": []})
        traj[str(c)]["col"].append(col); traj[str(c)]["mis"].append(mis)
    for (c, M, lam, s), col, mis in st:
        steady.setdefault(f"{c}-{M}-{lam}", []).append((col, mis))
    out = {"traj": {k: {"t": v["t"], "col": np.mean(v["col"], 0).tolist(),
                        "mis": np.mean(v["mis"], 0).tolist()} for k, v in traj.items()},
           "steady": {k: np.mean(v, 0).tolist() for k, v in steady.items()}}
    json.dump(out, open(CACHE, "w"), indent=1)
    return out


def _mstar(job):
    """capacity M*(phi; lam): root of total failure(M) = phi, solved on log M."""
    from scipy.optimize import brentq
    c, lam, phi = job
    f = lambda lm: (lambda x: x["collapse"] + x["misalign"])(misalignment(int(round(np.exp(lm))), lam, c=c, asks="uniform")) - phi
    lo, hi = np.log(5), np.log(20000)
    try:
        return job, float(np.exp(brentq(f, lo, hi, xtol=0.01)))
    except ValueError:
        return job, float("nan")


def capacity_curves():
    path = "projects/data/fig1_v4u_capacity.json"
    if os.path.exists(path):
        d = json.load(open(path))
        return {tuple(json.loads(k)): v for k, v in d.items()}
    jobs = [(c, float(l), phi) for c in SYS for l in LAMS_C for phi in (0.1, 0.25, 0.5)]
    with ProcessPoolExecutor() as ex:
        res = dict(ex.map(_mstar, jobs))
    json.dump({json.dumps(list(k)): v for k, v in res.items()}, open(path, "w"))
    return res


def traj_data():
    if os.path.exists(TRAJ_CACHE):
        return json.load(open(TRAJ_CACHE))
    with ProcessPoolExecutor() as ex:
        tr = list(ex.map(_traj, [(c, s) for c in SYS for s in TRAJ_SEEDS]))
    out = {}
    for (c, s), t, col, mis in tr:
        out.setdefault(str(c), {"t": t, "col": [], "mis": []})
        out[str(c)]["col"].append(col); out[str(c)]["mis"].append(mis)
    out = {k: {"t": v["t"], "col": np.mean(v["col"], 0).tolist(), "mis": np.mean(v["mis"], 0).tolist()}
           for k, v in out.items()}
    json.dump(out, open(TRAJ_CACHE, "w"))
    return out


def main():
    md = mirror_data()
    td = traj_data()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(14.2, 4.6))

    # encoding: color = failure mode, line style = memory system
    TOT, MISC, COLC = "#7b4fc9", "#2a78d6", "#c94f3d"
    LS = {0.15: "-", 0.05: "--"}                       # fast forgetting solid, slow dashed
    MFC = {0.15: None, 0.05: "white"}                  # simulation markers: filled fast, hollow slow

    # (a) failure over time (warm start, M = M_A, lam = LAM_B)
    for c in SYS:
        d = td[str(c)]
        t = np.array(d["t"]); col = np.array(d["col"]); mis = np.array(d["mis"])
        axA.plot(t, col + mis, color=TOT, lw=3.4, ls=LS[c])
        axA.plot(t, col, color=COLC, lw=2.6, ls=LS[c])
        axA.plot(t, mis, color=MISC, lw=2.6, ls=LS[c])
    axA.set_ylim(-0.03, 1.03)
    axA.set_xlabel("step ($t$)"); axA.set_ylabel("failure proportion $\\varphi$")
    axA.set_title(f"failure over time ($M$ = {M_A}, $\\bar\\lambda$ = {LAM_B})", loc="left")

    # (b) failure vs environment size at LAM_B
    Mth = np.unique(np.geomspace(15, 1500, 40).astype(int))
    for c in SYS:
        th = [misalignment(M, LAM_B, c=c, asks="uniform") for M in Mth]
        axB.plot(Mth, [x["collapse"] + x["misalign"] for x in th], color=TOT, lw=3.4, ls=LS[c])
        axB.plot(Mth, [x["collapse"] for x in th], color=COLC, lw=2.6, ls=LS[c])
        axB.plot(Mth, [x["misalign"] for x in th], color=MISC, lw=2.6, ls=LS[c])
        for M in MS:
            col, mis = md["steady"][f"{c}-{M}-{LAM_B}"]
            axB.plot([M], [col + mis], marker="o", color=TOT, ms=6.5, ls="",
                     markerfacecolor=MFC[c] or TOT, markeredgecolor=TOT, markeredgewidth=1.4, zorder=5)
    axB.set_xscale("log"); axB.set_ylim(-0.03, 1.03)
    axB.set_xlabel("environment size ($M$)")
    axB.set_ylabel("failure proportion $\\varphi$")
    axB.set_title(f"failure vs. size ($\\bar\\lambda$ = {LAM_B}, $t \\geq 200$)", loc="left")

    # (c) capacity curves M*(phi; lam): line at phi = 0.25, band phi in [0.1, 0.5]
    cap = capacity_curves()
    for c in SYS:
        for phi, al in ((0.1, 0.4), (0.5, 1.0)):
            y = np.array([cap[(c, float(l), phi)] for l in LAMS_C])
            axC.plot(LAMS_C, y, color=TOT, lw=3.2, ls=LS[c], alpha=al)
        for phi, al in ((0.1, 0.4), (0.5, 1.0)):          # mirror-derived capacity
            for lam in LAM_C:
                ph = np.array([sum(md["steady"][f"{c}-{M}-{lam}"]) for M in MS])
                k_ = np.flatnonzero((ph[:-1] < phi) & (ph[1:] >= phi))
                if len(k_):
                    k_ = k_[0]
                    lm = np.interp(phi, [ph[k_], ph[k_ + 1]], np.log([MS[k_], MS[k_ + 1]]))
                    axC.plot([lam], [np.exp(lm)], marker="o", ms=7, ls="", alpha=al,
                             markerfacecolor=MFC[c] or TOT, markeredgecolor=TOT,
                             markeredgewidth=1.4, zorder=6)
    axC.set_xscale("log"); axC.set_yscale("log")
    axC.set_yticks([30, 100, 300, 1000]); axC.set_yticklabels(["30", "100", "300", "1000"])
    axC.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axC.set_xlabel("environment speed ($\\bar\\lambda$)")
    axC.set_ylabel("epistemic capacity $M^\\star(\\varphi)$")
    axC.set_title("epistemic capacity", loc="left")
    axC.legend(handles=[Line2D([], [], color=TOT, lw=3.2, alpha=al, label=f"$\\varphi$ = {phi}")
                        for phi, al in ((0.5, 1.0), (0.1, 0.4))],
               loc="upper right")
    fig.tight_layout(w_pad=2.0, rect=(0, 0.1, 1, 1))
    handles = [Line2D([], [], color=TOT, lw=3.4, label="all failures"),
               Line2D([], [], color=COLC, lw=3.0, label="collapse"),
               Line2D([], [], color=MISC, lw=3.0, label="misalignment"),
               Line2D([], [], color=INK, lw=3.0, ls="-", label=SYS[0.15][0]),
               Line2D([], [], color=INK, lw=3.0, ls="--", label=SYS[0.05][0]),
               Line2D([], [], color=TOT, marker="o", ls="", ms=8, label="simulation")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.4, columnspacing=1.6)
    fig.savefig("projects/artifacts/fig1_v4.png", dpi=170, bbox_inches="tight")
    print("saved fig1_v4.png")


if __name__ == "__main__":
    main()
