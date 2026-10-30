"""Figure 1: epistemic failure and epistemic capacity, in one example system
(no mitigation; c = 0.05, lambda = 0.03, E = 3, K = 8, N = 10, k = 3 -- the
Figure 2 default).

(a) example dynamics: failure over time from a warm start, M = 400 (mirror, 200 seeds;
    mean +- 1 SEM), split into collapse and misalignment.
(b) steady-state failure vs environment size (theory lines; mirror markers).
    Dotted lines at phi = 0.1 and 0.5; where failure crosses them is the
    epistemic capacity M*(phi).
(c) epistemic capacity as a function of the failure tolerance phi (theory line;
    mirror markers = crossings of the mirror curve in (b)).
Open diamonds in all panels: the real system (gpt-4o-mini agents; 60 seeds in (a), 20 in (b)-(c),
T = 100, steady state from probes at t >= 50; runs f1real-* from
data/configs/real_fig1/, cached in data/figure1_real.json).

Channels: Set1 failure colors (all / collapse / misalignment); gray = the
method (no mitigation); solid = firsthand only (the paper's linestyle
attribute); line = theory, marker = simulation; opacity = phi in (b)-(c).

Compute stages cache into data/figure1_trajectories.json, figure1_mirror.json,
figure1_theory_size.json, figure1_theory_capacity.json (~3 min in all);
rendering reads only the caches.
"""
import epistemic_collapse_common as ec

import json
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import hv_style
import epistemic_collapse_mirror as mirror
from epistemic_collapse_theory import misalignment

hv_style.apply()
SZ = ec.SIZES

C, LAM, M_A = 0.05, 0.03, 400
MS = [10, 16, 25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000, 20000]
M_TH = np.unique(np.geomspace(6, 20000, 65).astype(int))
PHI_TH = np.linspace(0.02, 0.9, 45)
PHI_MIR = np.round(np.arange(0.05, 0.851, 0.05), 2)
PHIS = ((0.1, 0.45), (0.5, 1.0))          # highlighted tolerances (value, opacity)
STEADY_SEEDS = range(300, 320)     # 20 seeds, as the real runs
TRAJ_SEEDS = range(300, 500)       # (a) proxy line: 200 seeds, so its noise is negligible next to the real points

F_ALL, F_COL, F_MIS = ec.FAILURE['all failures'], ec.FAILURE['collapse'], ec.FAILURE['misalignment']
ST = ec.style('no mitigation')
REAL_MK = dict(marker='D', ms=9, ls='', mfc='white', mew=2.2, elinewidth=1.8, capsize=0, zorder=7)
LS = ST['ls']

TRAJ = f'{ec.DATA}/figure1_trajectories.json'
STEADY = f'{ec.DATA}/figure1_mirror.json'
TH_SIZE = f'{ec.DATA}/figure1_theory_size.json'
TH_CAP = f'{ec.DATA}/figure1_theory_capacity.json'
REAL = f'{ec.DATA}/figure1_real.json'
REAL_MS = [40, 100, 250, 630, 1600, 4000, 10000]
REAL_SEEDS = range(800, 820)                   # (b)-(c): 20 seeds per size
REAL_DYN_SEEDS = range(800, 860)               # (a): 60 seeds (850-859 also logged the in-vivo gate)
REAL_T0 = 50                                   # steady state = probes at t >= 50 (runs are T = 100)


# ---------------------------------------------------------------- compute
def _traj(s):
    r = mirror.run(M=M_A, c=C, lam=LAM, seed=s, T=60, warm_start=True, interval=1,
                   probe=100, select_rule='uniform')
    return r['t'].tolist(), r['collapse'].tolist(), r['misalign'].tolist()


def _steady(job):
    M, s = job
    r = mirror.run(M=M, c=C, lam=LAM, seed=s, T=300, select_rule='uniform', probe=min(M, 100))
    return job, float(np.mean(r['collapse'][-10:])), float(np.mean(r['misalign'][-10:]))


def _size(M):
    x = misalignment(int(M), LAM, c=C, asks='uniform')
    return str(int(M)), [x['collapse'], x['misalign']]


def _mstar(phi):
    from scipy.optimize import brentq
    f = lambda lm: (lambda x: x['collapse'] + x['misalign'])(
        misalignment(int(round(np.exp(lm))), LAM, c=C, asks='uniform')) - phi
    if f(np.log(2)) >= 0:          # below the refusal floor: no environment is small enough
        return f'{phi:.4f}', float('nan')
    return f'{phi:.4f}', float(np.exp(brentq(f, np.log(2), np.log(50000), xtol=0.005)))


def compute_real():
    """collect the finished real-system (gpt-4o-mini) runs into one cache."""
    import epistemic_collapse_real as R
    out = {'dyn': {}, 'steady': {}}
    for sd in REAL_DYN_SEEDS:
        t, col, mis = R.failure_series(R.run_dir(f'f1real-dyn-M{M_A}-s{sd}'))
        out['dyn']['t'] = t.tolist()
        out['dyn'].setdefault('runs', []).append([col.tolist(), mis.tolist()])
    for M in REAL_MS:
        for sd in REAL_SEEDS:
            t, col, mis = R.failure_series(R.run_dir(f'f1real-ss-M{M}-s{sd}'))
            k = t >= REAL_T0
            out['steady'].setdefault(str(M), []).append([float(col[k].mean()), float(mis[k].mean())])
    json.dump(out, open(REAL, 'w'), indent=1)


def compute():
    if not os.path.exists(REAL):
        compute_real()
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        if not os.path.exists(TRAJ):
            runs = list(ex.map(_traj, TRAJ_SEEDS))
            t = runs[0][0]
            col, mis = np.array([r[1] for r in runs]), np.array([r[2] for r in runs])
            n = np.sqrt(len(runs))
            json.dump({'t': t, 'col': col.mean(0).tolist(), 'mis': mis.mean(0).tolist(),
                       'tot_sem': ((col + mis).std(0) / n).tolist(),
                       'col_sem': (col.std(0) / n).tolist(), 'mis_sem': (mis.std(0) / n).tolist()},
                      open(TRAJ, 'w'), indent=1)
        if not os.path.exists(STEADY):
            st = {}
            jobs = sorted([(M, s) for M in MS for s in STEADY_SEEDS], key=lambda j: -j[0])
            for (M, s), col, mis in ex.map(_steady, jobs, chunksize=1):
                st.setdefault(str(M), []).append([col, mis])
            json.dump(st, open(STEADY, 'w'), indent=1)
        if not os.path.exists(TH_SIZE):
            json.dump(dict(ex.map(_size, M_TH)), open(TH_SIZE, 'w'), indent=1)
        if not os.path.exists(TH_CAP):
            json.dump(dict(ex.map(_mstar, [float(p) for p in PHI_TH] + [p for p, _ in PHIS])),
                      open(TH_CAP, 'w'), indent=1)


# ---------------------------------------------------------------- render
def crossing(y, phi, grid):
    """log-interpolated environment size where an increasing failure curve first reaches phi."""
    y = np.asarray(y)
    k = np.flatnonzero((y[:-1] < phi) & (y[1:] >= phi))
    if y[0] >= phi or not len(k):
        return None
    k = k[0]
    return float(np.exp(np.interp(phi, [y[k], y[k + 1]], np.log([grid[k], grid[k + 1]]))))


def render():
    traj = json.load(open(TRAJ))
    steady = {int(M): np.array(v) for M, v in json.load(open(STEADY)).items()}
    size = json.load(open(TH_SIZE))
    cap = json.load(open(TH_CAP))
    mir_tot = np.array([steady[M].sum(1).mean() for M in MS])
    real = json.load(open(REAL))
    rdyn = np.array(real['dyn']['runs'])                       # seeds x {col, mis} x t
    rt = np.array(real['dyn']['t'])
    rst = {int(M): np.array(v) for M, v in real['steady'].items()}
    real_tot = np.array([rst[M].sum(1).mean() for M in REAL_MS])
    real_sem = np.array([rst[M].sum(1).std(ddof=1) / np.sqrt(len(rst[M])) for M in REAL_MS])

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(21, 6.6))

    # (a) example dynamics
    t = np.array(traj['t']); col = np.array(traj['col']); mis = np.array(traj['mis'])
    for y, sem, colr, lw in ((col + mis, traj['tot_sem'], F_ALL, 5.0), (col, traj['col_sem'], F_COL, 4.2),
                             (mis, traj['mis_sem'], F_MIS, 4.2)):
        axA.plot(t, y, color=colr, lw=lw, ls=LS)
        axA.fill_between(t, y - np.array(sem), y + np.array(sem), color=colr, alpha=0.13, lw=0)
    nr = np.sqrt(len(rdyn))
    for y, colr in ((rdyn.sum(1), F_ALL), (rdyn[:, 0], F_COL), (rdyn[:, 1], F_MIS)):
        axA.errorbar(rt, y.mean(0), yerr=y.std(0, ddof=1) / nr, color=colr, **REAL_MK)
    axA.set_xlim(0, 57); axA.set_ylim(-0.03, 1.03)
    axA.set_xlabel('step ($t$)', fontsize=SZ['label'])
    axA.set_ylabel('failure proportion', fontsize=SZ['label'])
    axA.set_title(f'(a) example dynamics ($M$ = {M_A})', loc='left', fontsize=SZ['subtitle'])

    # (b) failure vs environment size; capacity = where failure crosses phi
    th = np.array([size[str(int(M))] for M in M_TH])
    axB.plot(M_TH, th.sum(1), color=F_ALL, lw=5.0, ls=LS)
    axB.plot(M_TH, th[:, 0], color=F_COL, lw=4.2, ls=LS)
    axB.plot(M_TH, th[:, 1], color=F_MIS, lw=4.2, ls=LS)
    axB.plot(MS, mir_tot, marker='o', ms=9, ls='', color=F_ALL, zorder=5)
    axB.errorbar(REAL_MS, real_tot, yerr=real_sem, color=F_ALL, **REAL_MK)
    for phi, al in PHIS:
        m = cap[f'{phi:.4f}']
        axB.axhline(phi, color=hv_style.REFLINE, ls=':', lw=2.0, zorder=0)
        axB.plot([m, m], [-0.03, phi], color=ST['color'], ls=':', lw=2.2, alpha=al, zorder=1)
        axB.plot([m], [phi], marker='o', ms=13, ls='', color=ST['color'], alpha=al, zorder=6)
    axB.set_xscale('log'); axB.set_ylim(-0.03, 1.03); axB.set_xlim(M_TH[0], M_TH[-1])
    axB.tick_params(labelleft=False)                     # (a) and (b) share the failure axis
    axB.set_xlabel('environment size ($M$)', fontsize=SZ['label'])
    axB.set_title('(b) failure vs. size', loc='left', fontsize=SZ['subtitle'])

    # (c) epistemic capacity as a function of the failure tolerance
    axC.plot(PHI_TH, [cap[f'{p:.4f}'] for p in PHI_TH], color=ST['color'], ls=LS, lw=ST['lw'] + 1.8)
    pts = [(p, crossing(mir_tot, p, MS)) for p in PHI_MIR]
    pts = [(p, m) for p, m in pts if m]
    axC.plot([p for p, _ in pts], [m for _, m in pts], marker='o', ms=9, ls='', color=ST['color'], zorder=5)
    rpts = [(p, crossing(real_tot, p, REAL_MS)) for p in PHI_MIR]
    rpts = [(p, m) for p, m in rpts if m]
    axC.errorbar([p for p, _ in rpts], [m for _, m in rpts], color=ST['color'], **REAL_MK)
    for phi, al in PHIS:
        axC.plot([phi], [cap[f'{phi:.4f}']], marker='o', ms=13, ls='', color=ST['color'], alpha=al,
                 markeredgecolor='white', markeredgewidth=1.5, zorder=6)
    axC.set_yscale('log')
    axC.set_yticks([30, 100, 300, 1000, 3000, 10000, 30000]); axC.set_yticklabels(['30', '100', '300', '1k', '3k', '10k', '30k'])
    axC.set_ylim(20, 40000)
    axC.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axC.set_xlim(0, 0.92)
    axC.set_xlabel('failure tolerance ($\\varphi$)', fontsize=SZ['label'])
    axC.set_ylabel('epistemic capacity $M^\\star(\\varphi)$', fontsize=SZ['label'])
    axC.set_title('(c) epistemic capacity', loc='left', fontsize=SZ['subtitle'])

    for ax in (axA, axB, axC):
        ax.tick_params(labelsize=SZ['tick'])
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.33, top=0.92, wspace=0.18)
    p = axC.get_position(); axC.set_position([p.x0 + 0.045, p.y0, p.width - 0.045, p.height])

    # orthogonal factors -> separate legends below the panels
    leg = fig.legend(handles=[Line2D([], [], color=F_ALL, lw=5.0, label='all'),
                              Line2D([], [], color=F_COL, lw=4.4, label='collapse'),
                              Line2D([], [], color=F_MIS, lw=4.4, label='misalignment')],
                     loc='lower center', ncol=3, bbox_to_anchor=(0.31, 0.0), fontsize=SZ['legend'],
                     handlelength=2.2, columnspacing=1.4, title='failure type', title_fontsize=SZ['legend'])
    fig.add_artist(leg)
    fig.legend(handles=[Line2D([], [], color=hv_style.INK, ls=LS, lw=4.4, label='theory'),
                        Line2D([], [], color=hv_style.INK, ls='', marker='o', ms=9, label='proxy'),
                        Line2D([], [], color=hv_style.INK, ls='', marker='D', ms=9, mfc='white',
                               mew=2.2, label='real (gpt-4o-mini)')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.74, 0.0), fontsize=SZ['legend'],
               handlelength=2.6, columnspacing=1.6, title='source', title_fontsize=SZ['legend'])
    ec.save_artifact(fig, 'epistemic_collapse_figure1_capacity')


if __name__ == '__main__':
    compute()
    render()
