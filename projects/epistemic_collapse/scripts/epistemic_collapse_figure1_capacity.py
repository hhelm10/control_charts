"""Figure 1: epistemic failure (collapse + misalignment) and epistemic capacity.

(a) failure over time: warm start, M = 400, lambda = 0.03 (mirror; mean +- 1 SEM).
(b) steady-state failure vs environment size at lambda = 0.03 (theory lines;
    mirror markers).
(c) epistemic capacity M*(phi; lambda) vs environment speed at phi = 0.1, 0.5
    (theory lines; mirror-derived capacity markers).

Channels: violet ramp = the reserved concept, epistemic failure (all failures /
collapse / misalignment); gray = the method (no mitigation); figure-wide
solid / dashed pair = fast / slow forgetting (declared in the legend); marker
fill: filled = fast, open = slow. Opacity in (c) = failure tolerance phi.

Compute stages write caches into projects/epistemic_collapse/data/ the first time
(mirror simulations ~3 min, theory ~1 min); rendering reads only the caches.
"""
import epistemic_collapse_common as ec  # noqa: F401  (pins cwd, paths, BLAS caps)

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
SZ = hv_style.SIZES

C_FAST, C_SLOW = 0.15, 0.05
FORGET = {C_FAST: ('fast forgetting', '-', None), C_SLOW: ('slow forgetting', '--', 'white')}
LAM_B, M_A = 0.03, 400
MS = [25, 50, 100, 200, 400]
LAM_C = [0.003, 0.03, 0.3]
LAMS_C = np.geomspace(1e-3, 0.3, 28)
M_TH = np.unique(np.geomspace(15, 1500, 40).astype(int))
PHIS = ((0.1, 0.45), (0.5, 1.0))          # (tolerance, opacity)
STEADY_SEEDS, TRAJ_SEEDS = range(300, 316), range(300, 320)

F_ALL, F_COL, F_MIS = ec.FAILURE['all failures'], ec.FAILURE['collapse'], ec.FAILURE['misalignment']
METHOD = ec.style('no mitigation')['color']

STEADY = f'{ec.DATA}/figure1_mirror.json'
TRAJ = f'{ec.DATA}/figure1_trajectories.json'
TH_SIZE = f'{ec.DATA}/figure1_theory_size.json'
TH_CAP = f'{ec.DATA}/figure1_theory_capacity.json'


# ---------------------------------------------------------------- compute
def _traj(job):
    c, s = job
    r = mirror.run(M=M_A, c=c, lam=LAM_B, seed=s, T=150, warm_start=True, interval=3,
                   probe=100, select_rule='uniform')
    return job, r['t'].tolist(), r['collapse'].tolist(), r['misalign'].tolist()


def _steady(job):
    c, M, lam, s = job
    r = mirror.run(M=M, c=c, lam=lam, seed=s, T=300, select_rule='uniform', probe=min(M, 100))
    return job, float(np.mean(r['collapse'][-10:])), float(np.mean(r['misalign'][-10:]))


def _mstar(job):
    """capacity M*(phi; lambda): root of total failure(M) = phi, solved on log M."""
    from scipy.optimize import brentq
    c, lam, phi = job
    f = lambda lm: (lambda x: x['collapse'] + x['misalign'])(
        misalignment(int(round(np.exp(lm))), lam, c=c, asks='uniform')) - phi
    try:
        return json.dumps([c, lam, phi]), float(np.exp(brentq(f, np.log(5), np.log(20000), xtol=0.01)))
    except ValueError:
        return json.dumps([c, lam, phi]), float('nan')


def _size(job):
    c, M = job
    x = misalignment(int(M), LAM_B, c=c, asks='uniform')
    return f'{c}-{M}', [x['collapse'], x['misalign']]


def compute():
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        if not os.path.exists(TRAJ):
            out = {}
            for (c, s), t, col, mis in ex.map(_traj, [(c, s) for c in FORGET for s in TRAJ_SEEDS]):
                d = out.setdefault(str(c), {'t': t, 'col': [], 'mis': []})
                d['col'].append(col); d['mis'].append(mis)
            res = {}
            for c, d in out.items():
                col, mis = np.array(d['col']), np.array(d['mis'])
                n = np.sqrt(len(col))
                res[c] = {'t': d['t'], 'col': col.mean(0).tolist(), 'mis': mis.mean(0).tolist(),
                          'tot_sem': ((col + mis).std(0) / n).tolist(),
                          'col_sem': (col.std(0) / n).tolist(), 'mis_sem': (mis.std(0) / n).tolist()}
            json.dump(res, open(TRAJ, 'w'), indent=1)
        if not os.path.exists(STEADY):
            st = {}
            for (c, M, lam, s), col, mis in ex.map(_steady, [
                    (c, M, lam, s) for c in FORGET for M in MS
                    for lam in sorted(set(LAM_C + [LAM_B])) for s in STEADY_SEEDS]):
                st.setdefault(f'{c}-{M}-{lam}', []).append((col, mis))
            json.dump({'steady': {k: np.mean(v, 0).tolist() for k, v in st.items()}},
                      open(STEADY, 'w'), indent=1)
        if not os.path.exists(TH_SIZE):
            json.dump(dict(ex.map(_size, [(c, int(M)) for c in FORGET for M in M_TH])),
                      open(TH_SIZE, 'w'), indent=1)
        if not os.path.exists(TH_CAP):
            json.dump(dict(ex.map(_mstar, [(c, float(l), phi) for c in FORGET for l in LAMS_C
                                           for phi in (0.1, 0.25, 0.5)])), open(TH_CAP, 'w'))


# ---------------------------------------------------------------- render
def render():
    traj = json.load(open(TRAJ))
    steady = json.load(open(STEADY))['steady']
    size = json.load(open(TH_SIZE))
    cap = {tuple(json.loads(k)): v for k, v in json.load(open(TH_CAP)).items()}

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(21, 6.4))

    # (a) failure over time
    for c, (_, ls, _) in FORGET.items():
        d = traj[str(c)]
        t = np.array(d['t']); col = np.array(d['col']); mis = np.array(d['mis'])
        for y, sem, colr, lw in ((col + mis, d['tot_sem'], F_ALL, 3.6), (col, d['col_sem'], F_COL, 2.6),
                                 (mis, d['mis_sem'], F_MIS, 2.6)):
            axA.plot(t, y, color=colr, lw=lw, ls=ls)
            axA.fill_between(t, y - np.array(sem), y + np.array(sem), color=colr, alpha=0.13, lw=0)
    axA.set_ylim(-0.03, 1.03)
    axA.set_xlabel('step ($t$)', fontsize=SZ['label'])
    axA.set_ylabel('failure proportion $\\varphi$', fontsize=SZ['label'])
    axA.set_title(f'failure over time ($M$ = {M_A}, $\\bar\\lambda$ = {LAM_B})',
                  loc='left', fontsize=SZ['subtitle'])

    # (b) steady-state failure vs environment size
    for c, (_, ls, mfc) in FORGET.items():
        th = np.array([size[f'{c}-{int(M)}'] for M in M_TH])
        axB.plot(M_TH, th.sum(1), color=F_ALL, lw=3.6, ls=ls)
        axB.plot(M_TH, th[:, 0], color=F_COL, lw=2.6, ls=ls)
        axB.plot(M_TH, th[:, 1], color=F_MIS, lw=2.6, ls=ls)
        for M in MS:
            col, mis = steady[f'{c}-{M}-{LAM_B}']
            axB.plot([M], [col + mis], marker='o', ms=9, ls='', markerfacecolor=mfc or F_ALL,
                     markeredgecolor=F_ALL, markeredgewidth=1.6, zorder=5)
    axB.set_xscale('log'); axB.set_ylim(-0.03, 1.03)
    axB.set_xlabel('environment size ($M$)', fontsize=SZ['label'])
    axB.set_ylabel('failure proportion $\\varphi$', fontsize=SZ['label'])
    axB.set_title(f'failure vs. size ($\\bar\\lambda$ = {LAM_B}, $t \\geq 200$)',
                  loc='left', fontsize=SZ['subtitle'])

    # (c) epistemic capacity vs environment speed (method color, opacity = phi)
    for c, (_, ls, mfc) in FORGET.items():
        for phi, al in PHIS:
            axC.plot(LAMS_C, [cap[(c, float(l), phi)] for l in LAMS_C], color=METHOD,
                     lw=3.2, ls=ls, alpha=al)
            for lam in LAM_C:                                  # mirror-derived capacity
                ph = np.array([sum(steady[f'{c}-{M}-{lam}']) for M in MS])
                k = np.flatnonzero((ph[:-1] < phi) & (ph[1:] >= phi))
                if len(k):
                    k = k[0]
                    lm = np.interp(phi, [ph[k], ph[k + 1]], np.log([MS[k], MS[k + 1]]))
                    axC.plot([lam], [np.exp(lm)], marker='o', ms=9, ls='', alpha=al,
                             markerfacecolor=mfc or METHOD, markeredgecolor=METHOD,
                             markeredgewidth=1.6, zorder=6)
    axC.set_xscale('log'); axC.set_yscale('log')
    axC.set_yticks([30, 100, 300, 1000]); axC.set_yticklabels(['30', '100', '300', '1000'])
    axC.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axC.set_xlabel('environment speed ($\\bar\\lambda$)', fontsize=SZ['label'])
    axC.set_ylabel('epistemic capacity $M^\\star(\\varphi)$', fontsize=SZ['label'])
    axC.set_title('epistemic capacity (no mitigation)', loc='left', fontsize=SZ['subtitle'])
    axC.legend(handles=[Line2D([], [], color=METHOD, lw=3.2, alpha=al, label=f'$\\varphi$ = {phi}')
                        for phi, al in PHIS[::-1]], loc='upper right', fontsize=SZ['legend'])

    for ax in (axA, axB, axC):
        ax.tick_params(labelsize=SZ['tick'])
    fig.tight_layout(w_pad=2.4, rect=(0, 0.12, 1, 1))

    # orthogonal factors -> separate legends, placed together below the panels
    leg_fail = fig.legend(handles=[Line2D([], [], color=F_ALL, lw=3.6, label='all failures'),
                                   Line2D([], [], color=F_COL, lw=3.0, label='collapse'),
                                   Line2D([], [], color=F_MIS, lw=3.0, label='misalignment')],
                          loc='lower center', ncol=3, bbox_to_anchor=(0.31, 0.0),
                          fontsize=SZ['legend'], handlelength=2.2, columnspacing=1.4)
    fig.add_artist(leg_fail)
    fig.legend(handles=[Line2D([], [], color=hv_style.INK, lw=3.0, ls=ls, marker='o', ms=9,
                               markerfacecolor=mfc or hv_style.INK, markeredgecolor=hv_style.INK,
                               label=f'{name} ($c$ = {c})')
                        for c, (name, ls, mfc) in FORGET.items()],
               loc='lower center', ncol=2, bbox_to_anchor=(0.72, 0.0),
               fontsize=SZ['legend'], handlelength=2.6, columnspacing=1.4)
    ec.save_artifact(fig, 'epistemic_collapse_figure1_capacity')


if __name__ == '__main__':
    compute()
    render()
