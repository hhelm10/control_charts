"""Figure 2: sensitivity of epistemic capacity to the system's parameters
(no mitigation).

Each panel varies one parameter from the default setting and plots epistemic
capacity M*(phi), the largest environment whose steady-state failure stays
below phi, for phi = 0.1 and 0.5:
  (a) environment speed lambda      (b) forgetting speed c
  (c) environment queries per step E (K = 11 - E peer asks)
  (d) number of agents N            (e) retrieval depth k
  (f) peer degree (mean neighbors per agent, Erdos-Renyi)
phi = 0.05, 0.1, 0.5 (opacity). Default: c = 0.05, lambda = 0.03, E = 3 (K = 8), N = 10, k = 3, degree 6
(the panel titles name each parameter; x-axes give its symbol).

Theory lines (analytic model; it has no topology, so (f) is flat) and mirror
markers (capacity = log-M interpolated crossing of the seed-mean steady-state
failure curve; uniform question selection, t >= 200). The mirror uses the
measured answering gate, which exists only for k = 1, 3, 7.

Real system (gpt-4o-mini): open diamonds = log-M interpolated crossing of the
mean failure over 10 real runs per cell (T = 100, probes at t >= 50; runs
f2real-*, cache data/figure2_real.json); the default value in every panel uses
the Fig 1 real runs (20 per size, data/figure1_real.json). Cells bracket the
mirror's crossings, so a real point is missing where its crossing falls
outside the cells that were run.

Channels: color + solid linestyle = the method (no mitigation); opacity = phi;
line = theory, filled marker = simulation, open diamond = real system.

Compute stages cache into data/figure2_theory.json (~1 min) and
data/figure2_mirror.json (~15 min); rendering reads only the caches.
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

TITLES = {
    'lam': 'environment speed ($\\bar\\lambda$)',
    'c': 'forgetting speed ($c$)',
    'E': 'environment queries per step ($E$)',
    'N': 'number of agents ($N$)',
    'k': 'retrieval depth ($k$)',
    'deg': 'peer degree ($d$)',
}

B = 11
BASE = dict(lam=0.03, c=0.05, E=3, N=10, k=3, deg=6)
# key: (axis label, theory grid, mirror values, log x)
PARAMS = {
    'lam': ('$\\bar\\lambda$', np.geomspace(1e-3, 1, 25), [0.003, 0.03, 0.3], True),
    'c':   ('$c$', np.geomspace(0.01, 1, 25), [0.02, 0.05, 0.15, 0.5], True),
    'E':   ('$E$', np.arange(1, 11), [1, 3, 6], False),
    'N':   ('$N$', np.array([2, 3, 4, 5, 7, 10, 14, 20, 30, 40]), [3, 10, 30], True),
    'k':   ('$k$', np.arange(1, 11), [1, 3, 7], False),
    'deg': ('$d$', None, [1, 3, 6, 9], False),
}
PHIS = ((0.05, 0.22), (0.1, 0.45), (0.5, 1.0))   # (tolerance, opacity)
MG = [6, 10, 16, 25, 40, 63, 100, 160, 250, 400, 630, 1000, 1600, 2500, 4000, 6300, 10000, 16000, 20000]   # pool has 25000 questions
SEEDS = range(500, 508)
TH = f'{ec.DATA}/figure2_theory.json'
MIR = f'{ec.DATA}/figure2_mirror.json'
REAL = f'{ec.DATA}/figure2_real.json'
REAL_F1 = f'{ec.DATA}/figure1_real.json'
REAL_TAG = {'lam': 'lam', 'c': 'c', 'E': 'E', 'N': 'N', 'k': 'k', 'deg': 'deg'}
ST = ec.style('no mitigation')


def setting(key, v):
    s = dict(BASE)
    s[key] = v
    return s


def key_of(s):
    return '-'.join(f'{k}={s[k]:g}' for k in sorted(s))


# ---------------------------------------------------------------- compute
def total(M, s):
    x = misalignment(int(M), s['lam'], N=int(s['N']), alpha=s['E'] / B, c=s['c'], k=int(s['k']),
                     asks='uniform')
    return x['collapse'] + x['misalign']


def _mstar(job):
    """theory capacity: root of total failure(M) = phi on log M."""
    from scipy.optimize import brentq
    key, v, phi = job
    s = setting(key, v)
    f = lambda lm: total(round(np.exp(lm)), s) - phi
    try:
        m = float(np.exp(brentq(f, np.log(10), np.log(50000), xtol=0.01)))
    except ValueError:
        m = float('nan')
    return f'{key}-{v:g}-{phi}', m


def _run(job):
    sk, M, seed = job
    s = dict(kv.split('=') for kv in sk.split('-'))
    s = {k: float(v) for k, v in s.items()}
    N = int(s['N'])
    r = mirror.run(M=M, N=N, E=int(s['E']), K=B - int(s['E']), k=int(s['k']), c=s['c'], lam=s['lam'],
                   edge_p=min(1.0, s['deg'] / (N - 1)), seed=seed, T=300, select_rule='uniform',
                   probe=min(M, 100))
    return job, float(np.mean(r['collapse'][-10:]) + np.mean(r['misalign'][-10:]))


def mirror_settings():
    out = {}
    for key, (_, _, vals, _) in PARAMS.items():
        for v in vals:
            s = setting(key, v)
            out[key_of(s)] = s
    return out


def compute():
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        th = json.load(open(TH)) if os.path.exists(TH) else {}
        jobs = [(key, float(v), phi) for key, (_, grid, _, _) in PARAMS.items() if grid is not None
                for v in grid for phi, _ in PHIS]
        jobs += [('deg', float(BASE['deg']), phi) for phi, _ in PHIS]
        jobs = [j for j in jobs if f'{j[0]}-{j[1]:g}-{j[2]}' not in th]   # incremental
        if jobs:
            th.update(dict(ex.map(_mstar, jobs)))
            json.dump(th, open(TH, 'w'), indent=1)
        out = json.load(open(MIR)) if os.path.exists(MIR) else {}
        jobs = [(sk, M, s) for sk in mirror_settings() for M in MG for s in SEEDS
                if f'{sk}|{M}' not in out]                     # incremental: missing cells only
        if jobs:
            jobs.sort(key=lambda j: -j[1] * float(j[0].split('N=')[1].split('-')[0]))
            new = {}
            for (sk, M, s), v in ex.map(_run, jobs, chunksize=1):
                new.setdefault(f'{sk}|{M}', []).append(v)
            out.update(new)
            json.dump(out, open(MIR, 'w'), indent=1)


# ---------------------------------------------------------------- render
def crossing(y, phi, grid, skip_first=False):
    """log-interpolated environment size where the failure curve reaches phi on its rising
    branch (the last upward crossing). At small M failure is not monotone: the refusal floor
    gives a noisy bump (~0.05-0.08 at M ~ 10) before the curve dips and rises, and a first-
    crossing rule would pick up that bump instead of the capacity. On the mirror grid
    (from M = 6) a crossing in the first interval is that bump itself: skip_first drops it."""
    y = np.asarray(y)
    k = np.flatnonzero((y[:-1] < phi) & (y[1:] >= phi))
    k = k[k > 0] if skip_first else k
    if not len(k):
        return None
    k = k[-1]
    return float(np.exp(np.interp(phi, [y[k], y[k + 1]], np.log([grid[k], grid[k + 1]]))))


def real_mstar(key, v, phi, real, f1):
    """real-system capacity for one setting (None if not bracketed by the runs)."""
    if v == BASE[key]:
        Ms = sorted(int(m) for m in f1)
        y = [float(np.mean(np.sum(f1[str(m)], axis=1))) for m in Ms]
    else:
        tag = f'f2real-{REAL_TAG[key]}{v:g}-M'
        Ms = sorted(int(c[len(tag):]) for c in real if c.startswith(tag))
        y = [float(np.mean(np.sum(real[f'{tag}{m}'], axis=1))) for m in Ms]
    return crossing(y, phi, Ms) if len(Ms) > 1 else None


def render():
    th = json.load(open(TH))
    mir = json.load(open(MIR))
    real = json.load(open(REAL)) if os.path.exists(REAL) else {}
    f1 = json.load(open(REAL_F1))['steady'] if os.path.exists(REAL_F1) else {}
    fig, axes = plt.subplots(2, 3, figsize=(21, 12.2), sharey=True)
    lw = ST['lw'] + 1.8
    for ax, (key, (label, grid, vals, logx)) in zip(axes.flat, PARAMS.items()):
        for phi, al in PHIS:
            if grid is not None:
                ax.plot(grid, [th[f'{key}-{float(v):g}-{phi}'] for v in grid], color=ST['color'],
                        ls=ST['ls'], lw=lw, alpha=al)
            else:                                          # theory has no topology: flat
                ax.axhline(th[f'deg-{float(BASE["deg"]):g}-{phi}'], color=ST['color'], ls=ST['ls'],
                           lw=lw, alpha=al)
            for v in vals:
                sk = key_of(setting(key, v))
                y = [np.mean(mir[f'{sk}|{M}']) for M in MG]
                m = crossing(y, phi, MG, skip_first=True)
                if m:
                    ax.plot([v], [m], marker='o', ms=10, ls='', color=ST['color'], alpha=al,
                            markeredgecolor=ST['color'], zorder=6)
                mr = real_mstar(key, v, phi, real, f1) if real else None
                if mr:
                    ax.plot([v], [mr], marker='D', ms=11, ls='', mfc='none', mew=2.2,
                            color=hv_style.INK, alpha=max(al, 0.35), zorder=7)
        ax.set_xlabel(label, fontsize=SZ['label'])
        ax.set_title(f'({"abcdef"[list(PARAMS).index(key)]}) {TITLES[key]}', loc='left',
                     fontsize=SZ['subtitle'])
        if logx:
            ax.set_xscale('log')
        if key in ('E', 'k'):
            ax.set_xticks([1, 2, 4, 6, 8, 10])
        if key == 'c':
            ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1]); ax.set_xticklabels(['0.01', '0.03', '0.1', '0.3', '1'])
        if key == 'N':
            ax.set_xticks([2, 5, 10, 20, 40]); ax.set_xticklabels(['2', '5', '10', '20', '40'])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if key == 'deg':
            ax.set_xticks(vals); ax.set_xlim(0.5, 9.5)
        ax.tick_params(labelsize=SZ['tick'])
    ax0 = axes[0, 0]
    ax0.set_yscale('log')
    ax0.set_yticks([10, 30, 100, 300, 1000, 3000, 10000, 30000])
    ax0.set_yticklabels(['10', '30', '100', '300', '1k', '3k', '10k', '30k'])
    ax0.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax0.set_ylim(8, 40000)
    for ax in axes[:, 0]:
        ax.set_ylabel('epistemic capacity $M^\\star(\\varphi)$', fontsize=SZ['label'])
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.17, top=0.96, wspace=0.08, hspace=0.36)

    # orthogonal factors -> separate legends below the panels
    leg = fig.legend(handles=[Line2D([], [], color=ST['color'], ls=ST['ls'], lw=lw, alpha=al,
                                     label=f'$\\varphi$ = {phi}') for phi, al in PHIS[::-1]],
                     loc='lower center', ncol=3, bbox_to_anchor=(0.32, 0.0), columnspacing=1.6, fontsize=SZ['legend'],
                     handlelength=2.8, title='failure tolerance', title_fontsize=SZ['legend'])
    fig.add_artist(leg)
    fig.legend(handles=[Line2D([], [], color=hv_style.INK, ls=ST['ls'], lw=lw, label='theory'),
                        Line2D([], [], color=hv_style.INK, ls='', marker='o', ms=10, label='simulation'),
                        Line2D([], [], color=hv_style.INK, ls='', marker='D', ms=11, mfc='none', mew=2.2,
                               label='real system')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.72, 0.0), fontsize=SZ['legend'],
               handlelength=2.8, title='source', title_fontsize=SZ['legend'])
    ec.save_artifact(fig, 'epistemic_collapse_figure2_sensitivity')


if __name__ == '__main__':
    compute()
    render()
