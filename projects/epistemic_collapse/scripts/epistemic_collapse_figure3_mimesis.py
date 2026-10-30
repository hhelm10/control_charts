"""Figure 3: relative epistemic capacity of sharing secondhand information
(mimesis), Gamma(phi) = M*_mimesis(phi) / M*_none(phi), as a function of the
failure tolerance phi.

Same parameters as Figure 2; each panel shows three values of one parameter
(low / default / high; shade, light = low) with the others at the default
(c = 0.05, lambda = 0.03, E = 3, K = 8, N = 10, k = 3, degree 6).
Gamma > 1: mimesis lets the system sustain a larger environment at the same
failure tolerance; the solid gray line marks Gamma = 1 (the no-mimesis baseline).

Mirror simulation only (the analytic model has no misalignment term under
mimesis yet). For each setting and strategy, capacity is the log-M
interpolated crossing of the seed-mean steady-state failure curve (uniform
question selection, t >= 200); both strategies use the same seeds. Bands:
+-1 SEM from a bootstrap over seeds (on log Gamma).
Lines end where either strategy's capacity exceeds the largest simulated
environment (M = 20000; the question pool has 25000), so Gamma is undefined
there; Gamma -> 1 as phi -> 1 (at phi = 1 both capacities are infinite).

Channels: hue + dashed linestyle = the method (sharing secondhand);
shade of that hue = parameter value (light = low).

Compute caches per-seed failure in data/figure3_mirror.json (~2 h, both
strategies, all SIM values); rendering reads only the cache.
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
# simulated values per parameter (compute)
SIM = {
    'lam': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    'c':   [0.01, 0.02, 0.05, 0.15, 0.5],
    'E':   [1, 2, 3, 4, 6, 8],
    'N':   [3, 5, 10, 20],
    'k':   [1, 3, 7],                      # measured answering gate
    'deg': [1, 2, 3, 6, 9],
}
# plotted: key -> (symbol, three values low -> high, format)
PARAMS = {
    'lam': ('$\\bar\\lambda$', [0.003, 0.03, 0.3], '{:g}'),
    'c':   ('$c$', [0.01, 0.05, 0.5], '{:g}'),
    'E':   ('$E$', [1, 3, 8], '{:g}'),
    'N':   ('$N$', [3, 10, 20], '{:g}'),
    'k':   ('$k$', [1, 3, 7], '{:g}'),
    'deg': ('$d$', [1, 3, 6], '{:g}'),
}
SHADES = ('#a9bdda', '#6f8dbd', '#2f4f86')          # parameter value, low -> high (method hue, light -> dark)
PHI_GRID = np.round(np.arange(0.05, 0.901, 0.01), 2)   # below 0.05 the crossings sit at M <= 8, where curves are flat (noise)
MG = [6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 63, 80, 100, 126, 160, 200, 250, 320, 400, 500,
      630, 1000, 1600, 2500, 4000, 6300, 10000, 16000, 20000]   # dense where the strict-tolerance crossings sit; pool has 25000
SEEDS = range(600, 624)
POLICIES = ('firsthand', 'open')                  # no mitigation, mimesis
N_BOOT = 400
CACHE = f'{ec.DATA}/figure3_mirror.json'
ST = ec.style('sharing secondhand')
BASE_ST = ec.style('no mitigation')                # Gamma = 1 baseline line


def setting(key, v):
    s = dict(BASE)
    s[key] = v
    return s


def key_of(s):
    return '-'.join(f'{k}={s[k]:g}' for k in sorted(s))


# ---------------------------------------------------------------- compute
def _run(job):
    pol, sk, M, seed = job
    s = {k: float(v) for k, v in (kv.split('=') for kv in sk.split('-'))}
    N = int(s['N'])
    r = mirror.run(M=M, N=N, E=int(s['E']), K=B - int(s['E']), k=int(s['k']), c=s['c'], lam=s['lam'],
                   edge_p=min(1.0, s['deg'] / (N - 1)), seed=seed, T=300, select_rule='uniform',
                   probe=min(M, 100), policy=pol)
    return job, float(np.mean(r['collapse'][-10:]) + np.mean(r['misalign'][-10:]))


def settings():
    return {key_of(setting(key, v)) for key, vals in SIM.items() for v in vals}


def compute():
    """cache: '{policy}|{setting}|{M}' -> {seed: failure}; fills only missing (cell, seed) runs
    and checkpoints as it goes, so more seeds or sizes can be added later."""
    out = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    jobs = [(p, sk, M, s) for p in POLICIES for sk in settings() for M in MG for s in SEEDS
            if str(s) not in out.get(f'{p}|{sk}|{M}', {})]
    if not jobs:
        return
    jobs.sort(key=lambda j: -j[2] * float(j[1].split('N=')[1].split('-')[0]))   # biggest first
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, ((p, sk, M, s), v) in enumerate(ex.map(_run, jobs, chunksize=1)):
            out.setdefault(f'{p}|{sk}|{M}', {})[str(s)] = v
            if i % 2000 == 1999:
                json.dump(out, open(CACHE, 'w'))
    json.dump(out, open(CACHE, 'w'))


# ---------------------------------------------------------------- render
def crossings(Y, phis):
    """log-interpolated environment size where each seed-mean failure curve first reaches each phi.
    Y: (M, ...) failure; returns (len(phis), ...)."""
    lm = np.log(MG)
    out = np.full((len(phis),) + Y.shape[1:], np.nan)
    for j, phi in enumerate(phis):
        above = Y >= phi
        first = np.where(above.any(0), above.argmax(0), -1)        # first grid index at/above phi
        ok = first > 0                                              # starts below phi, crosses on grid
        i = np.where(ok, first, 1)
        idx = np.indices(i.shape)
        y0, y1 = Y[(i - 1,) + tuple(idx)], Y[(i,) + tuple(idx)]
        w = (phi - y0) / np.where(y1 > y0, y1 - y0, 1)
        out[j] = np.where(ok, np.exp(lm[i - 1] + w * (lm[i] - lm[i - 1])), np.nan)
    return out


def gamma_curve(d, sk, rng):
    """Gamma(phi) on PHI_GRID from seed means, plus bootstrap SEM (seeds resampled jointly
    for both strategies). Uses the seeds present for every cell of this setting."""
    seeds = [str(s) for s in SEEDS
             if all(str(s) in d.get(f'{p}|{sk}|{M}', {}) for p in POLICIES for M in MG)]
    Y = {p: np.array([[d[f'{p}|{sk}|{M}'][s] for s in seeds] for M in MG]) for p in POLICIES}
    g = crossings(Y['open'].mean(1), PHI_GRID) / crossings(Y['firsthand'].mean(1), PHI_GRID)
    n = len(seeds)
    i = rng.integers(0, n, (N_BOOT, n))                              # (boot, n)
    bo = crossings(Y['open'][:, i].mean(2), PHI_GRID)               # (phi, boot)
    bf = crossings(Y['firsthand'][:, i].mean(2), PHI_GRID)
    b = np.log(bo / bf)
    fin = np.isfinite(b)
    se = np.where(fin.mean(1) > 0.9, np.nanstd(np.where(fin, b, np.nan), 1), np.nan)
    return g, se, n                                                  # se is on log Gamma


def render():
    d = json.load(open(CACHE))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12.2), sharex=True, sharey=True)
    lw = ST['lw'] + 1.8
    n_used = []
    for ax, (key, (sym, vals, fmt)) in zip(axes.flat, PARAMS.items()):
        handles = []
        for v, col in zip(vals, SHADES):
            g, se, n = gamma_curve(d, key_of(setting(key, v)), rng)
            n_used.append(n)
            ax.plot(PHI_GRID, g, color=col, ls=ST['ls'], lw=lw)
            band = np.isfinite(g) & np.isfinite(se)
            ax.fill_between(PHI_GRID[band], (g * np.exp(-se))[band], (g * np.exp(se))[band],
                            color=col, alpha=0.15, lw=0)
            tag = ' (default)' if v == BASE[key] else ''
            handles.append(Line2D([], [], color=col, ls=ST['ls'], lw=lw,
                                  label=f'{sym} = {fmt.format(v)}{tag}'))
        ax.axhline(1.0, color=BASE_ST['color'], ls=BASE_ST['ls'], lw=3.0, zorder=1.8)   # above the grid, below the data
        ax.legend(handles=handles, loc='upper right', fontsize=SZ['legend'] * 0.85, handlelength=2.4,
                  frameon=False, labelcolor=hv_style.INK)
        ax.set_title(f'({"abcdef"[list(PARAMS).index(key)]}) {TITLES[key]}', loc='left',
                     fontsize=SZ['subtitle'])
        ax.tick_params(labelsize=SZ['tick'])
    ax0 = axes[0, 0]
    ax0.set_xlim(0.05, 0.9)
    ax0.set_yscale('log')
    ticks = [0.5, 0.71, 1, 1.41, 2]
    ax0.set_yticks(ticks); ax0.set_yticklabels(['0.5', '0.7', '1', '1.4', '2']); ax0.set_ylim(0.45, 2.2)
    ax0.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes[:, 0]:
        ax.set_ylabel('relative capacity $\\Gamma(\\varphi)$', fontsize=SZ['label'])
    for ax in axes[1]:
        ax.set_xlabel('failure tolerance ($\\varphi$)', fontsize=SZ['label'])
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.165, top=0.96, wspace=0.08, hspace=0.22)
    fig.legend(handles=[Line2D([], [], color=ST['color'], ls=ST['ls'], lw=lw, label='w/ mimesis'),
                        Line2D([], [], color=BASE_ST['color'], ls=BASE_ST['ls'], lw=3.0, label='w/ no mimesis (baseline)')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=SZ['legend'],
               handlelength=2.8, title='strategy',
               title_fontsize=SZ['legend'])
    ec.save_artifact(fig, 'epistemic_collapse_figure3_mimesis')
    print(f'seeds used per setting: min {min(n_used)}, max {max(n_used)}')


if __name__ == '__main__':
    compute()
    render()
