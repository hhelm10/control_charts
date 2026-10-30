"""Figure 4: relative epistemic capacity of books, Gamma(phi) =
M*_books(phi) / M*_none(phi), as a function of the failure tolerance phi.

Books (mirror, books=True; no mimesis otherwise): a shared book with one entry
per question, last write wins, entry age invisible to readers. Each step an
agent attempts a write w.p. p, spending its whole budget; its W-th attempt
commits its current answers from memory, of which W * B (random) are verified
against the environment and the rest copied unchecked, so they may be stale
("slop"). Every ask reads the book. Expected write rate per agent: p / W.

Each panel shows three values of one parameter (shade, light = low) with the
others at the default (c = 0.05, lambda = 0.03, E = 3, K = 8, N = 10, k = 3,
degree 6; p = 0.05, W = 5):
  (e) write rate: p in {0.02, 0.05, 0.25} at W = 5 (more writing, more budget spent)
  (f) cost per write: W in {1, 5, 20} at fixed write rate p / W = 0.01
      (cheaper writes verify less: W * B verifications per write)
The solid gray line marks Gamma = 1 (the no-books baseline).

Capacity = log-M interpolated crossing of the seed-mean steady-state failure
curve (uniform question selection, t >= 200). The no-books baseline comes from
the Figure 3 cache (same seeds). Bands: +-1 SEM from a bootstrap over seeds
(on log Gamma). Lines end where either capacity exceeds the largest simulated
environment (M = 20000).

Compute caches per-seed [failure, slop] in data/figure4_mirror.json;
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

hv_style.apply()
SZ = ec.SIZES

TITLES = {
    'lam': 'environment speed ($\\bar\\lambda$)',
    'c': 'forgetting speed ($c$)',
    'E': 'environment queries per step ($E$)',
    'N': 'number of agents ($N$)',
    'p': 'write rate ($p$, $W$ = 5)',
    'W': 'cost per write ($W$, $p/W$ = 0.01)',
}

B = 11
BASE = dict(lam=0.03, c=0.05, E=3, N=10, k=3, deg=6, p=0.05, W=5)
BOOK_KEYS = ('p', 'W')
# plotted (and simulated): key -> (symbol, three values low -> high, format)
PARAMS = {
    'lam': ('$\\bar\\lambda$', [0.003, 0.03, 0.3], '{:g}'),
    'c':   ('$c$', [0.01, 0.05, 0.5], '{:g}'),
    'E':   ('$E$', [1, 3, 8], '{:g}'),
    'N':   ('$N$', [3, 10, 20], '{:g}'),
    'p':   ('$p$', [0.02, 0.05, 0.25], '{:g}'),
    'W':   ('$W$', [1, 5, 20], '{:g}'),
}
SIM = {key: vals for key, (_, vals, _) in PARAMS.items()}
SHADES = ('#9ccbff', '#3596ff', '#0b4f9e')          # parameter value, low -> high (method hue, light -> dark)
PHI_GRID = np.round(np.arange(0.05, 0.901, 0.01), 2)   # below 0.05 the crossings sit at M <= 8, where curves are flat (noise)
MG = [6, 8, 10, 13, 16, 20, 25, 32, 40, 50, 63, 80, 100, 126, 160, 200, 250, 320, 400, 500,
      630, 1000, 1600, 2500, 4000, 6300, 10000, 16000, 20000]   # dense where the strict-tolerance crossings sit; pool has 25000
SEEDS = range(600, 612)
N_BOOT = 400
CACHE = f'{ec.DATA}/figure4_mirror.json'
BASELINE = f'{ec.DATA}/figure3_mirror.json'      # no-books runs (Figure 3, same seeds)
ST = ec.style('books')
BASE_ST = ec.style('no mitigation')                     # Gamma = 1 baseline line


def setting(key, v):
    s = dict(BASE)
    s[key] = v
    if key == 'W':                       # fixed write rate p / W
        s['p'] = 0.01 * v
    return s


def base_key(s):
    """the Figure 3 cache key of the matching no-books setting"""
    return key_of({k: v for k, v in s.items() if k not in BOOK_KEYS})


def key_of(s):
    return '-'.join(f'{k}={s[k]:g}' for k in sorted(s))


# ---------------------------------------------------------------- compute
def _run(job):
    sk, M, seed = job
    s = {k: float(v) for k, v in (kv.split('=') for kv in sk.split('-'))}
    N, W = int(s['N']), int(s['W'])
    r = mirror.run(M=M, N=N, E=int(s['E']), K=B - int(s['E']), k=int(s['k']), c=s['c'], lam=s['lam'],
                   edge_p=min(1.0, s['deg'] / (N - 1)), seed=seed, T=300, select_rule='uniform',
                   probe=min(M, 100), policy='firsthand', books=True, write_p=s['p'], write_W=W)
    return job, [float(np.mean(r['collapse'][-10:]) + np.mean(r['misalign'][-10:])),
                 float(np.mean(r['book_slop'][-10:]))]


def settings():
    return {key_of(setting(key, v)) for key, vals in SIM.items() for v in vals}


def compute():
    """cache: '{setting}|{M}' -> {seed: [failure, slop]}; fills only missing runs, checkpoints."""
    base = json.load(open(BASELINE))
    for sk in settings():                                   # the baseline must already exist
        s = {k: float(v) for k, v in (kv.split('=') for kv in sk.split('-'))}
        bk = base_key(s)
        assert all(str(x) in base.get(f'firsthand|{bk}|{M}', {}) for M in MG for x in SEEDS), bk
    out = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    jobs = [(sk, M, s) for sk in settings() for M in MG for s in SEEDS
            if str(s) not in out.get(f'{sk}|{M}', {})]
    if not jobs:
        return
    jobs.sort(key=lambda j: -j[1] * float(j[0].split('N=')[1].split('-')[0]))   # biggest first
    with ProcessPoolExecutor(max_workers=max(1, os.cpu_count() - 2)) as ex:
        for i, ((sk, M, s), v) in enumerate(ex.map(_run, jobs, chunksize=1)):
            out.setdefault(f'{sk}|{M}', {})[str(s)] = v
            if i % 1000 == 999:
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


def gamma_curve(d, base, sk, rng):
    """Gamma(phi) on PHI_GRID from seed means, plus bootstrap SEM (seeds resampled jointly
    for books and baseline). Uses the seeds present for every cell of this setting."""
    s = {k: float(v) for k, v in (kv.split('=') for kv in sk.split('-'))}
    bk = base_key(s)
    seeds = [str(x) for x in SEEDS if all(str(x) in d.get(f'{sk}|{M}', {}) for M in MG)]
    Yb = np.array([[d[f'{sk}|{M}'][x][0] for x in seeds] for M in MG])
    Y0 = np.array([[base[f'firsthand|{bk}|{M}'][x] for x in seeds] for M in MG])
    g = crossings(Yb.mean(1), PHI_GRID) / crossings(Y0.mean(1), PHI_GRID)
    n = len(seeds)
    i = rng.integers(0, n, (N_BOOT, n))
    b = np.log(crossings(Yb[:, i].mean(2), PHI_GRID) / crossings(Y0[:, i].mean(2), PHI_GRID))
    fin = np.isfinite(b)
    se = np.where(fin.mean(1) > 0.9, np.nanstd(np.where(fin, b, np.nan), 1), np.nan)
    return g, se, n                                                  # se is on log Gamma


def render():
    d = json.load(open(CACHE))
    base = json.load(open(BASELINE))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(21, 12.2), sharex=True, sharey=True)
    lw = ST['lw'] + 1.8
    n_used = []
    for ax, (key, (sym, vals, fmt)) in zip(axes.flat, PARAMS.items()):
        handles = []
        for v, col in zip(vals, SHADES):
            g, se, n = gamma_curve(d, base, key_of(setting(key, v)), rng)
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
    ticks = [0.5, 0.71, 1, 1.41, 2, 2.83]
    ax0.set_yticks(ticks); ax0.set_yticklabels(['0.5', '0.7', '1', '1.4', '2', '2.8']); ax0.set_ylim(0.42, 4.6)
    ax0.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for ax in axes[:, 0]:
        ax.set_ylabel('relative capacity $\\Gamma(\\varphi)$', fontsize=SZ['label'])
    for ax in axes[1]:
        ax.set_xlabel('failure tolerance ($\\varphi$)', fontsize=SZ['label'])
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.165, top=0.96, wspace=0.08, hspace=0.22)
    fig.legend(handles=[Line2D([], [], color=ST['color'], ls=ST['ls'], lw=lw, label='w/ books'),
                        Line2D([], [], color=BASE_ST['color'], ls=BASE_ST['ls'], lw=3.0, label='w/ no books (baseline)')],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=SZ['legend'],
               handlelength=2.8, title='strategy',
               title_fontsize=SZ['legend'])
    ec.save_artifact(fig, 'epistemic_collapse_figure4_books')
    print(f'seeds used per setting: min {min(n_used)}, max {max(n_used)}')


if __name__ == '__main__':
    compute()
    render()
