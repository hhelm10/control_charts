"""Appendix A: the proxy reproduces the real gpt-4o-mini agent networks.

1x4 panels, one per metric: collapse and misalignment (the two components of
epistemic failure) and the per-agent rates behind them (agent IDK, agent correct).
Each point is one real-system cell: a Fig 1 environment size (20 seeds) or a Fig 2
setting x size (10 seeds), 72 cells, 790 runs, T = 100, steady state t >= 50.
y = real mean over seeds; x = proxy mean over matched replays: each real run is
replayed 10 times on its own question set and probe set (see
epistemic_collapse_matched_validation.py, which computes the caches read here).
Each panel reports chi^2 over cells (paired real - proxy differences, z from the
spread over seeds) and the median |real - proxy|.

    python epistemic_collapse_appendixA_proxy_validation.py
"""
import epistemic_collapse_common as ec

import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import hv_style

hv_style.apply()
SZ = ec.SIZES

CACHES = {f'{ec.DATA}/figure1_proxy_matched_ss.json': ('o', 'Fig 1 (environment size)'),
          f'{ec.DATA}/figure2_proxy_matched.json': ('D', 'Fig 2 (parameter settings)')}
METRICS = [('collapse', 0, ec.FAILURE['collapse']),
           ('misalignment', 1, ec.FAILURE['misalignment']),
           ('agent IDK', 2, hv_style.INK_MUTE),
           ('agent correct', 3, hv_style.INK_MUTE)]


def cells():
    """{(cache, cell): [(real metrics, proxy mean metrics) per run]}."""
    out = defaultdict(list)
    for cache in CACHES:
        for name, v in json.load(open(cache)).items():
            out[(cache, name.rsplit('-s', 1)[0])].append(
                (np.array(v['real']), np.mean(v['proxy'], axis=0)))
    return out


def render():
    cs = cells()
    fig, axes = plt.subplots(1, 4, figsize=(22, 7.0), sharey=True)
    for ax, (label, i, color) in zip(axes, METRICS):
        chi, devs = 0.0, []
        for (cache, _), runs in cs.items():
            r = np.array([a[i] for a, _ in runs]); p = np.array([b[i] for _, b in runs])
            d = r - p
            chi += (d.mean() / max(d.std(ddof=1) / np.sqrt(len(d)), 1e-4)) ** 2
            devs.append(abs(d.mean()))
            ax.plot([p.mean()], [r.mean()], marker=CACHES[cache][0], ms=9, ls='', color=color,
                    markeredgecolor='white', markeredgewidth=0.8, alpha=0.85, zorder=4)
        ax.plot([0, 1], [0, 1], color=hv_style.REFLINE, ls=':', lw=1.8, zorder=1)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect('equal')
        ax.set_title(label, loc='left', fontsize=SZ['subtitle'])
        ax.set_xlabel('proxy', fontsize=SZ['label'])
        ax.tick_params(labelsize=SZ['tick'])
        ax.text(0.97, 0.03, f'$\\chi^2$ = {chi:.0f} / {len(cs)} cells\nmedian $|\\Delta|$ = {np.median(devs):.3f}',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=SZ['annot'], color=hv_style.INK)
    axes[0].set_ylabel('real (gpt-4o-mini)', fontsize=SZ['label'])
    fig.tight_layout(w_pad=1.6, rect=(0, 0.1, 1, 1))
    fig.legend(handles=[Line2D([], [], color=hv_style.INK_MUTE, marker=mk, ls='', ms=10, label=lab)
                        for mk, lab in CACHES.values()],
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=SZ['legend'])
    ec.save_artifact(fig, 'epistemic_collapse_appendixA_proxy_validation')


if __name__ == '__main__':
    render()
