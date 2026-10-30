"""Appendix A: the measured-gate proxy reproduces real gpt-4o-mini agent networks.

1x4 calibration panels, one per varied factor (memory decay / mean degree /
environment speed / environment-query budget). Each point is one grid cell x
one metric: median over proxy seeds (x) against median over real seeds (y).
Color = method (no mitigation, sharing secondhand); marker shape = metric.
The identity line is perfect agreement; each panel reports median |delta|.

Compute stage (first run only): reads the raw run directories in the repo-level
experiments/results/ (gitignored, ~18 GB; grids real_grid_firsthand and
real_grid_mimesis, steady state = final 10 snapshots of 500 steps) and caches
per-run metrics in projects/epistemic_collapse/data/appendixA_proxy_validation.json.
"""
import epistemic_collapse_common as ec

import glob
import json
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import hv_style

hv_style.apply()
SZ = ec.SIZES

CACHE = f'{ec.DATA}/appendixA_proxy_validation.json'
GRIDS = {'f1v6': 'no mitigation', 'f1v7': 'sharing secondhand'}
FACTORS = {'memory decay': ['c0.05', 'base', 'c1.0'],
           'mean degree': ['deg2', 'base', 'mesh'],
           'environment speed': ['lam.001', 'base', 'lam.1'],
           'environment queries': ['E1', 'base', 'E6']}
METRICS = {'agent IDK': (0, 'o'), 'agent correct': (1, '^'),
           'collapse': (2, 's'), 'some agent correct': (3, 'D')}
IDK = ("i don't know",)


# ---------------------------------------------------------------- compute
def four(d):
    """(agent IDK, agent correct, collapse, some-agent-correct) over the final 10 snapshots."""
    fs = sorted(glob.glob(d + '/snapshots/snapshot_step_*_meta.json'))[-10:]
    if len(fs) < 5:
        return None
    acc = np.zeros(4); n = 0
    for f in fs:
        m = json.load(open(f))
        truths, qs, rows = m.get('truths', {}), m['questions'], m['responses']
        for j, q in enumerate(qs):
            col = [row[j] for row in rows]
            idk = [any(p in r.lower() for p in IDK) for r in col]
            cor = [r == truths.get(q) for r in col]
            acc += [np.mean(idk), np.mean(cor), all(idk), any(cor)]; n += 1
    return (acc / n).tolist()


def compute():
    if os.path.exists(CACHE):
        return
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for grid in GRIDS:
        for d in sorted(glob.glob(f'experiments/results/{grid}-*')):
            m = re.match(rf'.*/{grid}-(surr|real)-(.+)-s\d+_', d)
            if m and (v := four(d)) is not None:
                out[grid][m.group(1)][m.group(2)].append(v)
    json.dump(out, open(CACHE, 'w'), indent=1)


# ---------------------------------------------------------------- render
def render():
    data = json.load(open(CACHE))
    fig, axes = plt.subplots(1, 4, figsize=(22, 7.0), sharey=True)
    for ax, (fac, cells) in zip(axes, FACTORS.items()):
        devs = []
        for grid, method in GRIDS.items():
            st = ec.style(method)
            for cell in cells:
                S = np.array(data[grid]['surr'][cell]); R = np.array(data[grid]['real'][cell])
                for name, (i, mk) in METRICS.items():
                    sx, ry = np.median(S[:, i]), np.median(R[:, i])
                    ax.plot([sx], [ry], marker=mk, ms=10, ls='', color=st['color'],
                            markerfacecolor=st['color'], markeredgecolor='white',
                            markeredgewidth=0.8, zorder=4)
                    devs.append(abs(sx - ry))
        ax.plot([0, 1], [0, 1], color=hv_style.REFLINE, ls=':', lw=1.8, zorder=1)
        ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.03, 1.03); ax.set_aspect('equal')
        ax.set_title(fac, loc='left', fontsize=SZ['subtitle'])
        ax.set_xlabel('proxy', fontsize=SZ['label'])
        ax.tick_params(labelsize=SZ['tick'])
        ax.text(0.97, 0.03, f'median $|\\Delta|$ = {np.median(devs):.2f}', transform=ax.transAxes,
                ha='right', fontsize=SZ['annot'], color=hv_style.INK)
    axes[0].set_ylabel('real (gpt-4o-mini)', fontsize=SZ['label'])
    fig.tight_layout(w_pad=1.6, rect=(0, 0.13, 1, 1))
    leg = fig.legend(handles=[Line2D([], [], color=ec.style(m)['color'], marker='o', ls='', ms=11,
                                     label=m) for m in GRIDS.values()],
                     loc='lower left', ncol=2, bbox_to_anchor=(0.04, 0.0), fontsize=SZ['legend'])
    fig.add_artist(leg)
    fig.legend(handles=[Line2D([], [], color=hv_style.INK_MUTE, marker=mk, ls='', ms=11, label=n)
                        for n, (_, mk) in METRICS.items()],
               loc='lower right', ncol=4, bbox_to_anchor=(0.99, 0.0), fontsize=SZ['legend'])
    ec.save_artifact(fig, 'epistemic_collapse_appendixA_proxy_validation')


if __name__ == '__main__':
    compute()
    render()
