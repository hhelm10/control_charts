"""Matched validation of the mirror against the real system (gpt-4o-mini).

Each real run is replayed by the mirror on the SAME question set and probe set:
the real CLI draws its questions as default_rng(seed).choice(pool, M), which is
reconstructed exactly here, and the probe set is read from the run's snapshots.
Matching matters: each real seed reuses ~300 probe questions across all Fig 2
cells, so luck in that sample (e.g. more refusable questions than the pool's
1.5%) otherwise shows up as a systematic offset in every cell.

Real runs: Fig 2 (f2real-*, 650 runs, 10 per cell) and Fig 1 steady state
(f1real-ss-*, 20 per size); T = 100, steady state = probes at t >= 50.
REPS mirror replays per real run. Caches:
  data/figure2_proxy_matched.json, data/figure1_proxy_matched_ss.json
    {run name: {"real": [metrics], "proxy": [[metrics] per replay]}}
  metrics = (collapse, misalign, agent IDK, agent correct), steady state.
  Appendix A plots these four-metric comparisons.
Report: per cell the paired difference real - mirror (z from the spread of the
paired differences over runs), chi^2 over cells, and real/mirror capacity ratios
M*(phi) per Fig 2 setting.

    python epistemic_collapse_matched_validation.py [--force]
"""
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import yaml

import epistemic_collapse_common as ec
import epistemic_collapse_mirror as mirror
import epistemic_collapse_real as R

REPS = 10
T0 = 50
SETS = {  # cache: (config dir, name glob)
    f'{ec.DATA}/figure2_proxy_matched.json': ('real_fig2', 'f2real-*'),
    f'{ec.DATA}/figure1_proxy_matched_ss.json': ('real_fig1', 'f1real-ss-*'),
}
_Q = None


def _pool():
    global _Q
    if _Q is None:
        q = list(pd.read_parquet(f'{ec.DATA}/nq_embedded_oai.parquet', columns=['question']).question)
        _Q = (q, {s: i for i, s in enumerate(q)})
    return _Q


def _job(a):
    cdir, name = a
    Q, POS = _pool()
    c = yaml.safe_load(open(f'{ec.DATA}/configs/{cdir}/{name}.yaml'))
    s = c['simulation']
    M, N, seed = c['data']['total_questions'], c['agents']['count'], s['seed']
    idx = np.random.default_rng(seed).choice(len(Q), size=M, replace=False)   # as the real CLI
    d = R.run_dir(name)
    probe = [POS[q] for q in json.load(open(sorted(glob.glob(d + '/snapshots/*_meta.json'))[-1]))['questions']]
    out = []
    for rep in range(REPS):
        r = mirror.run(M=M, N=N, E=s['env_queries_per_turn'], K=s['questions_per_turn'],
                       k=c['agents']['retrieval_k'], c=s['forget_strategy']['decay_coefficient'],
                       lam=c['data']['lambda_mean'], edge_p=c['network']['edge_probability'],
                       seed=10000 * seed + rep, T=100, select_rule='uniform',
                       questions=idx, probe_questions=probe)
        k = np.asarray(r['t']) >= T0
        out.append([float(np.mean(r[m][k])) for m in ('collapse', 'misalign', 'agent_idk', 'correct')])
    return name, {'real': real_metrics(d), 'proxy': out}


def compute(force=False):
    for cache, (cdir, pat) in SETS.items():
        if os.path.exists(cache) and not force:
            continue
        names = sorted(os.path.basename(f)[:-5] for f in glob.glob(f'{ec.DATA}/configs/{cdir}/{pat}.yaml'))
        with ProcessPoolExecutor(max(1, os.cpu_count() - 2)) as ex:
            res = dict(ex.map(_job, [(cdir, n) for n in names], chunksize=1))
        json.dump(res, open(cache, 'w'))
        print('wrote', cache, len(res))


def real_metrics(d):
    """(collapse, misalign, agent IDK, agent correct) of a real run, probes at t >= T0."""
    acc = []
    for f in glob.glob(d + '/snapshots/snapshot_step_*_meta.json'):
        s = json.load(open(f))
        if s['step'] < T0:
            continue
        for j, q in enumerate(s['questions']):
            a = [row[j] for row in s['responses']]
            idk = np.array([x == R.IDK for x in a]); cor = np.array([x == s['truths'][q] for x in a])
            acc.append([idk.all(), not cor.any() and not idk.all(), idk.mean(), cor.mean()])
    return np.mean(acc, axis=0).tolist()


def _real(px, name):
    return np.array(px[name]['real'][:2])


def _proxy(px, name):
    return np.mean(px[name]['proxy'], axis=0)[:2]


def _crossing(Ms, y, phi):
    for i in range(len(Ms) - 1):
        if y[i] < phi <= y[i + 1]:
            f = (phi - y[i]) / (y[i + 1] - y[i])
            return float(np.exp(np.log(Ms[i]) + f * np.log(Ms[i + 1] / Ms[i])))
    return None


def report():
    for cache, (cdir, pat) in SETS.items():
        px = json.load(open(cache))
        cells = defaultdict(list)
        for name in px:
            cells[name.rsplit('-s', 1)[0]].append(name)
        chi, rows = 0.0, []
        for cell, names in cells.items():
            d = np.array([_real(px, n) - _proxy(px, n) for n in names])   # paired, per run
            tot = d.sum(axis=1)
            z = tot.mean() / max(tot.std(ddof=1) / np.sqrt(len(tot)), 1e-4)
            chi += z * z
            real = np.mean([_real(px, n).sum() for n in names])
            rows.append((cell, real, real - tot.mean(), z))
        print(f'\n{os.path.basename(cache)}: chi2 = {chi:.1f} / {len(cells)} cells, '
              f'mean z = {np.mean([r[3] for r in rows]):+.2f}')
        for cell, a, b, z in sorted(rows, key=lambda r: -abs(r[3]))[:5]:
            print(f'  {cell:24s} real {a:.3f}  mirror {b:.3f}  z {z:+.1f}')
        if cdir != 'real_fig2':
            continue
        by = defaultdict(dict)                                   # setting -> M -> (real, mirror)
        for cell, a, b, _ in rows:
            st, M = cell.replace('f2real-', '').rsplit('-M', 1)
            by[st][int(M)] = (a, b)
        for phi in (0.05, 0.1, 0.5):
            rat = []
            for st, v in by.items():
                Ms = sorted(v)
                a = _crossing(Ms, [v[m][0] for m in Ms], phi)
                b = _crossing(Ms, [v[m][1] for m in Ms], phi)
                if a and b:
                    rat.append(a / b)
            print(f'  M*(phi={phi}) real/mirror: median {np.median(rat):.2f}, '
                  f'range {min(rat):.2f}-{max(rat):.2f} ({len(rat)} settings)')


if __name__ == '__main__':
    compute(force='--force' in sys.argv)
    report()
