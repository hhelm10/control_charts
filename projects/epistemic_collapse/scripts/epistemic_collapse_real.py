"""Read real-system runs (experiments/results/<name>_<timestamp>/) into failure
series, with the same definitions as the mirror:
  collapse      every agent answers "I don't know"
  misalignment  no agent is correct, but at least one answers (stale)
per probed question, averaged over the probe sample at each snapshot.
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
RESULTS = os.path.join(ROOT, 'experiments', 'results')
IDK = "I don't know"


def run_dir(name):
    """latest finished run directory for an experiment name, or None."""
    ds = sorted(d for d in glob.glob(os.path.join(RESULTS, f'{name}_*'))
                if os.path.exists(os.path.join(d, 'results.json')))
    return ds[-1] if ds else None


def failure_series(d):
    """(t, collapse, misalignment) arrays over the run's snapshots."""
    t, col, mis = [], [], []
    for f in sorted(glob.glob(os.path.join(d, 'snapshots', 'snapshot_step_*_meta.json'))):
        s = json.load(open(f))
        R, truth = s['responses'], s['truths']          # R: agents x questions
        c = m = 0
        for j, q in enumerate(s['questions']):
            a = [R[i][j] for i in range(len(R))]
            if all(x == IDK for x in a):
                c += 1
            elif not any(x == truth[q] for x in a):
                m += 1
        n = len(s['questions'])
        t.append(s['step']); col.append(c / n); mis.append(m / n)
    o = np.argsort(t)
    return np.array(t)[o], np.array(col)[o], np.array(mis)[o]


def usage_cost(d, price_in=0.15e-6, price_out=0.6e-6):
    """API cost of a run in dollars (gpt-4o-mini prices)."""
    u = json.load(open(os.path.join(d, 'results.json'))).get('llm_usage', {})
    return u.get('prompt_tokens', 0) * price_in + u.get('completion_tokens', 0) * price_out
