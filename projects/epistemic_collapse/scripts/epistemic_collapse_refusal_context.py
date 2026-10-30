"""Context-dependent refusal, measured in vivo on the logged real runs
(data/llm_gate_dumps: f1real-* at k = 3, gate-k1-* at k = 1).

The real agent's IDK on a question whose own firsthand fact is retrieved depends on
  - the question's class, from the offline whole-pool measurements:
      "k3"  refused with 2 nearest neighbours as context (refusal_oai.json, context A)
      "k1"  refused with its own fact alone (refusal_oai_k1.json), but not "k3"
      "none" otherwise
  - n_other: how many retrieved entries are about OTHER questions (0, 1, 2+). With
    few distinct questions (small M) contexts fill with copies of the same question
    and refusal rises towards its k = 1 level.
For "none" questions the residual IDK is tiny and depends on the fact's position.

Within a run, refusal of a question is correlated across agents and time: a probit
random effect per (run, question), P(refuse | z) = Phi(a + sigma z), z ~ N(0, 1),
a = Phi^-1(p) sqrt(1 + sigma^2) so that the marginal is p. sigma by maximum likelihood.

Writes data/refusal_context.json:
  {"p": {class: [p(n_other=0), p(1), p(2+)]}, "n": {...}, "gate_none": {k: [p(pos 1), ...]},
   "sigma": s}
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import optimize, stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')
CLASSES = ('k3', 'k1', 'none')


def classes():
    q = pd.read_parquet(os.path.join(DATA, 'nq_embedded_oai.parquet'), columns=['question']).question
    a3 = [v[0] for v in json.load(open(os.path.join(DATA, 'refusal_oai.json')))['idk']]
    a1 = json.load(open(os.path.join(DATA, 'refusal_oai_k1.json')))['idk']
    return {s: ('k3' if x3 else 'k1' if x1 else 'none') for s, x3, x1 in zip(q, a3, a1)}


def events(cls):
    """(run file, question, class, n_other, position of own firsthand fact, k, idk)."""
    for f in sorted(glob.glob(os.path.join(DATA, 'llm_gate_dumps', '*.jsonl'))):
        for line in open(f):
            e = json.loads(line)
            ctx = e['ctx']
            pos = next((i + 1 for i, c in enumerate(ctx) if c[0] and c[1]), None)
            if pos is None:
                continue                                  # no firsthand fact: IDK by construction
            n_other = min(2, sum(1 for c in ctx if not c[0]))
            yield f, e['q'], cls[e['q']], n_other, pos, len(ctx), int(e['idk'])


def fit_sigma(groups):
    """groups: list of (array of p_e, array of y_e) per (run, question)."""
    x, w = np.polynomial.hermite_e.hermegauss(40)
    w = w / w.sum()

    def nll(s):
        tot = 0.0
        for p, y in groups:
            a = stats.norm.ppf(np.clip(p, 1e-6, 1 - 1e-6)) * np.sqrt(1 + s * s)
            pr = stats.norm.cdf(a[:, None] + s * x[None, :])
            lk = np.prod(np.where(y[:, None] == 1, pr, 1 - pr), axis=0)
            tot -= np.log(max(np.dot(w, lk), 1e-300))
        return tot
    return float(optimize.minimize_scalar(nll, bounds=(0.0, 5.0), method='bounded').x)


def main():
    cls = classes()
    acc = defaultdict(lambda: [0, 0])
    gate = defaultdict(lambda: [0, 0])
    ev = list(events(cls))
    for f, q, c, n_other, pos, k, y in ev:
        if c == 'none':
            a = gate[(k, pos)]
        else:
            a = acc[(c, n_other)]
        a[0] += y; a[1] += 1
    p = {c: [acc[(c, j)][0] / max(acc[(c, j)][1], 1) for j in range(3)] for c in ('k3', 'k1')}
    n = {c: [acc[(c, j)][1] for j in range(3)] for c in ('k3', 'k1')}
    groups = defaultdict(lambda: ([], []))
    for f, q, c, n_other, pos, k, y in ev:
        if c != 'none':
            g = groups[(f, q)]
            g[0].append(p[c][n_other]); g[1].append(y)
    sigma = fit_sigma([(np.array(a), np.array(b)) for a, b in groups.values()])
    out = {'p': p, 'n': n,
           'gate_none': {str(k): [gate[(k, j)][0] / gate[(k, j)][1] for j in range(1, k + 1) if gate[(k, j)][1]]
                         for k in sorted({k for k, _ in gate})},
           'sigma': sigma}
    for c in p:
        print(c, 'p(n_other = 0, 1, 2+) =', np.round(p[c], 3), 'n =', n[c])
    print('gate (none):', {k: np.round(v, 4).tolist() for k, v in out['gate_none'].items()}, ' sigma = %.2f' % sigma)
    json.dump(out, open(os.path.join(DATA, 'refusal_context.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
