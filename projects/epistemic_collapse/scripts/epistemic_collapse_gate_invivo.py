"""In-vivo IDK gate: P(LLM says IDK | retrieved-context structure), from the real
Fig 1(a) runs logged with LLM_DUMP (data/llm_gate_dumps/*.jsonl).

Each answering event records the top-k context: per entry (same question?,
firsthand?, revision tag, insertion time). Under the firsthand policy the reply
is the first firsthand same-question entry (pos), else IDK without asking.
Questions with measured refusal (refusal_oai.json, context A) are excluded:
refusal is modelled separately.

Cells: (pos, conflict) where conflict = another same-question entry in the
context carries a different revision than the relayed one.
Writes data/gate_invivo.json: {"pos,conflict": [p_idk, n]}.

    python epistemic_collapse_gate_invivo.py [k]
k = 3 (default): the Fig 1(a) runs f1real-*. Other k: the logged runs gate-k{k}-*
(Fig 1(a) setup with retrieval_k = k), refusable questions taken from
refusal_oai_k{k}.json; writes data/gate_invivo_k{k}.json.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', 'data')


K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SUFFIX = '' if K == 3 else f'_k{K}'


def events():
    df = pd.read_parquet(os.path.join(DATA, 'nq_embedded_oai.parquet'), columns=['question'])
    ref = json.load(open(os.path.join(DATA, f'refusal_oai{SUFFIX}.json')))['idk']
    refused = {q for q, v in zip(df.question, ref) if (v[0] if isinstance(v, list) else v) == 1}
    pattern = 'f1real-*.jsonl' if K == 3 else f'gate-k{K}-*.jsonl'
    for f in sorted(glob.glob(os.path.join(DATA, 'llm_gate_dumps', pattern))):
        for line in open(f):
            e = json.loads(line)
            if e['q'] in refused:
                continue
            same = [(i + 1, fh, rv) for i, (sq, fh, rv, _) in enumerate(e['ctx']) if sq]
            fh1 = next(((p, rv) for p, fh, rv in same if fh), None)
            if fh1 is None:
                continue                       # no firsthand entry: IDK by construction (no LLM decision)
            pos, rv = fh1
            conflict = any(r != rv for _, _, r in same)
            yield pos, conflict, len(same), e['t'], e['idk']


def table():
    acc = defaultdict(lambda: [0, 0])
    for pos, conflict, n_same, t, idk in events():
        a = acc[f'{pos},{int(conflict)}']
        a[0] += idk; a[1] += 1
    return {k: [s / n, n] for k, (s, n) in sorted(acc.items())}


if __name__ == '__main__':
    tab = table()
    for k, (p, n) in tab.items():
        print(f'pos,conflict={k}: P(IDK) = {p:.3f}  (n = {n})')
    json.dump(tab, open(os.path.join(DATA, f'gate_invivo{SUFFIX}.json'), 'w'), indent=1)
