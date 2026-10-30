"""Per-k refusal: like epistemic_collapse_measure_refusal.py (context A only), but the
retrieved context holds k entries: the question's own fact plus its k-1 nearest
pool neighbours. Refusal turned out to depend on k (in vivo, about 3% of questions
are refused at k=1 vs about 1.5% at k=3), not on how similar the neighbours are.

Output data/refusal_oai_k{k}.json: {"idk": [0/1 per pool question], "usage": {...}}.
Resumable.

    python epistemic_collapse_measure_refusal_k.py K [n_questions]
"""
import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd
from openai import AsyncOpenAI

import epistemic_collapse_measure_refusal as base

K = int(sys.argv[1])
OUT = os.path.join(base.DATA, f'refusal_oai_k{K}.json')


def neighbours(E, n_nb):
    out = []
    for s in range(0, len(E), 1000):
        S = E[s:s + 1000] @ E.T
        for r, i in enumerate(range(s, min(s + 1000, len(E)))):
            S[r, i] = -9
        out.extend(np.argsort(-S, axis=1)[:, :n_nb].tolist())
    return out


async def main(n_q):
    df = pd.read_parquet(os.path.join(base.DATA, 'nq_embedded_oai.parquet'))
    E = np.stack(df.embedding.values).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    nb = neighbours(E, K - 1)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'idk': [None] * len(df), 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0}}
    todo = [i for i in range(min(n_q, len(df))) if res['idk'][i] is None]
    sem = asyncio.Semaphore(base.CONC)
    for s in range(0, len(todo), base.BATCH):
        client = AsyncOpenAI(timeout=30.0, max_retries=2)
        ids = todo[s:s + base.BATCH]
        out = await asyncio.gather(*[base.call(client, sem, base.prompt(df, i, nb[i]), res['usage'])
                                     for i in ids])
        for i, v in zip(ids, out):
            res['idk'][i] = v
        await client.close()
        json.dump(res, open(OUT, 'w'))
        u = res['usage']
        cost = u['prompt_tokens'] * 0.15e-6 + u['completion_tokens'] * 0.6e-6
        print(f"k={K}: {s + len(ids)}/{len(todo)} questions, {u['calls']} calls, ${cost:.2f}", flush=True)


if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[2]) if len(sys.argv) > 2 else 10 ** 9))
