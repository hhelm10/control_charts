"""Measure, for every question in the NQ pool, whether the real agent (gpt-4o-mini,
the repo's prompt, temperature 0) refuses to answer it even when its own fact is
retrieved at position 1.

Two retrieval contexts per question, each = [own fact, 2 neighbours by similarity]:
  A  neighbours = the 2 most similar questions in the whole pool
  B  neighbours = the 2 most similar within a random 400-question environment
     (B only for the first B_LIMIT questions: refusal is a property of the
     question, so B serves to estimate its consistency across contexts)
Output data/refusal_oai.json: {"idk": [[A, B], ...] (pool order; 1 = IDK),
"usage": {...}}. Resumable; checkpoints every batch.

    python epistemic_collapse_measure_refusal.py [n_questions]
"""
import asyncio
import json
import os
import random
import sys

import numpy as np
import pandas as pd
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'src'))
from controlcharts.config import DEFAULT_PROMPT_TEMPLATE, DEFAULT_SYSTEM_PROMPT  # noqa: E402

for _line in open(os.path.join(ROOT, '.env')):              # repo-level .env (never printed)
    _k, _, _v = _line.strip().partition('=')
    if _k and not _k.startswith('#') and _v:
        os.environ.setdefault(_k.removeprefix('export ').strip(), _v.strip().strip('"\''))
DATA = os.path.join(HERE, '..', 'data')
OUT = os.path.join(DATA, 'refusal_oai.json')
MODEL, CONC, BATCH, ENV_M, B_LIMIT = 'gpt-4o-mini', 200, 2000, 400, 5000
TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def is_idk(text):
    low = text.lower()
    return int("i don't know" in low or "i do not know" in low or "i don’t know" in low)


def contexts(df, E):
    """per question: ([A neighbour ids], [B neighbour ids])."""
    rng = np.random.default_rng(0)
    n = len(E)
    out = []
    for s in range(0, n, 1000):                       # A: whole-pool nearest neighbours
        S = E[s:s + 1000] @ E.T
        for r, i in enumerate(range(s, min(s + 1000, n))):
            S[r, i] = -9
        out.extend(np.argsort(-S, axis=1)[:, :2].tolist())
    B = []
    for i in range(n):                                # B: nearest within a random environment
        env = rng.choice(n, size=ENV_M, replace=False)
        env = env[env != i][:ENV_M - 1]
        sims = E[env] @ E[i]
        B.append(env[np.argsort(-sims)[:2]].tolist())
    return list(zip(out, B))


def prompt(df, i, nb):
    ctx = '\n\n'.join(f"Q: {df.question[j]}\nA: {df.answer[j]}" for j in [i] + list(nb))
    return [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": DEFAULT_PROMPT_TEMPLATE.format(retrieved_context=ctx,
                                                                       question=df.question[i])}]


async def call(client, sem, msgs, usage):
    async with sem:
        for attempt in range(12):
            try:
                r = await client.chat.completions.create(model=MODEL, messages=msgs,
                                                         temperature=0.0, max_tokens=500)
                usage['prompt_tokens'] += r.usage.prompt_tokens
                usage['completion_tokens'] += r.usage.completion_tokens
                usage['calls'] += 1
                return is_idk(r.choices[0].message.content or '')
            except TRANSIENT:
                await asyncio.sleep(min(60, 2 ** attempt) * (0.5 + random.random()))
        raise RuntimeError('call failed after retries')


async def main(n_q):
    df = pd.read_parquet(os.path.join(DATA, 'nq_embedded_oai.parquet'))
    E = np.stack(df.embedding.values).astype(np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    ctx = contexts(df, E)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {
        'idk': [None] * len(df), 'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0}}
    todo = [i for i in range(min(n_q, len(df))) if res['idk'][i] is None]
    sem = asyncio.Semaphore(CONC)
    for s in range(0, len(todo), BATCH):
        client = AsyncOpenAI(timeout=30.0, max_retries=2)   # fresh client per batch (long-lived ones degraded)
        ids = todo[s:s + BATCH]
        jobs = [(i, call(client, sem, prompt(df, i, nb), res['usage']))
                for i in ids for c, nb in enumerate(ctx[i]) if c == 0 or i < B_LIMIT]
        out = await asyncio.gather(*[j for _, j in jobs])
        got = {}
        for (i, _), v in zip(jobs, out):
            got.setdefault(i, []).append(v)
        for i in ids:
            res['idk'][i] = got[i] + [None] * (2 - len(got[i]))
        await client.close()
        json.dump(res, open(OUT, 'w'))
        u = res['usage']
        cost = u['prompt_tokens'] * 0.15e-6 + u['completion_tokens'] * 0.6e-6
        print(f"{s + len(ids)}/{len(todo)} questions, {u['calls']} calls, ${cost:.2f}", flush=True)


if __name__ == '__main__':
    asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 10 ** 9))
