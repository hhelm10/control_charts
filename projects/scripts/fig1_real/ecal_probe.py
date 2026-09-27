"""E-cal: estimate P(IDK | answering state) for the real agent.

The answering state is the context composition: whether the probed question's
QA pair is present (and at which position), and how related the distractor
entries are. No timestamps reach the prompt, so IDK cannot depend on age.
Estimated per context size k in {1, 3, 7} over N_Q probe questions with
resampled distractors (which also tests distractor exchangeability).

Output: ecal_table.json, one 4-outcome distribution per state:
  {IDK, faithful (repeats the context entry), parametric (gives the true answer
  not supplied by context), other (wrong/near-dup answer)} for
  present / present_wrong (planted incorrect answer) / absent states.
The planted-wrong states separate faithful from parametric behavior -- i.e.
whether the model propagates stored staleness or silently corrects it.
Cost: ~1,900 gpt-4o-mini calls (~$0.20). Requires OPENAI_API_KEY.
"""
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
from controlcharts.config import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from controlcharts.agent import _hedged_chat

PARQUET = "/home/ubuntu/helivan-chat-a100/projects/data/nq_embedded.parquet"
MODEL = "gpt-4o-mini"
N_Q = 40
KS = (1, 3, 7)
# Bucket grid spans the k-th-best off-target similarity range for every pool
# size we simulate (M = 50 -> ~0.45-0.55 bulk; M = 2000 -> ~0.55-0.70; near-dup
# tail up to ~0.95). The table is bucket-parameterized and hence M-agnostic; a
# world of size M enters through its bucket-occupancy profile (m_profile.json).
SIM_BUCKETS = {"b1": (0.30, 0.42), "b2": (0.42, 0.50), "b3": (0.50, 0.58),
               "b4": (0.58, 0.66), "b5": (0.66, 0.78), "dup": (0.78, 0.95)}
PRESENT_DISTRACTOR_LEVELS = ("b3", "b5")   # typical for small-M and large-M worlds
IDK_PAT = ("i don't know", "i do not know", "i don’t know")


def build_conditions(df, emb, rng):
    """Yield (cond_key, question, gold_answer, context_pairs)."""
    S = emb @ emb.T
    probes = rng.choice(len(df), size=N_Q, replace=False)
    for qi in probes:
        q, gold = df.question[qi], df.answer[qi]
        sims = S[qi]

        def draw(lo, hi, n):
            pool = [j for j in np.argsort(-sims) if j != qi and lo <= sims[j] < hi]
            return list(rng.choice(pool, size=min(n, len(pool)), replace=False))

        wrong = df.answer[int(rng.integers(len(df)))]      # planted incorrect answer
        for k in KS:
            # present at each position x distractor level, true and planted-wrong variants
            for lvl in PRESENT_DISTRACTOR_LEVELS:
                for pos in range(k):
                    ds = draw(*SIM_BUCKETS[lvl], k - 1)
                    if len(ds) < k - 1:
                        continue
                    pairs = [(df.question[j], df.answer[j]) for j in ds]
                    for tag, planted in (("present", gold), ("present_wrong", wrong)):
                        ctx = pairs[:pos] + [(q, planted)] + pairs[pos:]
                        yield ((tag, k, pos + 1, lvl), q, gold, planted, ctx)
            # absent, per distractor bucket
            for bname, (lo, hi) in SIM_BUCKETS.items():
                ds = draw(lo, hi, k)
                if len(ds) < k:
                    continue
                yield (("absent", k, bname), q, gold, None,
                       [(df.question[j], df.answer[j]) for j in ds])


async def run_all(conds):
    sem = asyncio.Semaphore(50)

    async def one(cond):
        key, q, gold, planted, pairs = cond
        context = "\n\n".join(f"Q: {cq}\nA: {ca}" for cq, ca in pairs)
        prompt = DEFAULT_PROMPT_TEMPLATE.format(retrieved_context=context, question=q)
        async with sem:
            r = await _hedged_chat(MODEL, [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}])
        low = r.lower()
        hit = lambda s: s is not None and (s.lower()[:60] in low or low in s.lower())
        if any(p in low for p in IDK_PAT):
            cat = "idk"
        elif hit(planted) and (planted == gold or not hit(gold)):
            cat = "faithful"          # repeats what the context supplied
        elif hit(gold):
            cat = "parametric"        # true answer the context did not supply
        else:
            cat = "other"             # wrong / near-dup answer
        return key, cat

    return await asyncio.gather(*[one(c) for c in conds])


def save_m_profile(emb, rng):
    """Bucket occupancy of the j-th best competitor as a function of pool size M
    (free, embeddings only). Used to map a simulated world of size M onto the
    bucket-parameterized table."""
    prof = {}
    edges = {k: v for k, v in SIM_BUCKETS.items()}
    for M in (50, 100, 200, 400, 500, 1000, 2000):
        counts = {j: {b: 0 for b in edges} for j in range(1, 8)}
        n = 0
        for _ in range(15):
            pool = rng.choice(emb.shape[0], size=M, replace=False)
            S = emb[pool] @ emb[pool].T
            np.fill_diagonal(S, -1)
            srt = np.sort(S, axis=1)[:, ::-1]
            for row in srt[rng.choice(M, size=min(100, M), replace=False)]:
                n += 1
                for j in range(1, 8):
                    for b, (lo, hi) in edges.items():
                        if lo <= row[j - 1] < hi:
                            counts[j][b] += 1
        prof[M] = {str(j): {b: c / n for b, c in counts[j].items()} for j in counts}
    out = Path(__file__).parents[2] / "data" / "m_profile.json"
    json.dump(prof, open(out, "w"), indent=1)
    print("saved", out)


def main():
    df = pd.read_parquet(PARQUET).head(3000).reset_index()
    emb = np.vstack(df.embedding.values).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    save_m_profile(emb, rng)
    conds = list(build_conditions(df, emb, rng))
    print(f"{len(conds)} probe calls")
    results = asyncio.new_event_loop().run_until_complete(run_all(conds))

    CATS = ("idk", "faithful", "parametric", "other")
    agg = defaultdict(lambda: {c: 0 for c in CATS})
    for key, cat in results:
        agg[key][cat] += 1
    table = {}
    print(f"\n{'state':>32} " + "".join(f"{c:>11}" for c in CATS) + f" {'n':>5}")
    for key in sorted(agg, key=str):
        a = agg[key]; n = sum(a.values())
        table[str(key)] = {**{f"p_{c}": a[c] / n for c in CATS}, "n": n}
        print(f"{str(key):>32} " + "".join(f"{a[c]/n:>11.2f}" for c in CATS) + f" {n:>5}")
    out = Path(__file__).parents[2] / "data" / "ecal_table.json"
    json.dump(table, open(out, "w"), indent=1)
    print("\nsaved", out)


if __name__ == "__main__":
    main()
