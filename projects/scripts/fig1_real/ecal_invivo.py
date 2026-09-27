"""E-cal v2: probe the real LLM on IN-VIVO retrieval states harvested from
surrogate runs (ECAL_DUMP), instead of synthetic bucket-sampled contexts.

Samples late-run states for static questions: 'absent' states (exact match
crowded out -- where the surrogate says IDK and the real system apparently
often answers correctly) plus 'present' controls. Responses are classified
correct/wrong by nearest-neighbor among the pool's answer embeddings.

~250 calls (~$0.03). Requires OPENAI_API_KEY.
"""
import asyncio
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
from controlcharts.config import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from controlcharts.agent import _hedged_chat
from controlcharts.temporal_questions import TEMPORAL_QUESTIONS

DUMP_DIR = "/tmp/claude-1000/-home-ubuntu-helivan-chat-a100-projects-epistemic-collapse/bc7fbc71-88d8-4252-958c-93c08b0f83dc/scratchpad"
PARQUET = "/home/ubuntu/helivan-chat-a100/projects/data/nq_embedded.parquet"
MODEL = "gpt-4o-mini"
N_ABSENT, N_PRESENT = 200, 50
IDK_PAT = ("i don't know", "i do not know", "i don’t know")


def harvest(rng):
    temporal = set(TEMPORAL_QUESTIONS)
    absent, present = [], []
    for f in glob.glob(f"{DUMP_DIR}/dump-*.jsonl"):
        setting = Path(f).stem.split("-", 1)[1]
        for line in open(f):
            d = json.loads(line)
            if not (int(os.environ.get("TMIN", 500)) <= d["t"] <= int(os.environ.get("TMAX", 10**9))) or d["q"] in temporal or not d["ctx"]:
                continue
            d["setting"] = setting
            (present if d["present"] else absent).append(d)
    rng.shuffle(absent); rng.shuffle(present)
    print(f"harvested {len(absent)} absent / {len(present)} present states")
    return absent[:N_ABSENT], present[:N_PRESENT]


async def probe(states):
    sem = asyncio.Semaphore(50)

    async def one(d):
        ctx = "\n\n".join(f"Q: {q}\nA: {a}" for q, a in d["ctx"])
        prompt = DEFAULT_PROMPT_TEMPLATE.format(retrieved_context=ctx, question=d["q"])
        async with sem:
            r = await _hedged_chat(MODEL, [
                {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}])
        return d, r

    return await asyncio.gather(*[one(d) for d in states])


def main():
    rng = np.random.default_rng(0)
    absent, present = harvest(rng)
    results = asyncio.new_event_loop().run_until_complete(probe(absent + present))

    df = pd.read_parquet(PARQUET)
    gold = {q: a for q, a in zip(df.question, df.answer)}
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    uniq = sorted({r for _, r in results if not any(p in r.lower() for p in IDK_PAT)})
    remb = model.encode(uniq, batch_size=128, convert_to_numpy=True).astype(np.float32)
    remb /= np.linalg.norm(remb, axis=1, keepdims=True)
    aemb = model.encode(df.answer.tolist(), batch_size=256, convert_to_numpy=True).astype(np.float32)
    aemb /= np.linalg.norm(aemb, axis=1, keepdims=True)
    nn = (remb @ aemb.T).argmax(axis=1)
    ridx = {r: i for i, r in enumerate(uniq)}
    aidx = {a: i for i, a in enumerate(df.answer)}

    agg = defaultdict(lambda: defaultdict(int))
    rows = []
    for d, r in results:
        low = r.lower()
        if any(p in low for p in IDK_PAT):
            cat = "idk"
        else:
            g = gold.get(d["q"])
            cat = "correct" if g is not None and nn[ridx[r]] == aidx[g] else "wrong"
        key = ("present" if d["present"] else "absent", d["setting"])
        agg[key][cat] += 1
        rows.append({**{k: d[k] for k in ("t", "q", "present", "setting")},
                     "ctx": d["ctx"], "response": r, "cat": cat})
    tag = os.environ.get("TMIN", "500")
    json.dump(rows, open(Path(__file__).parents[2] / "data" / f"ecal_invivo_results_{tag}.json", "w"), indent=1)

    print(f"\n{'state / setting':>22} {'idk':>6} {'correct':>8} {'wrong':>6} {'n':>4}")
    for key in sorted(agg):
        a = agg[key]; n = sum(a.values())
        print(f"{str(key):>22} {a['idk']/n:>6.2f} {a['correct']/n:>8.2f} {a['wrong']/n:>6.2f} {n:>4}")


if __name__ == "__main__":
    main()
