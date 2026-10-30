"""Build the large question pool with OpenAI embeddings (replaces the
nomic-embed-text pool, which needed the Modal GPU setup).

Pool: the first POOL distinct questions of NQ (sentence-transformers/
natural-questions, train), so the old 3000-question pool is its prefix.
Questions and answers are embedded with text-embedding-3-small at 768
dimensions (same shape as before).

Outputs (the old files are left untouched):
  data/nq_embedded_oai.parquet   question, answer, embedding
  data/answer_alphabet_oai.npz   the verbatim-relay alphabet: every answer,
                                 "I don't know", the quine, integers 0..2000
Cost: ~3.6M tokens at $0.02/M, about $0.07. Needs OPENAI_API_KEY.
"""
import os
import time

import numpy as np
import pandas as pd
from datasets import load_dataset
from openai import OpenAI

import epistemic_collapse_common as ec

POOL = 25_000
MODEL, DIM = 'text-embedding-3-small', 768
BATCH = 500
OUT = f'{ec.DATA}/nq_embedded_oai.parquet'
ALPHA = f'{ec.DATA}/answer_alphabet_oai.npz'
EXTRA = ["I don't know", 'I lost the game'] + [str(i) for i in range(2001)]


def embed(client, texts):
    out = []
    for i in range(0, len(texts), BATCH):
        for attempt in range(8):
            try:
                r = client.embeddings.create(model=MODEL, input=texts[i:i + BATCH], dimensions=DIM)
                break
            except Exception as e:                    # transient API errors
                if attempt == 7:
                    raise
                print('retry', i, type(e).__name__); time.sleep(2 ** attempt)
        out += [d.embedding for d in r.data]
        print(f'  {i + BATCH}/{len(texts)}', flush=True)
    return np.asarray(out, dtype=np.float32)


def main():
    ds = load_dataset('sentence-transformers/natural-questions', split='train')
    df = pd.DataFrame({'question': ds['query'], 'answer': ds['answer']})
    df = df.drop_duplicates('question').head(POOL).reset_index(drop=True)
    client = OpenAI()
    print('questions'); q = embed(client, df.question.tolist())
    answers = list(dict.fromkeys(df.answer))           # unique, in order
    texts = answers + [t for t in EXTRA if t not in set(answers)]
    print('alphabet'); a = embed(client, texts)
    df['embedding'] = list(q)
    df.to_parquet(OUT, index=False)
    np.savez(ALPHA, texts=np.array(texts, dtype=object), embeddings=a)
    print(f'wrote {len(df)} questions, {len(texts)} alphabet strings')


if __name__ == '__main__':
    main()
