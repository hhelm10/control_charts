"""sim_mirror: a fast, vectorized reimplementation of the repo pipeline's
mechanics with the measured answering gate (i.e. the proxy), for sweeps.

Mirrors src/controlcharts (cli.run + Simulation._run_step + Agent + QADatabase):
  - questions in play: M sampled from the NQ pool; cosine similarity of their
    normalized embeddings (faiss IndexFlatIP on L2-normalized vectors)
  - per-question revision rates lambda_q ~ lognormal(mean lam, dispersion 1)
  - initial knowledge: agent i holds questions shuffled[i*q0:(i+1)*q0] (firsthand, t=0)
  - each step: revisions; E environment queries per agent, each written to memory
    immediately (firsthand, value = current truth); then K asks per agent to
    uniformly random neighbors (Erdos-Renyi graph), all answered on the post-
    observation state; non-IDK replies are written (secondhand under 'firsthand')
  - question selection (queries and asks): never-learned -> weight 1; learned ->
    candidate w.p. p = 1 - e^{-c age}, weight p (age since last acquisition)
  - retrieval: top-1000 by raw similarity, re-ranked by sim * e^{-c (t - t_ins)}, top k
  - answer: the first exact-question entry (firsthand under 'firsthand' policy)
    among the top k answers with prob 1 - P(IDK | k, position) (measured gate); else IDK
  - probe: a fixed sample of `probe` questions, every `interval` steps, all agents

Returns per-snapshot agent IDK, correct, and collapse proportion (all agents IDK).
"""
from __future__ import annotations

import ast
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
_EMB = None
_GATE = None


def _embeddings():
    global _EMB
    if _EMB is None:
        df = pd.read_parquet(os.path.join(DATA, "nq_embedded.parquet"))
        E = np.stack(df["embedding"].values).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        _EMB = E
    return _EMB


def _gate():
    global _GATE
    if _GATE is None:
        acc = defaultdict(lambda: [0.0, 0])
        for key, cell in json.load(open(os.path.join(DATA, "ecal_table.json"))).items():
            tup = ast.literal_eval(key)
            if tup[0] == "present":
                a = acc[(tup[1], tup[2])]
                a[0] += cell["p_idk"] * cell["n"]; a[1] += cell["n"]
        _GATE = {kp: s / n for kp, (s, n) in acc.items() if n}
    return _GATE


def run(M=50, N=10, E=3, K=8, k=3, c=0.05, lam=0.01, lam_disp=1.0, edge_p=0.6667,
        q0=5, policy="firsthand", T=300, seed=0, interval=10, probe=20,
        warm_start=False, gate=True, candidate_k=1000,
        sim_const=None, select_rule="repo", hearsay_crowds_self=True):
    """Ablation switches (defaults = proxy mechanics):
      sim_const: replace off-target similarities by a constant (toy-style)
      select_rule: "repo" | "uniform" (questions chosen uniformly)
      hearsay_crowds_self: if False, secondhand copies of q do not compete with
        the firsthand copy of q (they get similarity sim_floor-like 0 for q)"""
    rng = np.random.default_rng(seed)
    Emb = _embeddings()
    idx = rng.choice(len(Emb), size=M, replace=False)
    S = Emb[idx] @ Emb[idx].T
    if sim_const is not None:
        S = np.full((M, M), sim_const, dtype=np.float32)
    np.fill_diagonal(S, 1.0)
    lam_q = lam * np.exp(lam_disp * rng.standard_normal(M) - lam_disp ** 2 / 2) if lam > 0 else np.zeros(M)
    rev = np.zeros(M, dtype=np.int64)
    G = _gate() if gate else {}

    # graph
    A = np.zeros((N, N), dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < edge_p:
                A[i, j] = A[j, i] = True
    nbrs = [np.flatnonzero(A[i]) for i in range(N)]

    cap = (E + K) * (T + 1) + M + 10
    mq = np.zeros((N, cap), dtype=np.int32)        # memory: question id
    mt = np.zeros((N, cap), dtype=np.int32)        # insertion time
    mf = np.zeros((N, cap), dtype=bool)            # firsthand
    mv = np.zeros((N, cap), dtype=np.int64)        # stored value (revision)
    n_mem = np.zeros(N, dtype=np.int64)
    known = np.zeros((N, M), dtype=bool)
    learn = np.zeros((N, M), dtype=np.int64)

    def insert(i, q, t, fh, v):
        n = n_mem[i]
        mq[i, n] = q; mt[i, n] = t; mf[i, n] = fh; mv[i, n] = v
        n_mem[i] = n + 1
        known[i, q] = True; learn[i, q] = t

    order = rng.permutation(M)
    if warm_start:
        for i in range(N):
            for q in range(M):
                insert(i, q, 0, True, 0)
    else:
        for i in range(N):
            for q in order[i * q0:(i + 1) * q0]:
                insert(i, q, 0, True, 0)

    def select(i, t):
        if select_rule == "uniform":
            return int(rng.integers(M))
        p = 1.0 - np.exp(-c * (t - learn[i]))
        w = np.where(known[i], np.where(rng.random(M) < p, p, 0.0), 1.0)
        tot = w.sum()
        if tot <= 0:
            return None
        return int(rng.choice(M, p=w / tot))

    def answer(i, q, t):
        """returns stored value, or -1 for IDK."""
        n = n_mem[i]
        if n == 0:
            return -1
        qs = mq[i, :n]
        raw = S[q, qs]
        if not hearsay_crowds_self and policy == "firsthand":
            raw = np.where((qs == q) & ~mf[i, :n], 0.0, raw)
        if n > candidate_k:
            cand = np.argpartition(-raw, candidate_k)[:candidate_k]
        else:
            cand = np.arange(n)
        score = raw[cand] * np.exp(-c * (t - mt[i, cand]))
        kk = min(k, len(cand))
        top = cand[np.argsort(-score, kind="stable")[:kk]]
        for pos, e in enumerate(top, start=1):
            if mq[i, e] == q and (policy != "firsthand" or mf[i, e]):
                p_idk = G.get((kk, pos), 0.0)
                if p_idk > 0 and rng.random() < p_idk:
                    return -1
                return int(mv[i, e])
        return -1

    probe_q = rng.choice(M, size=min(probe, M), replace=False)
    out = {"t": [], "agent_idk": [], "correct": [], "collapse": [], "misalign": [], "stale": []}

    def snapshot(t):
        R = np.array([[answer(i, q, t) for q in probe_q] for i in range(N)])
        idk = R < 0
        corr = (R == rev[probe_q][None, :])
        out["t"].append(t); out["agent_idk"].append(float(idk.mean()))
        out["correct"].append(float(corr.mean())); out["collapse"].append(float(idk.all(axis=0).mean()))
        out["stale"].append(float((~idk & ~corr).mean()))
        # misalignment: no agent correct, but at least one answers (stale)
        out["misalign"].append(float((~corr.any(axis=0) & ~idk.all(axis=0)).mean()))

    for t in range(T):
        rev += rng.random(M) < lam_q
        for i in range(N):                                  # environment queries
            for _ in range(E):
                q = select(i, t)
                if q is not None:
                    insert(i, q, t, True, rev[q])
        asks = []
        for i in range(N):                                  # select all asks first
            for _ in range(K):
                q = select(i, t)
                if q is None or len(nbrs[i]) == 0:
                    continue
                asks.append((i, int(rng.choice(nbrs[i])), q))
        replies = [answer(j, q, t) for i, j, q in asks]     # answered on the same state
        for (i, j, q), v in zip(asks, replies):
            if v >= 0:
                insert(i, q, t, False, v)
        if t % interval == 0:
            snapshot(t)
    return {k_: np.array(v) for k_, v in out.items()}
