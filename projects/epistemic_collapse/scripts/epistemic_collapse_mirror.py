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
    among the top k answers with prob 1 - P(IDK | k, position) (measured gate, see _gate); else IDK
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
# question pool (default oai): EC_POOL=oai -> 25k NQ, OpenAI text-embedding-3-small (768-d);
# EC_POOL=nomic -> the original 3k pool (nomic-embed-text-v1.5)
POOL = {"oai": "nq_embedded_oai.parquet", "nomic": "nq_embedded.parquet"}[os.environ.get("EC_POOL", "oai")]
_EMB = None
_GATE = None
_REF = {}
_RC = None
NOTHER_STATS = None        # set to [0, 0, 0] to count answering events by n_other (diagnostics)
# per-answer refusal of a refusable question by a holding agent, measured in vivo on
# the real Fig 1(a) runs (data/llm_gate_dumps; n = 550 answering events, flat in t)
RHO_REF = 0.60
# "run" mode: a refusable question is refused for the whole run w.p. RHO_REF / RHO_IN,
# then by each holding agent w.p. RHO_IN (measured on the real Fig 1 warm-start runs
# at t = 0: 74 of 80 answers). Matches the real runs' within-run correlation (share of
# refusable questions that all 10 agents refuse, over t) far better than "context".
RHO_IN = 0.925
# per-k (RHO_REF, RHO_IN) where measured differently. k = 1: the context is the own
# fact alone, so a refusable question is refused (nearly) always: 99.5% of 2205
# in-vivo answering events (real Fig 1(a) setup with k = 1, data/llm_gate_dumps/gate-k1-*).
REF_RHO = {1: (0.995, 0.995)}


def refusal_table(k=3):
    """Refusable questions, measured on the real agent (gpt-4o-mini, repo prompt,
    own fact retrieved at position 1, its k-1 nearest pool neighbours as context;
    scripts/epistemic_collapse_measure_refusal.py for k = 3, ..._refusal_k.py for
    other k; k without its own measurement falls back to k = 3). Refusal depends
    on k: ~4.4% of NQ at k = 1, ~1.5% at k = 3 and 7. For these the
    stored passage does not contain the answer; revisions keep the passage, so
    the property is permanent. Returns (p_ref per pool question, summary):
    p_ref = RHO_REF for refusable questions, 0 otherwise (other questions' residual
    IDK is in the in-vivo gate, which excludes refusable ones). u = fraction
    refusable; rho_in = per-agent refusal once a question is refused in a run.
    Zeros for the nomic pool."""
    if k not in _REF:
        fk = os.path.join(DATA, f"refusal_oai_k{k}.json")
        f = fk if k != 3 and os.path.exists(fk) else os.path.join(DATA, "refusal_oai.json")
        rho, rho_in = REF_RHO.get(k, (RHO_REF, RHO_IN)) if f == fk else (RHO_REF, RHO_IN)
        n = len(_embeddings())
        if POOL != "nq_embedded_oai.parquet" or not os.path.exists(f):
            _REF[k] = (np.zeros(n), {"u": 0.0, "rho": 0.0, "rho_in": RHO_IN})
        else:
            A = np.array([v[0] if isinstance(v, list) else v for v in json.load(open(f))["idk"]], dtype=float)
            _REF[k] = (np.where(A == 1, rho, 0.0), {"u": float(A.mean()), "rho": rho, "rho_in": rho_in})
    return _REF[k]


def refusal_context():
    """Context-dependent refusal (scripts/epistemic_collapse_refusal_context.py,
    data/refusal_context.json): per pool question its class (0 none, 1 refused only
    with its own fact alone, 2 refused with 2 nearest neighbours), the in-vivo
    P(refuse | class, n_other = 0, 1, 2+ retrieved entries about other questions),
    the residual IDK of "none" questions by k and position, and sigma (probit
    random effect per run and question)."""
    global _RC
    if _RC is None:
        rc = json.load(open(os.path.join(DATA, "refusal_context.json")))
        a3 = np.array([v[0] for v in json.load(open(os.path.join(DATA, "refusal_oai.json")))["idk"]])
        a1 = np.array(json.load(open(os.path.join(DATA, "refusal_oai_k1.json")))["idk"])
        cls = np.where(a3 == 1, 2, np.where(a1 == 1, 1, 0))
        P = np.zeros((3, 3))
        P[1], P[2] = rc["p"]["k1"], rc["p"]["k3"]
        _RC = {"cls": cls, "P": P, "sigma": rc["sigma"],
               "gate": {int(k): v for k, v in rc["gate_none"].items()}}
    return _RC


def _embeddings():
    global _EMB
    if _EMB is None:
        df = pd.read_parquet(os.path.join(DATA, POOL))
        E = np.stack(df["embedding"].values).astype(np.float32)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        _EMB = E
    return _EMB


def _gate():
    """P(IDK | k, position of the firsthand entry). k = 3: measured in vivo on the
    real Fig 1(a) runs (data/gate_invivo.json; refused questions excluded; no
    dependence on conflicting versions, so pooled over it). k = 1: the same on
    logged k = 1 runs (data/gate_invivo_k1.json). Other k: the earlier
    synthetic-context table (data/ecal_table.json)."""
    global _GATE
    if _GATE is None:
        acc = defaultdict(lambda: [0.0, 0])
        for key, cell in json.load(open(os.path.join(DATA, "ecal_table.json"))).items():
            tup = ast.literal_eval(key)
            if tup[0] == "present":
                a = acc[(tup[1], tup[2])]
                a[0] += cell["p_idk"] * cell["n"]; a[1] += cell["n"]
        _GATE = {kp: s / n for kp, (s, n) in acc.items() if n}
        for kk, name in [(3, "gate_invivo.json"), (1, "gate_invivo_k1.json")]:
            f = os.path.join(DATA, name)
            if os.path.exists(f):
                iv = defaultdict(lambda: [0.0, 0])
                for key, (pr, n) in json.load(open(f)).items():
                    a = iv[int(key.split(",")[0])]
                    a[0] += pr * n; a[1] += n
                for pos, (s_, n) in iv.items():
                    _GATE[(kk, pos)] = s_ / n
    return _GATE


def gate_mean(k=3):
    """overall in-vivo P(IDK | holder), pooled over positions (theory's eps);
    k = 1 has its own measurement, other k use k = 3's."""
    f = os.path.join(DATA, "gate_invivo_k1.json" if k == 1 else "gate_invivo.json")
    if not os.path.exists(f):
        return 0.0
    v = json.load(open(f)).values()
    return sum(p * n for p, n in v) / sum(n for _, n in v)


def run(M=50, N=10, E=3, K=8, k=3, c=0.05, lam=0.01, lam_disp=1.0, edge_p=0.6667,
        q0=5, policy="firsthand", T=300, seed=0, interval=10, probe=20,
        warm_start=False, gate=True, candidate_k=1000,
        sim_const=None, select_rule="repo", hearsay_crowds_self=True,
        books=False, write_p=0.05, write_W=5, book_frac=1.0,
        refusal=True, refusal_mode="nother", debug=False,
        questions=None, probe_questions=None):
    """Ablation switches (defaults = proxy mechanics):
      sim_const: replace off-target similarities by a constant (toy-style)
      select_rule: "repo" | "uniform" (questions chosen uniformly)
      questions / probe_questions: pool indices of a real run's question set and
        probe set (matched comparison); default = drawn from the seed
      hearsay_crowds_self: if False, secondhand copies of q do not compete with
        the firsthand copy of q (they get similarity sim_floor-like 0 for q)

    Books (books=True): a shared book holds one entry per question; the last
    write wins and readers cannot see an entry's age.
      writing: each step an agent attempts a write w.p. write_p, spending its
        whole budget (no queries, no asks); on its write_W-th attempt it commits
        its current answer to every question it can answer, from memory. The
        write_W steps it spent buy write_W * (E + K) verifications: that many
        committed entries (random) are re-checked against the environment and
        written with the current truth (and enter the writer's memory as
        observations); the rest are copied unchecked and may be stale (slop).
      reading: each ask goes to the book w.p. book_frac (else to a peer); a
        non-empty entry is stored as grounded (answerable under 'firsthand').
      Expected write rate per agent: write_p / write_W. Snapshots add
      book_cov (fraction of questions with an entry) and book_slop (fraction
      of entries that are wrong now).

    Refusal (refusal=True; see refusal_table): a holding agent answering a
    refusable question says IDK w.p. RHO_REF on average. refusal_mode "run"
    (default): refused for the whole run w.p. RHO_REF / RHO_IN, then per agent
    w.p. RHO_IN; "context": one fixed draw per (run, question, retrieved
    context); "call": independent per answer. Drawn from a separate random stream,
    so all other randomness is unchanged.
    refusal_mode "nother" (default; see refusal_context): IDK w.p.
    P(class, n_other) with a per-(run, question) probit random effect; it replaces
    both the gate and the other refusal modes."""
    rng = np.random.default_rng(seed)
    Emb = _embeddings()
    idx = rng.choice(len(Emb), size=M, replace=False)
    if questions is not None:                      # matched run: a real run's pool indices
        idx = np.asarray(questions, dtype=np.int64)
        assert len(idx) == M
    S = Emb[idx] @ Emb[idx].T
    if sim_const is not None:
        S = np.full((M, M), sim_const, dtype=np.float32)
    np.fill_diagonal(S, 1.0)
    lam_q = lam * np.exp(lam_disp * rng.standard_normal(M) - lam_disp ** 2 / 2) if lam > 0 else np.zeros(M)
    rev = np.zeros(M, dtype=np.int64)
    nother = refusal_mode == "nother"
    G = _gate() if gate and not nother else {}
    rng_ref = np.random.default_rng([seed, 1])
    if nother:
        from scipy.stats import norm
        RC = refusal_context()
        cls_q = RC["cls"][idx] if refusal else np.zeros(M, dtype=np.int64)
        sg = RC["sigma"]
        z_q = rng_ref.standard_normal(M)                       # random effect per (run, question)
        a_tab = norm.ppf(np.clip(RC["P"], 1e-9, 1 - 1e-9)) * np.sqrt(1 + sg * sg)
        g_none = RC["gate"].get(k, RC["gate"][3]) if gate else []

        pr_q = norm.cdf(a_tab[cls_q] + sg * z_q[:, None])      # M x n_other (0, 1, 2+)

        def p_refuse(q, pos, n_oth):
            if cls_q[q] == 0:
                return g_none[min(pos, len(g_none)) - 1] if g_none else 0.0
            return pr_q[q, min(n_oth, 2)]
    p_ref = refusal_table(k)[0][idx] if refusal and not nother else np.zeros(M)
    rho_in = refusal_table(k)[1]["rho_in"]
    refused = rng_ref.random(M) < p_ref / rho_in if refusal_mode == "run" else np.zeros(M, dtype=bool)   # mean per-answer refusal = p_ref

    def refuses(q, ctx_q):
        """refusal decision for an answering event on q with retrieved questions ctx_q."""
        if refusal_mode == "run":
            return refused[q] and rng_ref.random() < rho_in
        if refusal_mode == "call":
            return rng_ref.random() < p_ref[q]
        # "context": a fixed draw per (run, question, retrieved context): identical
        # contexts refuse alike, different contexts independently
        u = (hash((seed, int(idx[q]), tuple(int(idx[x]) for x in ctx_q))) & 0xFFFFFFFF) / 2 ** 32
        return u < p_ref[q]

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
                if nother:
                    n_oth = int(np.sum(mq[i, top] != q))
                    if NOTHER_STATS is not None:
                        NOTHER_STATS[min(n_oth, 2)] += 1
                    if rng_ref.random() < p_refuse(q, pos, n_oth):
                        return -1
                    return int(mv[i, e])
                p_idk = G.get((kk, pos), 0.0)
                if p_idk > 0 and rng.random() < p_idk:
                    return -1
                if refusal and refuses(q, mq[i, top]):
                    return -1
                return int(mv[i, e])
        return -1

    probe_q = rng.choice(M, size=min(probe, M), replace=False)
    if probe_questions is not None:                # matched run: the real run's probe set
        where = {int(v): j for j, v in enumerate(idx)}
        probe_q = np.array([where[int(v)] for v in probe_questions], dtype=np.int64)
    out = {"t": [], "agent_idk": [], "correct": [], "collapse": [], "misalign": [], "stale": []}
    if books:
        out.update(book_cov=[], book_slop=[])
        book = np.full(M, -1, dtype=np.int64)       # entry value (revision), -1 = empty
        attempts = np.zeros(N, dtype=np.int64)

    def commit(i, t):
        qs = np.flatnonzero(known[i])
        vals = np.array([answer(i, q, t) for q in qs], dtype=np.int64)
        qs, vals = qs[vals >= 0], vals[vals >= 0]
        nv = min(len(qs), write_W * (E + K))
        ver = rng.choice(len(qs), size=nv, replace=False) if nv else np.array([], dtype=np.int64)
        for e in ver:                                  # verification = environment queries
            q = int(qs[e]); vals[e] = rev[q]
            insert(i, q, t, True, rev[q])
        book[qs] = vals

    def snapshot(t):
        R = np.array([[answer(i, q, t) for q in probe_q] for i in range(N)])
        if debug:
            out.setdefault("dbg", []).append((t, idx[probe_q].copy(), R.copy(), rev[probe_q].copy()))
        idk = R < 0
        corr = (R == rev[probe_q][None, :])
        out["t"].append(t); out["agent_idk"].append(float(idk.mean()))
        out["correct"].append(float(corr.mean())); out["collapse"].append(float(idk.all(axis=0).mean()))
        out["stale"].append(float((~idk & ~corr).mean()))
        # misalignment: no agent correct, but at least one answers (stale)
        out["misalign"].append(float((~corr.any(axis=0) & ~idk.all(axis=0)).mean()))
        if books:
            has = book >= 0
            out["book_cov"].append(float(has.mean()))
            out["book_slop"].append(float((book[has] != rev[has]).mean()) if has.any() else 0.0)

    for t in range(T):
        rev += rng.random(M) < lam_q
        writing = np.zeros(N, dtype=bool)
        if books:                                           # write attempts take the whole step
            writing = rng.random(N) < write_p
            for i in np.flatnonzero(writing):
                attempts[i] += 1
                if attempts[i] >= write_W:
                    attempts[i] = 0
                    commit(i, t)
        for i in range(N):                                  # environment queries
            if writing[i]:
                continue
            for _ in range(E):
                q = select(i, t)
                if q is not None:
                    insert(i, q, t, True, rev[q])
        asks = []
        for i in range(N):                                  # select all asks first
            if writing[i]:
                continue
            for _ in range(K):
                q = select(i, t)
                if q is None:
                    continue
                if books and rng.random() < book_frac:
                    asks.append((i, -1, q))                 # read the book
                    continue
                if len(nbrs[i]) == 0:
                    continue
                asks.append((i, int(rng.choice(nbrs[i])), q))
        replies = [int(book[q]) if j < 0 else answer(j, q, t) for i, j, q in asks]   # same state
        for (i, j, q), v in zip(asks, replies):
            if v >= 0:
                insert(i, q, t, j < 0, v)                   # book entries are grounded
        if t % interval == 0:
            snapshot(t)
    return {k_: (v if k_ == "dbg" else np.array(v)) for k_, v in out.items()}
