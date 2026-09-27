"""Toy simulator v4: crowding retrieval -- forgetting is relative, not absolute.

The database never deletes. Retrieval for question q ranks every stored entry by
    score = <q, entry> * exp(-c_dec * (t - t_recv_entry))
and returns the top k_ctx into a finite, fixed context. The exact-match entry has
<q,k> = 1; entries for other questions have cross-similarity chi < 1. The agent
can answer q iff its q-entry makes the context, i.e. iff fewer than k_ctx other
entries outscore it. A competitor received at time s outscores the q-entry
(received at t_recv) iff  chi * e^{-c(t-s)} > e^{-c(t-t_recv)}, i.e. iff
    s > t_recv + Delta,   Delta = ln(1/chi) / c_dec.
So the q-entry is visible iff (# insertions after t_recv + Delta) < k_ctx:

    tau_eff = Delta + time for k_ctx further insertions ~ ln(1/chi)/c_dec + k_ctx / r_ins.

Forgetting is therefore ACTIVITY-DEPENDENT: every answer received and every
observation inserts a new entry (the repo always inserts) and crowds the rest;
an agent that stops learning has its old memories resurface. There is no
retrievability threshold and no contradiction rule -- IDK means crowded out.

Everything else as v3: provenance t_obs travels unchanged, newest evidence wins
on merge (but every received answer still counts as an insertion), budget B with
observation share alpha, per-question change rates lambda_q. No caution: a fast
environment makes answers stale, not absent.
"""
from __future__ import annotations
import numpy as np

NEG = -10 ** 9


def zipf(M, s, rng):
    r = rng.permutation(M) + 1.0
    p = r ** (-s); return p / p.sum()


def run(N=20, M=100, B=5, alpha=0.1, c_dec=0.035, chi=0.5, k_ctx=3,
        lam_mean=0.01, lam_disp=1.0, frac_static=0.0,
        s_env=0.0, s_ask=0.0, sigma="staleness_aware", peer="uniform", mean_degree=None,
        n_obs_frac=1.0, answer_policy="open", books=False, book_pub=100, book_frac=0.5,
        book_write_p=0.0, book_write_W=1,
        T=1000, seed=0, record_every=5):
    """chi: cross-question similarity (score floor of off-target entries);
    k_ctx: context size (retrieval depth). Delta = ln(1/chi)/c_dec.
    answer_policy: "open" = answer from any retrievable entry (default);
    "firsthand" = no miming -- only self-observed evidence grounds an answer,
    and hearsay is never re-told. Received answers still insert (and crowd).
    books: an agent-written, slow-moving environment. Every observation is
    recorded into a shared manuscript; every book_pub steps the manuscript is
    published as an edition; a fraction book_frac of ask actions read the
    edition instead of a peer. A book read returns the edition's (value,
    provenance) -- it inserts and crowds like any answer, and it counts as
    legitimate grounding under BOTH policies (a documented record is read,
    not mimed).
    Costly writing mode (book_write_p > 0): the book is fed ONLY by agent writes.
    Each step, w.p. book_write_p an agent spends its ENTIRE budget on a write
    attempt; attempts accumulate and the W-th commits the agent's currently
    groundable knowledge into the book (newest evidence per question wins).
    Reads cost one action as before. Expected publication lag ~ W/book_write_p."""
    rng = np.random.default_rng(seed)
    if lam_disp > 0:
        lam = lam_mean * np.exp(lam_disp * rng.standard_normal(M) - lam_disp ** 2 / 2)
    else:
        lam = np.full(M, lam_mean)
    if frac_static > 0:                                # a random subset of questions never changes
        lam[rng.random(M) < frac_static] = 0.0
    delta = np.log(1.0 / chi) / c_dec
    p_env = zipf(M, s_env, rng)
    p_ask = zipf(M, s_ask, rng)
    truth = np.zeros(M, dtype=np.int64)

    t_obs = np.full((N, M), NEG, dtype=np.int64)
    t_recv = np.full((N, M), NEG, dtype=np.int64)
    val = np.zeros((N, M), dtype=np.int64)
    t_fh = np.full((N, M), NEG, dtype=np.int64)       # grounded slot: receipt time (self-obs or book read)
    o_fh = np.full((N, M), NEG, dtype=np.int64)       # grounded slot: evidence (observation) time
    v_fh = np.zeros((N, M), dtype=np.int64)
    ms_t = np.full(M, NEG, dtype=np.int64)            # book manuscript: newest recorded evidence
    ms_v = np.zeros(M, dtype=np.int64)
    pub_t = np.full(M, NEG, dtype=np.int64)           # last published edition
    pub_v = np.zeros(M, dtype=np.int64)
    C = np.zeros((N, T + 1), dtype=np.int64)          # cumulative insertions per agent, end of step
    write_attempts = np.zeros(N, dtype=np.int64)      # costly-writing progress per agent
    # seed: every question observed once at t=0 by a random agent (each seed inserts)
    seed_a = rng.integers(0, N, size=M)
    t_obs[seed_a, np.arange(M)] = 0; t_recv[seed_a, np.arange(M)] = 0
    t_fh[seed_a, np.arange(M)] = 0; o_fh[seed_a, np.arange(M)] = 0
    ms_t[:] = 0                                        # the seeding observations are all recorded
    np.add.at(C[:, 0], seed_a, 1)

    if mean_degree is None or mean_degree >= N - 1:
        A = np.ones((N, N), dtype=bool); np.fill_diagonal(A, False)
    else:
        A = rng.random((N, N)) < mean_degree / (N - 1)
        A = np.triu(A, 1); A = A | A.T
    nbrs = [np.flatnonzero(A[i]) for i in range(N)]
    n_obs = int(round(n_obs_frac * N))
    rows = np.arange(N)[:, None]

    fh = answer_policy == "firsthand"

    def visible(tnow, Cnow, tref):
        """entry in context iff fewer than k_ctx insertions after its receipt + delta."""
        held = tref > NEG
        cutoff = np.floor(tref + delta).astype(np.int64)
        safe = cutoff >= tnow                          # competitors can't outscore yet
        cutoff = np.clip(cutoff, 0, tnow)
        n_after = Cnow[:, None] - C[rows, cutoff]      # insertions after the cutoff
        return held & (safe | (n_after < k_ctx))

    out = {k: [] for k in ["t", "idk", "idk_demand", "correct", "stale", "dead_q", "sys_corr"]}
    for t in range(1, T + 1):
        truth += rng.random(M) < lam
        Cprev = C[:, t - 1]
        knows = visible(t - 1, Cprev, t_fh if fh else t_recv)   # start-of-step, policy-grounded
        age = t - (t_fh if fh else t_recv)
        ins = np.zeros(N, dtype=np.int64)              # insertions this step

        k_obs = rng.binomial(B, alpha, size=N)
        k_obs[n_obs:] = 0
        k_ask = B - k_obs

        # costly writing: a writer spends the whole step's budget on one attempt
        if books and book_write_p > 0:
            writers = np.flatnonzero(rng.random(N) < book_write_p)
            k_obs[writers] = 0; k_ask[writers] = 0
            write_attempts[writers] += 1
            for i in writers:
                if write_attempts[i] >= book_write_W:
                    write_attempts[i] = 0
                    # commit: currently groundable knowledge -> book, newest evidence wins
                    if fh:
                        ok = (t_fh[i] > NEG) & ((t - t_fh[i]) <= 10**9)
                        src_t, src_v = o_fh[i], v_fh[i]
                        can = visible(t - 1, C[:, t - 1], t_fh)[i]
                    else:
                        src_t, src_v = t_obs[i], val[i]
                        can = visible(t - 1, C[:, t - 1], t_recv)[i]
                    for q in np.flatnonzero(can):
                        if src_t[q] > pub_t[q]:
                            pub_t[q] = src_t[q]; pub_v[q] = src_v[q]

        # The database commits ONCE per iteration: everything read during step t
        # (peer answers, values, ages) is the end-of-(t-1) state; observations and
        # received answers are staged in new_* and written together at the end.
        new_t = t_obs.copy(); new_v = val.copy(); new_r = t_recv.copy()
        new_ft = t_fh.copy(); new_fo = o_fh.copy(); new_fv = v_fh.copy()
        for i in np.flatnonzero(k_obs):
            qs = rng.choice(M, size=k_obs[i], p=p_env)
            new_t[i, qs] = t; new_r[i, qs] = t; new_v[i, qs] = truth[qs]
            new_ft[i, qs] = t; new_fo[i, qs] = t; new_fv[i, qs] = truth[qs]
            if not (books and book_write_p > 0):       # costly mode: only writes feed the book
                ms_t[qs] = t; ms_v[qs] = truth[qs]     # observations are recorded in the manuscript
            ins[i] += k_obs[i]
        for i in np.flatnonzero(k_ask):
            k = k_ask[i]
            if sigma == "uniform":
                w = p_ask.copy()
            elif sigma == "unknown_first":
                # unknown: weight 1; known: the repo's decay strategy, (1 - e^{-c age})^2
                w = np.where(knows[i], (1.0 - np.exp(-c_dec * np.clip(age[i], 0, None))) ** 2, 1.0) * p_ask
            elif sigma == "staleness_aware":
                # known: re-ask by expected staleness of the EVIDENCE, (1 - e^{-lambda_q evidence_age})^2
                # -- a fast world drains the ask budget into re-verification and churns the context
                ev_age = np.clip(t - (o_fh[i] if fh else t_obs[i]), 0, None)
                w = np.where(knows[i], (1.0 - np.exp(-lam * ev_age)) ** 2, 1.0) * p_ask
            elif sigma == "oldest":
                w = np.where(knows[i], np.clip(age[i], 0, None) / (delta + k_ctx), 2.0) * p_ask + 1e-12
            else:
                raise ValueError(sigma)
            if w.sum() <= 0:
                continue
            if sigma == "oldest":
                qs = np.argsort(-w)[:k]
            else:
                qs = rng.choice(M, size=k, p=w / w.sum())
            nb = nbrs[i]
            if len(nb) == 0:
                continue
            for q in qs:
                if books and rng.random() < book_frac:
                    if pub_t[q] > NEG:                 # read the last published edition
                        ins[i] += 1                    # the copy inserts (and crowds) like any answer
                        if pub_t[q] >= new_fo[i, q]:   # a documented record grounds under both policies
                            new_ft[i, q] = t; new_fo[i, q] = pub_t[q]; new_fv[i, q] = pub_v[q]
                        if pub_t[q] >= new_t[i, q]:
                            new_t[i, q] = pub_t[q]; new_v[i, q] = pub_v[q]; new_r[i, q] = t
                    continue
                if peer == "uniform":
                    j = nb[rng.integers(len(nb))]
                elif peer == "freshest":
                    j = nb[np.argmax(t_obs[nb, q] + rng.random(len(nb)) * 0.5)]
                else:
                    raise ValueError(peer)
                if knows[j, q]:
                    ins[i] += 1                        # every received answer inserts (and crowds)
                    # under firsthand policy hearsay grounds nothing and is never re-told
                    if not fh and t_obs[j, q] >= new_t[i, q]:  # newest evidence wins the slot
                        new_t[i, q] = t_obs[j, q]; new_v[i, q] = val[j, q]; new_r[i, q] = t
        upd = new_r > t_recv
        t_obs[upd] = new_t[upd]; val[upd] = new_v[upd]; t_recv[upd] = new_r[upd]
        fupd = new_ft > t_fh
        t_fh[fupd] = new_ft[fupd]; o_fh[fupd] = new_fo[fupd]; v_fh[fupd] = new_fv[fupd]
        C[:, t] = Cprev + ins
        if books and book_write_p == 0 and t % book_pub == 0:   # a new edition appears
            pub_t = ms_t.copy(); pub_v = ms_v.copy()

        if t % record_every == 0:
            knows = visible(t, C[:, t], t_fh if fh else t_recv)
            corr = knows & ((v_fh if fh else val) == truth[None, :])
            out["t"].append(t)
            out["idk"].append(1 - knows.mean())
            out["idk_demand"].append(float(((1 - knows.mean(axis=0)) * p_ask).sum()))
            out["correct"].append(corr.mean())
            out["stale"].append((knows & ~corr).mean())
            out["dead_q"].append(1 - knows.any(axis=0).mean())
            out["sys_corr"].append(corr.any(axis=0).mean())   # >=1 agent answers correctly
    res = {k: np.asarray(v) for k, v in out.items()}
    res["_state"] = dict(t=T, t_recv=t_recv, t_obs=t_obs, t_fh=t_fh, C=C, delta=delta, c_dec=c_dec, k_ctx=k_ctx, lam=lam)
    return res
