"""Toy simulator v3: memory decay instead of caution.

Memory is infinite; a stored answer's retrieval relevance is <q,k> * exp(-c_dec * dt)
with dt = time since the entry was received (exact match: <q,k> = 1). An entry is
retrievable -- and its holder answers -- iff relevance >= theta_ret, i.e. iff
receipt age <= tau_eff = ln(1/theta_ret)/c_dec. There is no caution: agents answer
with whatever is retrievable, so a fast environment produces STALE answers, not IDK.

State per (agent i, question q): one copy, carrying t_obs (when the value was
observed from the environment; travels unchanged through exchange), t_recv
(receipt time; the decay clock), and val. Merge rule on receipt:
  merge="evidence" (default, the project's decided rule): accept iff the incoming
      t_obs >= own t_obs; accepting refreshes t_recv (re-hearing refreshes memory).
  merge="receipt" (repo-faithful): always accept -- the freshest receipt is the
      most relevant entry, so circulation refreshes retrievability and can launder
      staleness (old evidence overwrites newer), as in the repo's database.

Each step every agent takes B actions; each is an observation w.p. alpha, else an ask.
  observe: q ~ pi_env; t_obs <- t, t_recv <- t, val <- truth[q]   (only N_obs agents)
  ask:     q chosen by policy sigma (weighted by pi_ask); peer chosen by policy;
           the peer answers iff its copy is retrievable; the asker stores it fresh.
An agent "knows" q iff its copy is retrievable. IDK fraction = 1 - mean(knows).
"""
from __future__ import annotations
import numpy as np

NEG = -10 ** 9


def zipf(M, s, rng):
    """Zipf(s) weights over M questions assigned to a random ranking."""
    r = rng.permutation(M) + 1.0
    p = r ** (-s); return p / p.sum()


def run(N=20, M=100, B=5, alpha=0.1, c_dec=0.035, theta_ret=0.5,
        lam_mean=0.01, lam_disp=0.0,
        s_env=0.0, s_ask=0.0, sigma="unknown_first", peer="uniform", mean_degree=None,
        n_obs_frac=1.0, merge="evidence", conflict=False, T=1000, seed=0, record_every=5):
    """c_dec: relevance decay rate; theta_ret: retrievability threshold.
    Effective memory lifetime tau_eff = ln(1/theta_ret)/c_dec on the receipt clock.
    conflict: memory never deletes, so the previously held copy stays retrievable
    until its own relevance decays; if it is still retrievable and disagrees with
    the current copy, the retrieved context is contradictory and the agent
    abstains (IDK). This is the one channel through which environment speed can
    reach P(IDK): a faster world makes successive copies disagree more often."""
    rng = np.random.default_rng(seed)
    if lam_disp > 0:
        lam = lam_mean * np.exp(lam_disp * rng.standard_normal(M) - lam_disp ** 2 / 2)
    else:
        lam = np.full(M, lam_mean)
    tau = np.log(1.0 / theta_ret) / c_dec             # effective lifetime, receipt clock
    p_env = zipf(M, s_env, rng)
    p_ask = zipf(M, s_ask, rng)
    truth = np.zeros(M, dtype=np.int64)

    t_obs = np.full((N, M), NEG, dtype=np.int64)
    t_recv = np.full((N, M), NEG, dtype=np.int64)
    val = np.zeros((N, M), dtype=np.int64)
    p_val = np.zeros((N, M), dtype=np.int64)          # previous (superseded) copy, for conflict
    p_recv = np.full((N, M), NEG, dtype=np.int64)
    # seed: every question observed once at t=0 by a random agent
    seed_a = rng.integers(0, N, size=M)
    t_obs[seed_a, np.arange(M)] = 0; t_recv[seed_a, np.arange(M)] = 0

    # contact graph
    if mean_degree is None or mean_degree >= N - 1:
        A = np.ones((N, N), dtype=bool); np.fill_diagonal(A, False)
    else:
        A = rng.random((N, N)) < mean_degree / (N - 1)
        A = np.triu(A, 1); A = A | A.T
    nbrs = [np.flatnonzero(A[i]) for i in range(N)]
    n_obs = int(round(n_obs_frac * N))

    def answerable(t):
        ret = (t_recv > NEG) & ((t - t_recv) <= tau)  # current copy retrievable
        if not conflict:
            return ret
        contra = (p_recv > NEG) & ((t - p_recv) <= tau) & (p_val != val)
        return ret & ~contra                          # contradictory context -> abstain

    out = {k: [] for k in ["t", "idk", "idk_demand", "correct", "stale", "dead_q", "conflicted"]}
    for t in range(1, T + 1):
        truth += rng.random(M) < lam
        r_age = t - t_recv
        knows = answerable(t)                         # start-of-step state

        # --- budget split ---
        k_obs = rng.binomial(B, alpha, size=N)
        k_obs[n_obs:] = 0                             # agents without environmental access
        k_ask = B - k_obs

        # --- observations ---
        for i in np.flatnonzero(k_obs):
            qs = rng.choice(M, size=k_obs[i], p=p_env)
            sh = qs[(t_obs[i, qs] < t) & (t_recv[i, qs] > NEG)]   # supersede: keep old copy as prev
            p_val[i, sh] = val[i, sh]; p_recv[i, sh] = t_recv[i, sh]
            t_obs[i, qs] = t; t_recv[i, qs] = t; val[i, qs] = truth[qs]

        # --- asks (peer state frozen at start of step) ---
        new_t = t_obs.copy(); new_v = val.copy(); new_r = t_recv.copy()
        for i in np.flatnonzero(k_ask):
            k = k_ask[i]
            if sigma == "uniform":
                w = p_ask.copy()
            elif sigma == "unknown_first":
                # unknown: weight 1; known: weight (receipt age / tau)^2 (re-ask as memory fades)
                w = np.where(knows[i], np.clip(r_age[i] / tau, 0, 1) ** 2, 1.0) * p_ask
            elif sigma == "oldest":
                w = np.where(knows[i], r_age[i] / tau, 2.0) * p_ask + 1e-12
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
                if peer == "uniform":
                    j = nb[rng.integers(len(nb))]
                elif peer == "freshest":
                    j = nb[np.argmax(t_obs[nb, q] + rng.random(len(nb)) * 0.5)]
                else:
                    raise ValueError(peer)
                if knows[j, q] and (merge == "receipt" or t_obs[j, q] >= new_t[i, q]):
                    new_t[i, q] = t_obs[j, q]; new_v[i, q] = val[j, q]; new_r[i, q] = t
        upd = new_r > t_recv                          # observations this step already wrote t; keep them
        sh = upd & (new_t > t_obs) & (t_recv > NEG)   # strictly newer evidence supersedes: keep old copy as prev
        p_val[sh] = val[sh]; p_recv[sh] = t_recv[sh]
        t_obs[upd] = new_t[upd]; val[upd] = new_v[upd]; t_recv[upd] = new_r[upd]

        if t % record_every == 0:
            knows = answerable(t)
            corr = knows & (val == truth[None, :])
            out["t"].append(t)
            out["idk"].append(1 - knows.mean())
            out["idk_demand"].append(float(((1 - knows.mean(axis=0)) * p_ask).sum()))
            out["correct"].append(corr.mean())
            out["stale"].append((knows & ~corr).mean())
            out["dead_q"].append(1 - knows.any(axis=0).mean())
            out["conflicted"].append(((t_recv > NEG) & ((t - t_recv) <= tau) & ~knows).mean())
    res = {k: np.asarray(v) for k, v in out.items()}
    res["_state"] = dict(t=T, t_recv=t_recv, p_recv=p_recv, t_obs=t_obs, tau=tau, c_dec=c_dec)
    return res
