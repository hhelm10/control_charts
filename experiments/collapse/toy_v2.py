"""Toy simulator v2: provenance clock + caution + budget.

State per (agent i, question q): t_obs (when the held answer was observed from the
environment; propagates unchanged through exchange; -inf = none) and val.
All M questions are dynamic with per-question change rate lambda_q.

Each step every agent takes B actions; each is an observation w.p. alpha, else an ask.
  observe: q ~ pi_env; t_obs <- t, val <- truth[q]           (only N_obs agents can)
  ask:     q chosen by policy sigma (weighted by pi_ask); peer chosen by policy;
           the peer answers iff it holds q and exp(-lambda_q * (t - t_obs)) >= c;
           the asker keeps whichever of (own, received) has the newer t_obs.
An agent "knows" q iff it holds q and exp(-lambda_q * age) >= c   (caution rule).
IDK fraction = 1 - mean(knows) = per-(agent, question) collapse probability.
"""
from __future__ import annotations
import numpy as np

NEG = -10 ** 9


def zipf(M, s, rng):
    """Zipf(s) weights over M questions assigned to a random ranking."""
    r = rng.permutation(M) + 1.0
    p = r ** (-s); return p / p.sum()


def run(N=20, M=100, B=5, alpha=0.1, c=0.5, lam_mean=0.01, lam_disp=0.0,
        s_env=0.0, s_ask=0.0, sigma="unknown_first", peer="uniform", mean_degree=None,
        n_obs_frac=1.0, tau_recv=None, T=1000, seed=0, record_every=5):
    """tau_recv: memory lifetime on the receipt clock (None = infinite). A copy is
    answerable only if BOTH its receipt age <= tau_recv (agent memory) AND its
    evidence age satisfies the caution rule (environment validity)."""
    rng = np.random.default_rng(seed)
    # environment: per-question speeds, log-normal with mean lam_mean and dispersion lam_disp
    if lam_disp > 0:
        lam = lam_mean * np.exp(lam_disp * rng.standard_normal(M) - lam_disp ** 2 / 2)
    else:
        lam = np.full(M, lam_mean)
    theta = np.log(1.0 / c) / lam                     # evidence-age threshold per question
    p_env = zipf(M, s_env, rng)
    p_ask = zipf(M, s_ask, rng)
    truth = np.zeros(M, dtype=np.int64)

    t_obs = np.full((N, M), NEG, dtype=np.int64)
    t_recv = np.full((N, M), NEG, dtype=np.int64)
    val = np.zeros((N, M), dtype=np.int64)
    tau = np.inf if tau_recv is None else tau_recv
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

    out = {k: [] for k in ["t", "idk", "idk_demand", "correct", "stale", "dead_q"]}
    for t in range(1, T + 1):
        truth += rng.random(M) < lam
        age = t - t_obs
        knows = (t_obs > NEG) & (age <= theta) & ((t - t_recv) <= tau)   # start-of-step state

        # --- budget split ---
        k_obs = rng.binomial(B, alpha, size=N)
        k_obs[n_obs:] = 0                             # agents without environmental access
        k_ask = B - k_obs

        # --- observations (newest evidence by construction) ---
        for i in np.flatnonzero(k_obs):
            qs = rng.choice(M, size=k_obs[i], p=p_env)
            t_obs[i, qs] = t; t_recv[i, qs] = t; val[i, qs] = truth[qs]

        # --- asks ---
        new_t = t_obs.copy(); new_v = val.copy(); new_r = t_recv.copy()
        for i in np.flatnonzero(k_ask):
            k = k_ask[i]
            if sigma == "uniform":
                w = p_ask.copy()
            elif sigma == "unknown_first":
                # unknown: weight 1; known: weight (age/theta)^2 (re-verify as evidence ages)
                w = np.where(knows[i], np.clip(age[i] / theta, 0, 1) ** 2, 1.0) * p_ask
            elif sigma == "oldest":
                w = np.where(knows[i], age[i] / theta, 2.0) * p_ask + 1e-12
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
                if knows[j, q] and t_obs[j, q] >= new_t[i, q]:  # newest evidence wins; re-hearing refreshes memory
                    new_t[i, q] = t_obs[j, q]; new_v[i, q] = val[j, q]; new_r[i, q] = t
        # observations made this step already wrote t (newest); keep them
        upd = new_r > t_recv
        t_obs[upd] = new_t[upd]; val[upd] = new_v[upd]; t_recv[upd] = new_r[upd]

        if t % record_every == 0:
            age = t - t_obs
            knows = (t_obs > NEG) & (age <= theta) & ((t - t_recv) <= tau)
            corr = knows & (val == truth[None, :])
            out["t"].append(t)
            out["idk"].append(1 - knows.mean())
            out["idk_demand"].append(float(((1 - knows.mean(axis=0)) * p_ask).sum()))
            out["correct"].append(corr.mean())
            out["stale"].append((knows & ~corr).mean())
            out["dead_q"].append(1 - knows.any(axis=0).mean())
    return {k: np.asarray(v) for k, v in out.items()}
