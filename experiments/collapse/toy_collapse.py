"""Toy pilot for epistemic collapse: a stripped-down version of the control_charts
no-LLM dynamics with (i) absolute forgetting (a copy is retrievable only for
tau_mem steps after its last insertion) and (ii) an environment-observation
channel (each agent, each step, w.p. eps observes the current true answer of a
uniformly random question).

Everything else mirrors controlcharts.fastsim.core: each agent asks K questions
per step of a uniformly random peer (full mesh or ER graph); the responder
answers from a live copy or says IDK; IDK is never inserted; temporal answers
random-walk with per-step change probability p_env.

Per-question knowledge is an SIS process: birth rate ~ K/Q per live copy,
death rate ~ 1/tau_mem, so R0 ~ K*tau_mem/Q. Collapse (knowledge extinction)
is predicted for R0 < 1 when eps = 0.
"""
from __future__ import annotations
import numpy as np


def run(N=20, Q=100, K=5, tau_mem=10, eps=0.0, p_env=0.0, n_temporal=0,
        T=600, seed=0, strategy="unknown_first", rho=None, mean_degree=None,
        seed_frac=1.0, record_every=1, capacity=None, obs_frac=1.0, obs_mode="random",
        obs_zipf=0.0, ask_zipf=0.0, rank_coupling="same"):
    """obs_zipf / ask_zipf: Zipf exponents of the distribution over questions that the
    environment reveals (observation) and that agents ask about (demand); 0 = uniform.
    rank_coupling: 'same' (a question rare in the environment is rare in demand),
    'independent' (unrelated rankings), 'reversed' (rare in environment = popular in demand)."""
    """Returns dict of per-step series. Questions [0, Q-n_temporal) are static,
    the rest temporal (answers change). Knowledge is seeded once per question
    across the network (as in the repo), on a random agent.

    strategy: 'uniform' (ask uniformly random questions),
              'unknown_first' (repo decay rule: unknown weight 1, known weight
                               (1-exp(-rho*dt))^2, rho defaults to 1/tau_mem),
              'oldest' (ask the K questions with the oldest / missing copies).
    """
    rng = np.random.default_rng(seed)
    if rho is None:
        rho = 1.0 / tau_mem
    NEG = -10**9
    learn = np.full((N, Q), NEG, dtype=np.int64)       # last insertion time
    val = np.zeros((N, Q), dtype=np.int64)             # stored value
    truth = np.zeros(Q, dtype=np.int64)                # current truth (temporal drift)
    n_static = Q - n_temporal

    # seeding: each question seeded exactly once on a random agent (repo rule),
    # optionally only a fraction of questions
    seeded_q = rng.choice(Q, size=int(round(seed_frac * Q)), replace=False)
    owners = rng.integers(0, N, size=len(seeded_q))
    learn[owners, seeded_q] = 0

    # contact graph
    if mean_degree is None or mean_degree >= N - 1:
        nbrs = None
    else:
        p = mean_degree / (N - 1)
        A = rng.random((N, N)) < p
        A = np.triu(A, 1); A = A | A.T
        nbrs = [np.flatnonzero(A[i]) for i in range(N)]

    # rarity: rank-based Zipf weights for observation and demand
    ranks = np.arange(1, Q + 1, dtype=float)
    obs_rank = rng.permutation(Q)                       # obs_rank[q] = rank of q in environment (0 = most common)
    if rank_coupling == "same":
        ask_rank = obs_rank
    elif rank_coupling == "reversed":
        ask_rank = Q - 1 - obs_rank
    else:
        ask_rank = rng.permutation(Q)
    p_obs = ranks[obs_rank] ** (-obs_zipf); p_obs /= p_obs.sum()
    p_ask = ranks[ask_rank] ** (-ask_zipf); p_ask /= p_ask.sum()

    # observation channel: which agents can observe, and which questions
    n_obs = int(round(obs_frac * N))
    observers = np.arange(n_obs)
    if obs_mode == "assigned" and n_obs > 0:
        # partition questions among observers (ownership-like)
        perm = rng.permutation(Q)
        assigned = [perm[i::n_obs] for i in range(n_obs)]

    out = {k: [] for k in ["t", "idk", "idk_demand", "correct", "stale", "alive_q", "coverage",
                           "agent_idk_max", "agent_idk_min", "n_collapsed_agents"]}
    out["p_ask"], out["p_obs"], out["ask_rank"], out["obs_rank"] = p_ask, p_obs, ask_rank, obs_rank

    for t in range(1, T + 1):
        # environment drift
        if n_temporal:
            bump = rng.random(n_temporal) < p_env
            truth[n_static:][bump] += 1
        live = (t - learn) <= tau_mem                    # retrievable copies

        # environment observation channel
        if eps > 0 and n_obs > 0:
            obs = rng.random(n_obs) < eps
            a = observers[obs]
            if obs_mode == "assigned":
                qs = np.array([rng.choice(assigned[i]) for i in a], dtype=np.int64)
            else:
                qs = rng.choice(Q, size=len(a), p=p_obs)
            learn[a, qs] = t
            val[a, qs] = truth[qs]

        # question selection
        if strategy == "uniform":
            q = rng.choice(Q, size=(N, K), p=p_ask)
        elif strategy == "unknown_first":
            dt = np.clip(t - learn, 0, None)
            p_known = (1 - np.exp(-rho * dt)) ** 2
            w = np.where(live, p_known, 1.0) * p_ask[None, :]
            cdf = np.cumsum(w, axis=1); cdf /= cdf[:, -1:]
            u = rng.random((N, K))
            q = np.stack([(cdf[i] < u[i][:, None]).sum(axis=1) for i in range(N)])
            q = np.minimum(q, Q - 1)
        elif strategy == "oldest":
            age = np.where(live, t - learn, 10**6) + rng.random((N, Q))  # tie-break
            q = np.argsort(-age, axis=1)[:, :K]
        else:
            raise ValueError(strategy)

        # peer selection
        if nbrs is None:
            b = rng.integers(0, N - 1, size=(N, K))
            b += (b >= np.arange(N)[:, None])
        else:
            b = np.empty((N, K), dtype=np.int64)
            for i in range(N):
                if len(nbrs[i]) == 0:
                    b[i] = -1
                else:
                    b[i] = rng.choice(nbrs[i], size=K)
        a = np.repeat(np.arange(N), K)
        qf, bf = q.ravel(), b.ravel()
        ok = bf >= 0
        a, qf, bf = a[ok], qf[ok], bf[ok]
        # responder answers from a live copy (state at start of step)
        has = live[bf, qf]
        a, qf, bf = a[has], qf[has], bf[has]
        learn[a, qf] = t
        val[a, qf] = val[bf, qf]

        # bounded memory: keep only the `capacity` most recent live entries
        if capacity is not None:
            live_now = (t - learn) <= tau_mem
            over = np.flatnonzero(live_now.sum(axis=1) > capacity)
            for i in over:
                idx = np.flatnonzero(live_now[i])
                drop = idx[np.argsort(-learn[i, idx])[capacity:]]
                learn[i, drop] = NEG

        if t % record_every == 0:
            live = (t - learn) <= tau_mem
            corr = live & (val == truth[None, :])
            idk_frac = 1 - live.mean()
            per_agent_idk = 1 - live.mean(axis=1)
            out["t"].append(t)
            out["idk"].append(idk_frac)
            out["idk_demand"].append(float(((1 - live.mean(axis=0)) * p_ask).sum()))
            out["idk_by_q"] = 1 - live.mean(axis=0)        # last recorded step
            out["correct"].append(corr.mean())
            out["stale"].append((live & ~corr).mean())
            out["alive_q"].append(live.any(axis=0).mean())        # 1 - system-level dead fraction
            out["coverage"].append(live.any(axis=0).sum())
            out["agent_idk_max"].append(per_agent_idk.max())
            out["agent_idk_min"].append(per_agent_idk.min())
            out["n_collapsed_agents"].append((per_agent_idk >= 0.95).sum())
    return {k: np.asarray(v) for k, v in out.items()}


def time_to_collapse(res, thresh=0.95, key="idk"):
    hit = np.flatnonzero(res[key] >= thresh)
    return res["t"][hit[0]] if len(hit) else np.inf
