"""Analytic P(epistemic collapse) and P(misalignment) for the crowding-memory system.

Per agent, per question (renewal argument, discrete time):
  An agent observes q in a step w.p. mu = 1 - (1 - alpha/M)^B. Its grounding
  entry for q, received a steps ago, is retrievable iff a <= D = floor(Delta)
  (Delta = ln(1/sim_floor)/c: newer entries cannot outscore it yet), or fewer
  than k_ctx insertions have landed in the a - D steps since. Each of the B
  actions inserts w.p. q_ins (observations always; asks only when answered),
  so insertions over m steps ~ Bin(mB, q_ins). With the age since the last
  grounding acquisition ~ Geom(mu),
      p = P(IDK) = sum_{a > D} mu (1-mu)^a P(Bin((a-D) B, q_ins) >= k_ctx),
  solved self-consistently with q_ins = alpha + (1 - alpha)(1 - p).

Per question, across agents (birth-death chain on n = # agents that can answer):
  losses:  each holder at rate delta = mu p / (1 - p)   (matches the renewal
           occupancy when mu is the only acquisition channel)
  gains:   each non-holder at rate g(n) = mu + mitigation(n)
      none      g(n) = mu                                  -> Binomial(N, 1-p)
      mimesis   g(n) = mu + a * n/(N-1),  a = K/(M p)       (asks target unknown
                questions; answered iff the random peer holds q)
      books     g(n) = mu_b + a_b * b,  a_b = f K_b/(M p), b = book coverage;
                writers lose their step w.p. p_w (mu_b, K_b scaled by 1 - p_w)
  pi_0 = P(q in collapse) from the stationary distribution.

System (chi-collapse): questions ~ independent, so the number collapsed is
Binomial(M, pi_0) and P(chi-collapse) = P(Bin(M, pi_0) >= ceil(chi M)).

Measured gate (proxy / real LLM): if an agent that holds q still abstains w.p.
eps (and one that lacks q answers w.p. ~0), per-agent abstention is
p + (1-p) eps; for independent agents pi_0 = (p + (1-p) eps)^N. eps = 0 in the toy.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def _polylog_neg(n, r):
    """sum_{m>=1} m^n r^m (Eulerian-number closed form), 0 <= r < 1."""
    # A(n, j): Eulerian numbers
    A = [1] if n == 0 else None
    if n > 0:
        A = [1]
        for nn in range(1, n + 1):
            A = [((j + 1) * (A[j] if j < len(A) else 0) + (nn - j) * (A[j - 1] if j >= 1 else 0))
                 for j in range(nn)]
    if n == 0:
        return r / (1 - r)
    return r * sum(a * r ** j for j, a in enumerate(A)) / (1 - r) ** (n + 1)


def renewal_idk(mu, D, B, q_ins, k):
    """P(IDK) for one agent, closed form (O(k^2) work).

    p = P(age > D) - mu (1-mu)^D sum_{j<k} (q/(1-q))^j sum_{m>=1} C(mB, j) r^m,
    r = (1-mu)(1-q)^B, with C(mB, j) expanded as a polynomial in m.
    """
    q = q_ins
    if q >= 1 - 1e-12:                       # every action inserts: lost at the first step after D
        return float((1 - mu) ** (D + 1) - mu * (1 - mu) ** D * sum(
            math.comb(B, j) * 0 for j in range(k)))
    r = (1 - mu) * (1 - q) ** B
    total = 0.0
    for j in range(k):
        # C(mB, j) = prod_{i<j} (mB - i) / j!  as polynomial in m
        poly = np.poly1d([1.0])
        for i in range(j):
            poly = poly * np.poly1d([B, -i])
        poly = poly / math.factorial(j)
        coeffs = poly.coefficients[::-1]          # c_0 + c_1 m + ...
        sj = sum(cn * _polylog_neg(n, r) for n, cn in enumerate(coeffs))
        total += (q / (1 - q)) ** j * sj
    return float((1 - mu) ** (D + 1) - mu * (1 - mu) ** D * total)


_SIM_SURV = None


def similarity_survival(n_sample=2000, seed=0):
    """Empirical survival function F(x) = P(sigma > x) of off-target cosine
    similarity between NQ questions (normalized embeddings, as in retrieval)."""
    global _SIM_SURV
    if _SIM_SURV is None:
        import os
        import pandas as pd
        pool = {"oai": "nq_embedded_oai.parquet", "nomic": "nq_embedded.parquet"}[
            os.environ.get("EC_POOL", "oai")]               # same pool as the mirror
        df = pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "data", pool))
        E = np.stack(df["embedding"].values).astype(np.float64)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        rng = np.random.default_rng(seed)
        X = E[rng.choice(len(E), size=min(n_sample, len(E)), replace=False)]
        S = X @ X.T
        off = np.sort(S[~np.eye(len(X), dtype=bool)])
        _SIM_SURV = lambda x: 1.0 - np.searchsorted(off, x, side="right") / len(off)
    return _SIM_SURV


def renewal_idk_het(mu, c, B, q_ins, k, surv=None, amax=None):
    """P(IDK) with heterogeneous similarities.

    An entry inserted when the q-entry is u steps old permanently outranks it iff
    its similarity exceeds e^{-c u}; each of the B actions in step u inserts such a
    threat w.p. q_ins * h(u), h(u) = P(sigma > e^{-c u}). The q-entry is visible
    at age a iff fewer than k threats arrived in steps 1..a (exact DP over the
    count, truncated at k); the age since last acquisition is Geom(mu).
    """
    if surv is None:
        surv = similarity_survival()
    if amax is None:
        amax = int(60 / max(mu, 1e-6)) + 50
    dist = np.zeros(k); dist[0] = 1.0          # P(count = j), j < k (mass >= k dropped)
    p_vis = np.empty(amax + 1); p_vis[0] = 1.0
    for u in range(1, amax + 1):
        pr = q_ins * surv(math.exp(-c * u))
        if pr > 0:
            pmf = stats.binom.pmf(np.arange(k), B, pr)   # threats this step (only < k matter)
            dist = np.convolve(dist, pmf)[:k]
        p_vis[u] = dist.sum()
        if p_vis[u] < 1e-12:
            p_vis[u + 1:] = 0.0
            break
    a = np.arange(amax + 1)
    w = mu * (1 - mu) ** a
    return float(np.sum(w * (1 - p_vis)) + (1 - mu) ** (amax + 1))


def renewal_idk_targeted(E_obs, M, c, B, q_ins, k, surv=None, amax=None, iters=60):
    """P(IDK) with heterogeneous similarities AND the pipeline's selection rule:
    an agent observes q at a rate proportional to w(a) = (1 - e^{-c a})^2, a = age
    since it last acquired q, normalized over its M questions:
        hazard(a) = E_obs * w(a) / (M * wbar),  wbar = E_age[w(a)] (self-consistent).
    The stationary age law of this renewal process is P(a) proportional to
    prod_{u<a} (1 - hazard(u)).
    """
    if surv is None:
        surv = similarity_survival()
    if amax is None:
        amax = int(80 * M / max(E_obs, 1e-6)) + 200
    a = np.arange(amax + 1)
    # visibility by age (independent of the observation process)
    dist = np.zeros(k); dist[0] = 1.0
    p_vis = np.empty(amax + 1); p_vis[0] = 1.0
    for u in range(1, amax + 1):
        pr = q_ins * surv(math.exp(-c * u))
        if pr > 0:
            dist = np.convolve(dist, stats.binom.pmf(np.arange(k), B, pr))[:k]
        p_vis[u] = dist.sum()
    w = (1 - np.exp(-c * a)) ** 2
    wbar = 0.5
    for _ in range(iters):
        haz = np.clip(E_obs * w / (M * wbar), 0, 1)
        surv_obs = np.concatenate([[1.0], np.cumprod(1 - haz[:-1])])
        page = surv_obs / surv_obs.sum()
        wbar_new = float(np.sum(page * w))
        if abs(wbar_new - wbar) < 1e-10:
            break
        wbar = 0.5 * wbar + 0.5 * max(wbar_new, 1e-9)
    return float(np.sum(page * (1 - p_vis)))


def renewal_idk_sum(mu, D, B, q_ins, k, amax=None):
    """Reference implementation (explicit sum over ages) for checking renewal_idk."""
    if amax is None:
        amax = int(D + 50 / max(mu, 1e-6) + 50)
    a = np.arange(D + 1, amax + 1)
    w = mu * (1 - mu) ** a
    lost = stats.binom.sf(k - 1, (a - D) * B, q_ins)     # P(Bin >= k)
    return float(np.sum(w * lost) + (1 - mu) ** (amax + 1))


def agent_idk(M, B, alpha, c, sim_floor, k, iters=200):
    """Self-consistent single-agent P(IDK), no mitigation."""
    mu = 1 - (1 - alpha / M) ** B
    D = math.floor(math.log(1 / sim_floor) / c)
    p = 0.5
    for _ in range(iters):
        q_ins = alpha + (1 - alpha) * (1 - p)
        p_new = renewal_idk(mu, D, B, q_ins, k)
        if abs(p_new - p) < 1e-10:
            break
        p = 0.5 * p + 0.5 * p_new
    return p, mu


def chain_pi(N, delta, gain):
    """Stationary distribution of the birth-death chain; gain(n) per non-holder."""
    logw = [0.0]
    for n in range(1, N + 1):
        birth = (N - (n - 1)) * gain(n - 1)
        death = n * delta
        logw.append(logw[-1] + math.log(birth) - math.log(death))
    logw = np.array(logw)
    w = np.exp(logw - logw.max())
    return w / w.sum()


def collapse(M, N=10, B=11, alpha=3 / 11, c=0.05, sim_floor=0.5, k=3,
             strategy="none", book_frac=1.0, book_cov=1.0, write_p=0.0, eps=0.0,
             iters=300, het=False, targeted=False, asks="unknown"):
    """Returns dict(agent_idk, pi0, pi) for one question under a mitigation strategy.

    Everything is solved self-consistently: the insertion rate q_ins (which sets
    how fast memories are crowded out) depends on how often asks are answered,
    which depends on how many agents hold the asked question.
    """
    D = math.floor(math.log(1 / sim_floor) / c)
    K = B * (1 - alpha)
    live = 1 - write_p                          # fraction of agent-steps not spent writing
    mu = (1 - (1 - alpha / M) ** B) * live
    n = np.arange(N + 1)
    p, h = 0.5, 0.5                             # agent IDK; P(an ask is answered)
    for _ in range(iters):
        q_ins = live * (alpha + (1 - alpha) * h)
        if targeted:
            p0 = renewal_idk_targeted(alpha * B * live, M, c, B, q_ins, k)
        elif het:
            p0 = renewal_idk_het(mu, c, B, q_ins, k)
        else:
            p0 = renewal_idk(mu, D, B, q_ins, k)  # occupancy with observation as the only channel
        delta = mu * p0 / (1 - p0)
        if strategy == "none":
            gain = lambda m: mu
        elif strategy == "mimesis":
            a = K * live / (M * p) if asks == "unknown" else K * live / M
            gain = lambda m, a=a: mu + a * m / (N - 1)
        elif strategy == "books":
            a = book_frac * K * live / (M * p) if asks == "unknown" else book_frac * K * live / M
            gain = lambda m, a=a: mu + a * book_cov
        else:
            raise ValueError(strategy)
        pi = chain_pi(N, delta, gain)
        lack = (N - n) / N
        p_new = float(np.sum(pi * lack))
        # an ask is made by a non-holder of q; answered iff the target holds q
        if asks == "unknown":     # asks target questions the asker lacks
            held_given_lack = float(np.sum(pi * (N - n) * n / (N - 1)) / max(np.sum(pi * (N - n)), 1e-12))
        else:                     # uniform asks: a random peer holds the question w.p. 1 - p
            held_given_lack = 1.0 - float(np.sum(pi * (N - n) / N))
        if strategy == "books":
            h_new = book_frac * book_cov + (1 - book_frac) * held_given_lack
        else:
            h_new = held_given_lack
        if abs(p_new - p) < 1e-10 and abs(h_new - h) < 1e-10:
            break
        p = 0.5 * p + 0.5 * p_new
        h = 0.5 * h + 0.5 * h_new
    agent = p_new
    if eps > 0:     # measured gate: holders still abstain w.p. eps
        pi0 = float(np.sum(pi * eps ** n))
        agent = agent + (1 - agent) * eps
    else:
        pi0 = float(pi[0])
    return {"agent_idk": agent, "pi0": pi0, "pi": pi, "q_ins": q_ins}


def book_coverage(M, t, N=10, B=11, alpha=3 / 11, c=0.05, sim_floor=0.5, k=3,
                  write_p=0.05, W=5, dt=1.0):
    """Fraction of questions in the book at time t (cold start: empty book).

    The book never forgets; a question enters when a committing writer holds it.
    An uncovered question is held only through observation, but the agent's
    memory is also crowded by reads of the covered ones, so its occupancy
    p_u(b) depends on coverage b. Commits arrive at N * write_p / W per step:
        d(1 - b)/dt = - (N write_p / W) (1 - p_u(b)) (1 - b).
    """
    D = math.floor(math.log(1 / sim_floor) / c)
    live = 1 - write_p
    mu = (1 - (1 - alpha / M) ** B) * live
    rate = N * write_p / W
    b, tt = 0.0, 0.0
    while tt < t:
        q_ins = live * (alpha + (1 - alpha) * b)
        p_u = renewal_idk(mu, D, B, q_ins, k)
        b = 1 - (1 - b) * math.exp(-rate * (1 - p_u) * dt)
        tt += dt
    return b


def visibility_profile(c, B, q_ins, k, amax, surv=None):
    """P(the q-entry is still retrievable at age a), heterogeneous similarities."""
    if surv is None:
        surv = similarity_survival()
    dist = np.zeros(k); dist[0] = 1.0
    p_vis = np.zeros(amax + 1); p_vis[0] = 1.0
    for u in range(1, amax + 1):
        pr = q_ins * surv(math.exp(-c * u))
        if pr > 0:
            dist = np.convolve(dist, stats.binom.pmf(np.arange(k), B, pr))[:k]
        p_vis[u] = dist.sum()
        if p_vis[u] < 1e-13:                      # memory certainly gone; rest stays 0
            break
    return p_vis


_OTHER_G = {}


def n_other_dist(M, B, q_ins, c, k, q_ins_self=None, L=12.0, n=3000):
    """P(n_other = 0, 1, 2+): how many of the k retrieved entries are about OTHER
    questions when a holder answers q (refusal depends on it,
    epistemic_collapse_mirror.refusal_context).

    Two independent streams of entries, ranked by score = similarity x e^{-c age}:
    copies of q (similarity 1) arrive at rho = B q_ins / M per step, so
    N_q(x) = rho (-ln x) / c of them score above x; other entries arrive at
    B q_ins (1 - 1/M) with the empirical off-target similarity, so
    N_o(x) = B q_ins (1 - 1/M) sum_a P(sigma > x e^{c a}). Scanning x down from 1
    (grid in ln x), the top k fill with whichever stream's point comes first; the
    split (copies, others) at k is conditioned on at least one copy of q (the
    holder's own entry). q_ins_self: insertion rate of q's own copies when it
    differs from the average (a refused question's asks go unanswered, so it has
    fewer copies and more other-question context). Within ~0.05 of the mirror's measured split for M >= 100;
    for M <= 40 it slightly overstates contexts made only of q.
    """
    if k <= 1:
        return np.array([1.0, 0.0, 0.0])
    key = (round(c, 9), L, n)
    if key not in _OTHER_G:                       # G(y) = sum_a P(sigma > e^{y + c a}), y = ln x
        surv = similarity_survival()
        lx = np.linspace(0.0, -L, n + 1)
        G = np.empty(n + 1)
        for i, y in enumerate(lx):
            t = np.exp(y + c * np.arange(int(-y / c) + 1))
            G[i] = float(np.sum(surv(t[t <= 1.0])))
        _OTHER_G[key] = (lx, G)
    lx, G = _OTHER_G[key]
    r = B * q_ins
    r_self = B * (q_ins if q_ins_self is None else q_ins_self)
    dq = np.full(n, r_self / M * (L / n) / c)
    do = np.diff(r * (1 - 1 / M) * G)
    states = [(a, b) for a in range(k) for b in range(k - a)]
    sid = {st: i for i, st in enumerate(states)}
    pst = np.zeros(len(states)); pst[sid[(0, 0)]] = 1.0
    done = np.zeros(k + 1)                        # by number of copies of q in the top k
    for i in range(n):
        pq = min(dq[i], 1.0); po = min(do[i], 1.0 - pq)
        new = pst * (1.0 - pq - po)
        for (a, b), j in sid.items():
            if pst[j] == 0.0:
                continue
            for (a2, b2), pp in (((a + 1, b), pq), ((a, b + 1), po)):
                if a2 + b2 == k:
                    done[a2] += pst[j] * pp
                else:
                    new[sid[(a2, b2)]] += pst[j] * pp
        pst = new
    for (a, b), j in sid.items():                 # fewer than k entries scored above e^{-L}
        if a > 0:
            done[a] += pst[j]
    d = done[1:] / done[1:].sum()
    out = np.zeros(3)
    for a in range(1, k + 1):
        out[min(k - a, 2)] += d[a - 1]
    return out


def misalignment(M, lam_mean, lam_disp=1.0, N=10, B=11, alpha=3 / 11, c=0.05, k=3,
                 n_lam=120, asks="unknown", refusal=True, gate=True):
    """No mitigation (firsthand). Returns agent-level P(correct), P(stale),
    P(IDK) and system-level P(question in collapse), P(question misaligned),
    averaged over lambda_q ~ lognormal(lam_mean, lam_disp).

    An agent holding q answers with its last observation; age a since that
    observation ~ Geom(mu); visible w.p. p_vis(a); correct iff q has not been
    revised in those a steps: (1 - lambda)^a. Agents are independent, so
        collapse  = P_idk^N
        misalign  = (1 - P_correct)^N - P_idk^N.
    Refusal (measured; epistemic_collapse_mirror.refusal_context): questions of
    class k3 / k1 (fractions of the pool) are refused by a holder w.p.
    Phi(a[class, n_other] + sigma z), z ~ N(0, 1) shared by all agents in a run
    (probit random effect), n_other from n_other_dist; results mix the classes
    and integrate over z. gate: other questions' holders abstain w.p. the small
    measured residual (position 1).
    """
    base = collapse(M, N=N, B=B, alpha=alpha, c=c, k=k, het=True, asks=asks)
    p_idk = base["agent_idk"]
    q_ins = base.get("q_ins", alpha + (1 - alpha) * (1 - p_idk))
    mu = 1 - (1 - alpha / M) ** B
    amax = int(60 / mu) + 50
    a = np.arange(amax + 1)
    held = mu * (1 - mu) ** a * visibility_profile(c, B, q_ins, k, amax)   # P(age = a, visible)
    # lognormal quantiles of lambda_q (mean lam_mean)
    z = stats.norm.ppf((np.arange(n_lam) + 0.5) / n_lam)
    lams = lam_mean * np.exp(lam_disp * z - lam_disp ** 2 / 2)
    lams = np.clip(lams, 0, 1)
    H = np.cumsum(held)
    L = lams[:, None]
    pr = L * (1 - L) ** a[None, :]                          # P(R = r | lambda)
    tail = (1 - lams) ** (amax + 1)                         # R beyond amax: H ~ p_hold

    def one(h):
        """question type whose holders answer w.p. h (1 = normal, 1 - refuse_p = refused)."""
        pc = np.array([np.sum(h * held * (1 - l) ** a) for l in lams])  # P(correct | lambda)
        p_hold = h * held.sum()
        col = float((1 - p_hold) ** N)
        # agents' correctness is correlated through the shared time R since q's last
        # revision: given R, agent i is correct iff it holds evidence of age <= R.
        # R ~ Geom(lambda), so  P(no agent correct) = E_R[(1 - H(R))^N],  H(r) = sum_{a<=r} held(a)
        nc = np.sum(pr * ((1 - h * H) ** N)[None, :], axis=1) + tail * (1 - p_hold) ** N
        return np.array([pc.mean(), p_hold - pc.mean(), 1 - p_hold, col, np.mean(nc) - col])

    from epistemic_collapse_mirror import refusal_context
    RC = refusal_context()
    gn = RC["gate"].get(k, RC["gate"][3])
    g = 1.0 - gn[0] if gate else 1.0              # residual IDK of other questions (position 1)
    if refusal:
        sg = RC["sigma"]
        zx, zw = np.polynomial.hermite_e.hermegauss(24)
        zw = zw / zw.sum()
        v = 0.0
        hold = 1.0 - p_idk
        for cl in (1, 2):
            f_c = float(np.mean(RC["cls"] == cl))
            a_c = stats.norm.ppf(np.clip(RC["P"][cl], 1e-9, 1 - 1e-9)) * math.sqrt(1 + sg * sg)
            # self-consistent: refusal leaves q's asks unanswered -> fewer copies of q
            # -> more other-question context -> less refusal
            q_self = q_ins
            for _ in range(8):
                pn = n_other_dist(M, B, q_ins, c, k, q_ins_self=q_self)
                ref_c = float(np.dot(pn, RC["P"][cl]))
                q_new = alpha + (1 - alpha) * hold * (1 - ref_c)
                if abs(q_new - q_self) < 1e-4:
                    break
                q_self = 0.5 * (q_self + q_new)
            for z, w in zip(zx, zw):
                ref = float(np.dot(pn, stats.norm.cdf(a_c + sg * z)))
                v = v + f_c * w * one(1.0 - ref)
        v = v + (1 - float(np.mean(RC["cls"] > 0))) * one(g)
    else:
        v = one(g)
    agent_correct, agent_stale, agent_idk, collapse_q, misalign_q = map(float, v)
    return {"agent_correct": agent_correct, "agent_stale": agent_stale, "agent_idk": agent_idk,
            "collapse": collapse_q, "misalign": misalign_q}


def p_chi_collapse(M, pi0, chi):
    return float(stats.binom.sf(math.ceil(chi * M) - 1, M, pi0))
