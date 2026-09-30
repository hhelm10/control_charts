"""Analytic P(epistemic collapse) for the crowding-memory system (toy_v4 mechanics).

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
             iters=300):
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
        p0 = renewal_idk(mu, D, B, q_ins, k)    # occupancy with observation as the only channel
        delta = mu * p0 / (1 - p0)
        if strategy == "none":
            gain = lambda m: mu
        elif strategy == "mimesis":
            a = K * live / (M * p)
            gain = lambda m, a=a: mu + a * m / (N - 1)
        elif strategy == "books":
            a = book_frac * K * live / (M * p)
            gain = lambda m, a=a: mu + a * book_cov
        else:
            raise ValueError(strategy)
        pi = chain_pi(N, delta, gain)
        lack = (N - n) / N
        p_new = float(np.sum(pi * lack))
        # an ask is made by a non-holder of q; answered iff the target holds q
        held_given_lack = float(np.sum(pi * (N - n) * n / (N - 1)) / max(np.sum(pi * (N - n)), 1e-12))
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
    return {"agent_idk": agent, "pi0": pi0, "pi": pi}


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


def p_chi_collapse(M, pi0, chi):
    return float(stats.binom.sf(math.ceil(chi * M) - 1, M, pi0))
