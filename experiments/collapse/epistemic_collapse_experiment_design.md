# Epistemic collapse in multi-agent systems: experiment design

Follow-on to *Control Charts for Multi-agent Systems* (Helm, Priebe, Duderstadt; arXiv:2605.11135) using the `hhelm10/control_charts` simulator. Draft v0.1, 2026-09-06.

## 1. The proposal in one paragraph

Treat "epistemic collapse" as **knowledge extinction**. In the simulator every question's answer lives only in agent memories; a copy is born when an agent asks a peer who still holds one, and it dies when memory decay makes it unretrievable. Per question, that is a susceptible–infected–susceptible (SIS) contagion in which "infected" means *knows the answer*. SIS processes have a sharp threshold: knowledge persists when each live copy spawns on average more than one new copy before it expires, and goes extinct otherwise. With K questions asked per agent per step, Q questions in play, and a memory lifetime of τ_mem steps, that reproduction number is approximately

> **R₀ ≈ K · τ_mem / Q**  (independent of N to first order; N controls *how long* extinction takes, not *whether*),

and the environment enters as a source term (agents that observe the world re-seed knowledge) and as a death term (when the world changes, existing copies become stale). Everything on the factor list maps onto a term in that expression: environment speed → death rate of *correct* copies; information per agent from the environment → source rate; memory decay → 1/τ_mem; bandwidth → K; strategy and topology → the effective K/Q. The experiments below are organised to (i) demonstrate collapse and locate the threshold, (ii) measure how fast collapse happens as a function of distance from the threshold and of N, (iii) separate agent-level from system-level collapse, and (iv) show the trade-off that makes collapse hard to avoid in a fast-moving environment: forgetting fast enough to not be *wrong* pushes the system below the threshold at which it can *know anything at all*. A toy pilot (Section 6) confirms the threshold, the N-scaling of time-to-collapse, and the environment-source rescue, so the design is not speculative.

## 2. What the paper and repo give us

The paper's simulation: N agents (N=5 in the paper) each with an LLM (gpt-4o-mini) plus a FAISS vector memory; M queries split into static and environmental; each step every agent picks a peer on a complete graph and asks K=5 questions; retrieval returns the top-k=3 entries scored by ⟨q,k⟩ + e^{−ρΔt}; the LLM is told to return the stored answer or "I don't know". Static agents (ρ=0) converge to a fixed configuration and hit chance-level accuracy on environmental queries; dynamic agents (ρ=0.05) keep learning. The monitoring machinery (TDKPS → iso-mirror → Shewhart chart) is downstream of the response snapshots and is unchanged by anything proposed here.

How the factors of interest map onto the code (all paths relative to `src/controlcharts/`):

| Factor | Where it lives | Current values used |
|---|---|---|
| Environment speed | `data.temporal_change_probability` (per temporal question per step, `simulation.py:79-82`, `fastsim/core.py:387-389`); temporal answers are the integer iteration counter | 0.04 in configs; 0.004 in the paper text |
| Information each agent gets from the environment | **Ownership only**: `owned_temporal_questions` — owners always answer with the live value (`agent.py:211-213`). Round-robin, `n_temporal // N` per agent. No parameter for static questions; each static question is seeded exactly once network-wide (`cli.py:270-284`) | n_temporal=10, N=5 → 2 owned/agent |
| Memory decay | `simulation.forget_strategy.{strategy, decay_coefficient, decay_mode}`; ranking in `database.py:57-104` | decay 0.05, additive in configs |
| Communication bandwidth | `simulation.questions_per_turn` (K) | 5 |
| Communication strategy | `Agent.select_question_to_ask` (`agent.py:154-197`): `none` = ask only unknowns; `decay` = unknowns weight 1, knowns weight (1−e^{−ρΔt})² (inclusion × weight). Peer choice is uniform over neighbours (`network.py:79-84`) | `decay` |
| Question pool | `data.total_questions` (Q), `data.questions_per_agent` (initial seeding) | Q=50 (40 static + 10 temporal), 8 seeded per agent |
| Topology | `network.topology` (`full_mesh`, `ring`, `star`, `random`, `custom`); fastsim: `er` with `mean_degree` | full mesh |
| Retrieval depth | `agents.retrieval_k` (fastsim caps at 7) | 3 |

Two simulators exist. The LLM path (`use_llm: true`) is the paper's. The `use_llm: false` path is a lookup-table stand-in, and `fastsim/` is a vectorised re-implementation of it that scales to N=10⁵ and writes snapshots in the same format, so `run_tdkps_analysis` and the figure scripts work unchanged. The rebuttal grid (`experiments/REBUTTAL_scaling.md`) ran 30 seeds per cell across N ∈ {5…10⁵} × degree × Q ∈ {50, 500} with fastsim, so the infrastructure for large replicated grids is in place.

## 3. Where "I don't know" comes from today, and two gaps

An agent answers IDK exactly when no entry for the asked question appears in its top-k retrieval. Three things follow from reading the code that matter for this project.

**The database never deletes, so forgetting is relative, not absolute.** Under additive decay an old exact-match entry scores 1 + ε forever; it becomes invisible only when ≥k *fresher* entries for *other* questions outrank it (cos + e^{−ρΔt} with Δt small). Under multiplicative decay the exact entry's score goes to 0, but so do all the other old entries, so again an entry is only hidden by fresher ones. Consequently: (a) an agent that stops receiving answers has its old memories resurface, which is a self-correcting force against collapse; (b) the effective memory lifetime depends on the agent's *insertion rate* and on the cosine floor of the embedding model (nomic), not on ρ alone. Fastsim hard-codes the calibrated result as `SAME_Q_VISIBILITY = 17` steps (`fastsim/core.py:56`). This "crowding" forgetting is real and worth one experiment (E0 below), but it is an awkward primary knob because it is coupled to everything else.

**Fastsim cannot say IDK about anything it has ever learned.** `_resolve` returns CORRECT whenever the recency window is non-empty (`core.py:310`), so in fastsim IDK means "never learned", full stop. Any collapse study on fastsim needs an explicit expiry.

**The only environment channel is ownership of temporal questions**, and owners are perfect, permanent oracles. There is no way to vary "how much information each agent gets from the environment" for static questions, or to make environmental access partial or noisy.

One small discrepancy to be aware of: in the LLM path `Agent.answer` calls `database.search` without `decay_mode` (`agent.py:270-276`), so the LLM sim always uses *multiplicative* decay even when the config says `additive`; the no-LLM path passes it through (`agent.py:226-233`). The paper's formula is additive. Whichever is intended, the two arms currently differ, and the LLM-validation arm (E7) should pin this down before comparing to fastsim.

## 4. Definitions and metrics

Responses on the probe panel (the snapshot hook already queries every panel agent on a fixed set of probe questions every `interval` steps) are partitioned four ways: **correct**, **stale** (a substantive answer that is no longer the truth), **IDK**, and **quine** (adversarial payload; zero in these experiments unless E6 adds one). This is a straightforward extension of the accuracy code in `experiments/figures/figure1.py:57-79`, which already separates IDK from substantive answers.

Agent-level collapse. For agent n at time t, f_n(t) = fraction of probe questions answered IDK, computed over questions n does not own. Agent n is *collapsed at t* if f_n(t) ≥ 0.95, and its collapse time is the first t after which it stays collapsed for ≥ w steps (w = the snapshot interval × 5, to reject transients). Report the distribution of collapse times across agents and seeds.

System-level collapse. Question q is *dead at t* if no agent answers it substantively (equivalently, no live copy exists anywhere and no owner). Let A(t) = fraction of questions alive. The system is collapsed when A(t) ≤ 0.05; T_sys is the first such t. Because dead static questions can never be revived without an environment source, A(t) is monotone in the ε = 0 case and T_sys is a clean survival endpoint (Kaplan–Meier across seeds, censored at max_iterations).

Two ancillary system quantities are cheap and diagnostic: **coverage** C(t) = |∪_n known_n| / Q (how much the collective still knows) versus **mean redundancy** (live copies per alive question). Collapse proper is C → 0; "epistemic homogenisation" is C stable while redundancy → N (everyone knows the same few things). Both should be plotted, since some parameter regimes (e.g. targeted strategies) may trade one for the other.

Why both levels are needed: system collapse implies every agent is collapsed, but the converse fails in two interesting ways. Under sparse topologies or heterogeneous decay, peripheral agents collapse while a core keeps the knowledge alive; and under an environment source, the *system* can hold every question alive somewhere while each *agent* is almost entirely ignorant (the pilot's Figure B, left panel, shows exactly this at R₀ = 0.5: 95% of questions alive somewhere, 80% IDK per agent).

Monitoring view. The same snapshots feed TDKPS/iso-mirror. Under collapse every agent's response vector converges to the embedding of "I don't know", so TDKPS points coalesce and the iso-mirror trajectory flattens. E6 asks whether the paper's adaptive chart notices.

## 5. Minimal model extensions (spec)

Three additions, all backwards compatible (defaults reproduce current behaviour).

**(a) Absolute forgetting via a retrievability threshold.** Add `forget_strategy.score_threshold: θ` (default 0 = off). In `VectorDatabase.search`, drop candidates whose discounted score is below θ before taking top-k; in the LLM path the prompt then contains no matching entry and the model says IDK; in the no-LLM path return IDK if no exact match survives. Under multiplicative decay this gives a lifetime τ_mem = ln(1/θ)/ρ for an exact match (cos = 1); under additive decay use threshold on the recency term alone. For fastsim, add `tau_mem` directly: in `_resolve`, treat a window as empty unless `t − learn_time[b,q] ≤ tau_mem` (one extra mask on line 292), and count "known" the same way in `_weight_matrix`. Also expose a geometric variant (each copy expires w.p. 1/τ_mem per step) so we can check that the threshold depends on mean lifetime and not on the deterministic cutoff.

Rationale for a threshold rather than deletion: it is the smallest change to the paper's retrieval function that turns "old" into "unretrievable", it is what any production RAG system with a similarity cutoff already does, and it leaves the memory intact for the monitoring analysis.

**(b) Environment observation rate ε.** Add `data.env_observation_rate: ε` (default 0). Each step, each agent w.p. ε observes the current true answer to a uniformly random question (static or temporal) and inserts it as a fresh copy. Ownership becomes the special case "ε = 1 on a fixed subset"; keep `n_temporal` ownership as an option (`data.env_channel: owners | sampled | both`) because the paper's results use it. In fastsim this is a vectorised insert before `_select_questions` (`core.py:391`).

**(c) Strategy variants for `select_question_to_ask`.** Keep `none` and `decay`; add `uniform` (ignore knowledge state) and `oldest` (ask the K questions whose own copy is oldest or missing — the greedy refresh policy). Optionally add an informed peer choice, `peer_selection: uniform | last_answered` (prefer peers who answered substantively recently), which is the cheapest "communication strategy" that could raise the effective K/Q.

**(d) Logging.** Per step in fastsim `history`: `alive_q`, `coverage`, `mean_redundancy`, `n_collapsed_agents`, `idk_frac`, `stale_frac`, `correct_frac`. Per snapshot: the four-way response partition per agent. These are a few lines each on top of what `step()` already records.

## 6. Mean-field prediction and pilot evidence

Per question, with X live copies among N agents in a full mesh: each of the N − X non-holders asks that question w.p. ≈ K/Q per step and reaches a holder w.p. X/(N−1), so births ≈ (K/Q)·X·(N−X)/(N−1); each copy dies at rate 1/τ_mem. Hence R₀ = K·τ_mem/Q. Standard SIS results then predict: for R₀ < 1, extinction in O(τ_mem · log N) steps; for R₀ > 1, a quasi-stationary state with a fraction ≈ 1 − 1/R₀ of agents knowing each question and an extinction time growing exponentially in N; near R₀ = 1, a critical slowing-down. Because every static question is seeded on a single agent, there is also an early "founder" extinction risk ≈ 1/R₀ per question even above threshold — visible as the initial drop in the trajectories below. With a source rate ε, per-question re-seeding at rate εN/Q makes extinction impossible and replaces it with an endemic level. Below threshold, linearising the SIS dynamics gives an expected ε·τ_mem/(Q(1−R₀)) live copies per (agent, question), so the agent-level IDK fraction is ≈ 1 − ε·τ_mem/(Q(1−R₀)) (pilot check at τ_mem = 10, Q = 100, R₀ = 0.5: predicted 0.94 and 0.80 at ε = 0.3 and 1, observed 0.94 and 0.81), while the fraction of questions alive somewhere is ≈ 1 − exp(−εN·τ_mem/(Q(1−R₀))) (predicted 0.70 at ε = 0.3, observed 0.60). The N in the second expression and not the first is why the *system* recovers long before *agents* do. For a temporal question changing w.p. p_env per step, a copy is correct only if inserted after the last change, so the reproduction number for *correct* knowledge is K/(Q·(1/τ_mem + p_env)) unless correct copies are re-injected from the environment, and the population of correct copies is fed only by environmental observation at rate εN/Q. That yields three conditions for a healthy system — K·τ_mem/Q ≳ 1 (spread), τ_mem ≲ 1/p_env (don't be stale), εN/Q ≳ p_env (someone is actually watching) — and a feasibility window τ_mem ∈ (Q/K, 1/p_env) that is **empty when Q·p_env/K > 1**. In that regime, no choice of memory decay avoids both collapse and staleness. That is the headline result to aim for; it is the epistemic analogue of the paper's learning–security trade-off (Theorem 2).

A pilot (`pilot/toy_collapse.py`, ~120 lines of numpy) implements the repo's asking/answering rules with (a) and (b) added, 12 seeds per cell, N ∈ {5, 20, 100}, Q = 100, K = 5, T = 800.

![](pilot/pilot_fig_A_threshold.png)

*Figure A.* Left: fraction of questions still known by anyone at the end of the run versus R₀ = K·τ_mem/Q. The transition sits at R₀ ≈ 1.25–1.5 for all three N, slightly above the mean-field 1 (duplicate asks and the seeding-on-one-agent founder effect both reduce effective births). Middle: median time to system collapse rises steeply approaching the threshold (τ_mem = 25, R₀ = 1.25: ~170–290 steps at N ≥ 20). Right: mean trajectories at N = 20; the two-stage shape — a fast founder-extinction drop, then slow attrition — is expected and is what the LLM sim should reproduce.

![](pilot/pilot_fig_B_factors.png)

*Figure B.* Left: at R₀ = 0.5 an environment channel keeps most questions alive somewhere for ε ≥ 0.3 while agents remain > 90% IDK — system-level and agent-level collapse decouple. Middle: strategy shifts the threshold modestly (oldest-first > unknown-first > uniform), as expected since none of them changes K. Right: on ER graphs at N = 100, mean degree ≤ 4 collapses at R₀ = 1.5 where the full mesh does not; sparse graphs lower the effective birth rate.

![](pilot/pilot_fig_C_env_speed.png)

*Figure C.* All questions temporal, ε = 0.1. Below the threshold (τ_mem ≤ 20) the answer is IDK regardless of p_env; above it, the answer is *wrong* at a rate set by p_env, and the correct fraction is capped by the environment sampling rate (εN/Q = 0.02 per question per step) rather than by τ_mem. Note this grid did not exhibit the "IDK-or-stale" impossibility because ε was too low for any cell to be correct; E3 fixes this with a proper ε × p_env × τ_mem design.

Caveats: the pilot uses deterministic expiry, ignores the retrieval-crowding mechanism, and has no LLM. Its role is to fix the parameter ranges and the predictions, not to stand in for results.

## 6b. Factor taxonomy and the one-panel-per-factor figure

Every knob in the simulator (and every one on the original factor list) sorts into four factors. The fourth is the one the list left as "?": the **observation** channel — how information gets from the environment into agents at all. It is neither an agent property nor an environment property; it is the interface, and in the repo it is currently hard-coded as "owners are perfect oracles for their temporal questions".

| Factor | Variables (code parameter) | Enters the mean-field model as |
|---|---|---|
| **Agent** | number N (`agents.count`); memory decay speed ρ / lifetime τ_mem (`forget_strategy.decay_coefficient` + proposed threshold); memory size (proposed `capacity`; `retrieval_k` is the read-side analogue) | death rate 1/τ_mem; capacity caps knowledge at capacity/Q; N sets extinction time, not the threshold |
| **Environment** | size Q (`data.total_questions`); speed p_env (`data.temporal_change_probability`); share of dynamic answers (`data.n_temporal`/Q) | Q dilutes the birth rate (K/Q); p_env kills *correct* copies, not copies |
| **Communication** | bandwidth K (`simulation.questions_per_turn`); topology (`network.topology`, `mean_degree`); asking strategy (`forget_strategy.strategy`, proposed `question_strategy`); peer selection | birth rate K/Q × graph factor |
| **Observation** | rate ε (proposed `env_observation_rate`); how many agents observe (proposed `obs_frac`; currently all agents own ~n_temporal/N questions); what they observe (random vs assigned subset, i.e. sampled vs ownership) | source term εN_obs/Q |

The figure below varies each factor in its own panel while holding the other three at a common baseline (N = 20, τ_mem = 30, unbounded memory; Q = 100, flat environment, p_env = 0.003, all answers dynamic; K = 5, full mesh, unknown-first; ε = 0.03, all agents observing random questions — chosen so the baseline sits just above the collapse threshold, R₀ = 1.5, and both directions are visible). Every x-axis is a *resource* — memory lifetime, environmental flatness, bandwidth, observation rate — so all curves fall as x grows and scarcity of the resource is the factor-specific lever that induces collapse. Encoding is identical across panels: hue = factor; lightness = level of the factor's second variable (light → dark = small → large); solid/filled-circle vs dashed/hollow-square = the factor's third variable (baseline vs alternative); grey dotted vertical = baseline value; grey dashed vertical = R₀ = 1 where the x variable maps onto R₀. Metric: agent-level "I don't know" fraction, mean over the last 200 of 1000 steps and 12 seeds — equivalently the average collapse likelihood of an (agent, question) pair.

![](pilot/fig_factors_idk.png)

What each panel says. *Agent*: collapse is a threshold in τ_mem between 20 and 40 (R₀ = 1 → 2) and N barely moves it; a memory that holds only 25 entries pins IDK at ≥ 0.75 regardless of τ_mem (size bounds knowledge but does not cause collapse). *Environment*: flatness is the lever. As the world's revealed questions go from flat (normalised entropy 1) to Zipf-concentrated (entropy 0.2), average collapse likelihood at Q = 100 rises from 0.03 to 0.56; at Q = 125 the system is already near-collapsed and flatness only finishes the job (0.87 → 0.99); at Q = 50 (R₀ = 3) it does nothing, because peer transmission alone sustains every question and the environment channel is not load-bearing. So flatness matters exactly when the system is leaning on the environment to stay alive. Speed p_env has no effect on IDK at any flatness — solid and dashed coincide — because a faster world makes answers stale, not absent. *Communication*: bandwidth K has a sharp threshold at K ≈ 3–5 on the full mesh, sparse graphs (mean degree 2) push it to K ≈ 10, and the asking strategy is nearly irrelevant. *Observation*: ε is the one knob that moves the curve smoothly rather than as a threshold; concentrating observation in a few agents (1 of 20) needs ~10× the per-agent rate to reach the same IDK level, and random vs assigned observation makes little difference.

Two companion figures with the same layout and encoding are in `pilot/`: `fig_factors_notcorrect.png` swaps the metric for 1 − correct (IDK + confidently stale), which is where p_env acts, and where bandwidth shows a non-monotonic effect (more sharing spreads stale answers faster than fresh ones arrive); `fig_factors_system.png` uses the system-level dead-question fraction, where N does matter (more agents → more environmental observations per question → fewer dead questions). The script `pilot/plot_factors.py` regenerates all three from `factor_results.json`, and `run_factors.py` holds the baseline and per-factor grids in one dict, so the same figure can be regenerated from fastsim output once extensions (a)–(c) land — only the loader changes.

### 6c. Rarity: non-uniform distributions over questions

The repo, the pilot above, and the mean-field model all assume every question is equally likely to be revealed by the environment and equally likely to be asked. Relaxing that adds a variable to two factors at once — *environment* (which questions the world reveals: π_obs) and *communication* (which questions agents ask about: π_ask) — and it changes the character of collapse. With per-question asking probability π_ask(q), the reproduction number becomes question-specific, R₀(q) = K·τ_mem·π_ask(q), so instead of one threshold the system has a **knowledge frontier**: questions with π_ask(q) > 1/(K·τ_mem) survive by peer transmission, the rest survive only on whatever the environment feeds them. Under Zipf demand with exponent s = 1 and Q = 100 that frontier sits at rank ≈ K·τ_mem/H_Q ≈ 29 for uniform asking; the paper's unknown-first strategy pushes it out (an agent that already knows the head spends its asks on the tail) but does not remove it.

![](pilot/fig_rarity.png)

Pilot (same baseline as the factor figure, Zipf exponent s from 0 to 2). Three things stand out. First, rarity in the *environment* alone (uniform asking, skewed observation) raises IDK from 0.03 to 0.48 at s = 2: the observation channel's rescue is only as broad as its support, so the tail reverts to its ε = 0 fate. Second, rarity in *demand* alone raises uniform-weighted IDK to 0.56 while the demand-weighted IDK stays at 0.01–0.04 — the tail dies and no probe that samples questions the way agents ask them will register it. Any collapse metric therefore has to declare its weighting; the snapshot hook's fixed probe panel is uniform, which is the right default, but a demand-weighted companion belongs in every table. Third, when both are skewed, whether the rankings coincide matters only at moderate skew: at s = 1, "rare in the environment = popular in demand" halves IDK relative to aligned rankings (0.24 vs 0.40) because peer transmission and observation cover complementary parts of the question set; by s ≥ 1.5 both channels are too concentrated for complementarity to help.

For the experiment list this adds one sweep to E2/E3 (π_obs skew) and one to E4 (π_ask skew, with the rank-coupling arm), and it adds the per-question-rank plot as a standard output: collapse under rarity is graded rather than all-or-nothing, so "how quickly" becomes "how fast the frontier moves inward". Implementation is a `p` argument to two `rng.choice` calls (`run_rarity.py` shows both) — in the repo, a weight vector in `Agent.select_question_to_ask` and in the proposed observation step.

### 6d. Model v2: provenance clock, caution, budget — and the revised factor figure

Design decisions taken after the first round (all reflected in `pilot/toy_v2.py`): every stored answer carries two timestamps, t_obs (when it was observed from the environment; travels unchanged through exchange) and t_recv (when this agent received it); when an agent receives an answer it already holds, the newer t_obs wins; all M questions are dynamic with per-question change rates λ_q (log-normal about a mean λ̄ with dispersion σ_λ; static questions are the λ_q → 0 limit); an agent answers q only if its estimated probability of still being right, e^{−λ_q(t − t_obs)}, is at least a caution level c, so the evidence-age threshold is θ_q = ln(1/c)/λ_q and in a fast enough world an agent declines even on evidence one step old; each agent has a budget of B actions per step, a fraction α of them observations of the environment (question drawn from π_env) and the rest asks (question drawn by policy σ weighted by π_ask, peer drawn by policy); memory lifetime on the receipt clock and capacity are kept as parameters but set to ∞; N_obs = N, every agent can observe every question, observations are exact. Notation from here on: N agents, M questions indexed by q, m = (1−α)B bandwidth, G topology.

Two consequences. Once forgetting runs on the observation clock, communication can no longer sustain knowledge on its own — a lineage dies θ_q steps after it was observed regardless of how much it circulates — so with infinite agent memory the system-level survival of a question depends only on whether someone observed it within θ_q (≈ εN π_env(q) θ_q ≳ 1 with ε = αB), and communication decides only how far that evidence spreads before it expires. In that setting the communication panel shows agent-level collapse but *no* system-level collapse at all (the first v2 sweep, at τ = ∞, had a flat zero there). For communication to be load-bearing at the system level, agent memory has to be finite and shorter than both evidence validity and the observation interval: the observer forgets the copy (receipt clock, τ) before the evidence goes stale (θ_q) and before anyone re-observes it (M/(αBN)), so the evidence survives only if it was passed on and kept circulating — the SIS threshold R₀ = mτ/M from v1, now confined to the validity window. The second consequence is that caution and speed multiply: at fixed c the lifetime is ln(1/c)/λ_q, so environment speed, which had no effect on IDK in v1, becomes one of the steepest levers.

The baseline was therefore chosen so that all three clocks are ordered τ < M/(αBN) < θ: N = 20, M = 50, B = 5, α = 0.01 (99 asks per observation, one observation of each question every ~50 steps), τ = 20, c = 0.5, λ̄ = 0.003 (θ = 231), flat π_env and π_ask, full mesh, random peer, ask unknown-first, giving R₀ ≈ 2 on the mesh. At this baseline the system is healthy (IDK 0.11, 1% of questions dead) and every factor has a lever that carries it to collapse at both levels.

![](pilot/figure1_collapse_by_factor.png)

*Figure 1 candidate.* Top row: agent-level P("I don't know"). Bottom row: system-level P(no agent can answer). Columns: one intrinsic lever per factor; legends apply to both rows. Reading the rows together: *Agent — caution c.* Agent-level IDK rises from ≈0.05 at c = 0.3 to 0.6 at c = 0.8 and 0.96 at c = 0.95; N now matters (N = 5 starts at 0.7 because αBN observations per step scale with N), and asking oldest-evidence-first lowers agent-level IDK but *raises* system-level death at small N, because re-verifying what you already know spends asks that would otherwise spread rare lineages. The companion not-correct figure keeps the U-shape in c: an optimal caution exists for a given speed. *Communication — demand concentration.* On the mesh, system-level death goes 0.01 → 0.05 → 0.35 → 0.50 across s = 0, 1, 2, 3 and agent-level IDK 0.13 → 0.82; a degree-2 graph starts at 0.15 dead and 0.64 IDK with flat demand. Peer selection by freshest evidence cuts agent-level IDK substantially at every s but barely moves system-level death — it redistributes evidence, it cannot create it. *Environment — mean speed λ̄.* System death 0.01 → 0.62 → 0.89 across λ̄ = 0.003, 0.03, 0.1 with uniform speeds; spreading speeds across questions lowers both metrics at every λ̄ (collapse concentrates on the volatile minority) and a skewed π_env raises both. *Observation — asks per observation.* From 9 to 999 asks per observation, system death goes 0 → 0.72 at B = 5; a budget of 10 tolerates ~3× the ratio; halving the observing population shifts the curve left by about one step of the ratio.

## 7. Prioritised experiments

Seeds: 30 per cell for fastsim (matches the rebuttal grids), 5 per cell for LLM runs. Report, for every cell: A(t) and C(t) curves, Kaplan–Meier T_sys, per-agent collapse-time distribution, and the four-way response partition on the probe panel. Costs are rough: fastsim cells at N ≤ 1000, T = 2000 run in seconds to a minute; LLM cells at N = 5, K = 5, T = 1000 are ~25k completions plus snapshot queries (~$2–3 each at gpt-4o-mini prices, an hour or two wall-clock with the existing thread pool).

### E0 — Collapse in the unmodified code (do first; zero implementation cost)

Question: does the existing simulator already exhibit collapse, and by which mechanism? The `gen_configs.py` docstring reports that Q = 8N + 10 "freezes" the mesh: mean knowledge per agent stays at the seeded ~9 and nothing propagates. That is a never-learned form of collapse driven by K/Q.

Design: fastsim, N = 5 and N = 100, K = 5, `forget_strategy: decay` (coefficient 0.05, additive), Q ∈ {50, 100, 200, 500, 1000, 2000} with `n_temporal` = 20% of Q (as in the rebuttal) and `questions_per_agent` = 0.8·Q/N (each static question seeded exactly once; note that if N × questions_per_agent is less than the static count, the remainder are never seeded and are dead from t = 0), T = 2000. Metrics: IDK fraction on probes over time, `mean_known`, A(t) (needs only the `alive_q` log line, since in fastsim alive = ever learned).

Prediction: a sharp drop in acquired knowledge around Q/K ≈ SAME_Q_VISIBILITY·(something O(1)), i.e. Q ≈ 100–200 at K = 5. Also run 3 LLM seeds at N = 5, Q ∈ {50, 200, 500} to see whether crowding (Section 3) produces *post-learning* IDK in the real system — this is the one thing fastsim cannot show, and it decides how much E7 matters.

### E1 — Locate the threshold: the (K, τ_mem, Q) phase diagram (core experiment)

Question: is collapse a threshold phenomenon in R₀ = K·τ_mem/Q, and is the threshold N-invariant?

Design: fastsim with extension (a); ε = 0; static questions only (`n_temporal: 0`) so that extinction is absorbing; full mesh. Grid: K ∈ {1, 2, 5, 10, 20} × τ_mem ∈ {5, 10, 20, 40, 80, 160} × Q ∈ {50, 200, 1000}, N ∈ {5, 20, 100, 1000}, T = 3000, 30 seeds. That is 90 cells × 4 values of N; prune by sampling R₀ ∈ [0.25, 8] on a log grid rather than the full cross if needed. Run both deterministic and geometric expiry for one N to confirm the threshold depends on the mean lifetime.

Predictions: (i) A(∞) as a function of R₀ collapses onto one curve across (K, τ_mem, Q) combinations with the same product; (ii) the transition is at R₀ ≈ 1.2–1.5 and does not move with N; (iii) median T_sys ∝ τ_mem·log N below threshold and grows exponentially with N above it; (iv) the per-agent collapse-time distribution is narrow above threshold (everyone collapses together, late) and wide below it (founder extinctions). Deliverable: Figure A of this document, done properly, plus a data-collapse plot of A(∞) vs R₀.

### E2 — Environment information: ε and ownership, and the agent/system split

Question: how much environmental input keeps a sub-threshold system alive, and does it rescue agents or only the system?

Design: fastsim with (a) + (b). Fix Q = 200, K = 5, N ∈ {20, 200}. τ_mem ∈ {10, 20, 40} (R₀ = 0.25, 0.5, 1.0) and, above threshold, τ_mem = 80. ε ∈ {0, 10⁻³, 3·10⁻³, 10⁻², 3·10⁻², 0.1, 0.3, 1}. Two channel types: `sampled` (ε as defined) and `owners` (a fraction of questions owned by one agent each, matching the paper's mechanism), with total observation budget matched (εN observations per step vs. number of owned questions).

Predictions: below threshold, A(∞) rises with εN/Q while per-agent IDK stays ≈ 1 until ε·τ_mem is O(1); the endemic level of live copies per question is ≈ εN·τ_mem/Q·(1/(1−R₀)); ownership concentrates knowledge (high A, low C for non-owners) whereas sampling spreads it. This experiment produces the "agent-level vs system-level collapse" figure.

### E3 — Environment speed × memory: the IDK-or-stale trade-off (headline experiment)

Question: for a fast-moving environment, is there any memory lifetime that avoids both collapse and staleness?

Design: fastsim with (a) + (b), all-temporal or 50/50 mix (report separately). N = 20, K = 5, Q = 100. p_env ∈ {10⁻³, 3·10⁻³, 10⁻², 3·10⁻², 0.1}; τ_mem ∈ {5, 10, 20, 40, 80, 160, 320}; ε chosen so that εN/Q ∈ {0.3, 1, 3} × p_env (i.e. the environment is sampled slower than, at, and faster than it changes). Also a matched `owners` arm reproducing the paper's ownership channel. 30 seeds, T = 2000.

Predictions: (i) IDK fraction ≈ 1 for τ_mem < Q/K regardless of p_env; (ii) stale fraction ≈ p_env·τ_mem/(1 + p_env·τ_mem) above threshold; (iii) correct fraction is non-negligible only where εN/Q ≳ p_env *and* τ_mem is inside (Q/K, 1/p_env); (iv) the maximum over τ_mem of the correct fraction falls to ≈ 0 once Q·p_env/K exceeds ~1. Deliverable: a three-panel heatmap like Figure C, plus a line plot of max_τ correct(τ) versus Q·p_env/K showing the window closing. If (iv) holds, it is the paper's Theorem-2 style result for epistemics and deserves a short analytic treatment.

### E4 — Bandwidth versus strategy

Question: at fixed K, how much can a smarter communication strategy move the threshold, and is bandwidth or strategy the cheaper lever?

Design: fastsim with (a) + (c). Q = 200, N = 20, ε = 0. Strategies: `uniform`, `decay` (the paper's), `oldest`, and `oldest` + `peer_selection: last_answered`. For each, sweep K ∈ {1, 2, 5, 10} and τ_mem so that R₀ ∈ {0.5, 0.75, 1, 1.25, 1.5, 2, 3}. Add one "broadcast" variant if cheap: each answer is also written to a random second agent (models re-sharing), which doubles effective K.

Predictions: strategy shifts the threshold by ≲ 30% (pilot: oldest > unknown-first > uniform), whereas K shifts it linearly; informed peer selection helps most near threshold and on sparse graphs. Report the threshold R₀* per strategy by fitting a logistic to A(∞) vs R₀.

### E5 — Topology and heterogeneity: who collapses first

Question: how do sparse contact graphs and heterogeneous agents produce agent-level collapse without system collapse?

Design: fastsim with (a). N = 100 and 1000, Q = 200, K = 5, τ_mem giving R₀ ∈ {1.5, 2, 4}. Topologies: ER with mean degree ∈ {2, 4, 8, 16, N−1}; ring; star (`network.topology`); for N = 100 also a two-community stochastic block model via `custom` adjacency. Heterogeneity: draw τ_mem per agent from a log-normal with the same mean and CV ∈ {0, 0.5, 1}; and a "mixed" population with 20% of agents at τ_mem = 5.

Predictions: the epidemic threshold on ER graphs scales with ⟨k²⟩/⟨k⟩, so low-degree agents collapse first and the star hub last; heterogeneous τ_mem yields a persistent tail of collapsed agents even when the system is far above threshold (the short-memory agents are permanently ignorant "free riders" who nonetheless keep asking and get answers, so their agent-level IDK is set by 1 − K·τ_mem,i/Q). Deliverable: per-agent collapse fraction versus degree and versus τ_mem,i.

### E6 — Does the control chart see collapse coming?

Question: the paper shows adaptive charts miss slow adversarial drift. Slow collapse is also a drift; does the iso-mirror chart raise an alarm, and at what point relative to T_sys?

Design: re-use the snapshot pipeline unchanged. Take E1 cells at R₀ ∈ {0.5, 0.9, 1.1} (fast, slow, and marginal collapse) with N = 5 and N = 100 (panel of 100) and run `run_tdkps_analysis` with fixed and adaptive control (burn-in 100, window 200, k ∈ {2, 3}). Compare the first alarm time to T_sys and to the time the IDK fraction crosses 0.5. Also compute two candidate statistics directly from the snapshots: the IDK rate and the TDKPS dispersion (mean pairwise distance between agents at time t), and run the same Shewhart rule on each.

Predictions: the fixed chart alarms in every collapse case; the adaptive chart alarms for fast collapse only and is blind to marginal (R₀ ≈ 1) collapse, mirroring Theorem 2; the dispersion statistic alarms earliest because collapse is a contraction of the perspective space, which the iso-mirror (a *between-timepoint* distance) is slow to register. This is the experiment that ties the follow-on back to the control-chart paper.

### E7 — LLM validation of selected cells

Question: does the real system (gpt-4o-mini + nomic + FAISS) behave like fastsim once the threshold is added?

Design: the paper's configuration (N = 5, Q = 50 with 10 temporal, K = 5, retrieval_k = 3, decay 0.05) plus extension (a) with θ chosen to give τ_mem ∈ {10, 20, 40, 80} (R₀ ≈ 1, 2, 4, 8 at K/Q = 0.1) and, separately, Q ∈ {50, 200} at τ_mem = 20. 5 seeds, T = 1000, snapshot interval 10. Fix the `decay_mode` discrepancy first and run one arm each way. Score LLM outputs with the existing IDK phrase list, and manually audit 200 sampled responses for hallucinated (non-IDK, non-stored) answers, because an LLM that guesses instead of saying IDK converts collapse into confabulation, which is a different failure and worth its own number.

Predictions: same threshold location within a factor of ~1.5 (crowding adds an effective extra death rate, so the LLM system should collapse slightly *earlier* than fastsim); a non-zero confabulation rate that grows as the retrieved context gets less relevant.

## 8. Suggested order and effort

E0 needs nothing but configs and can start today; it also tells us whether crowding alone produces post-learning IDK in the LLM sim, which sets how much weight E7 carries. Extensions (a), (b), (d) to fastsim are roughly a day; (c) is another half day. E1 and E3 are the core of the paper and should run next, followed by E2 (which shares E3's code paths). E4 and E5 are cheap once the grid harness exists (`fastsim/gen_configs.py` and `phase_grid.sh` are the templates). E6 reuses E1's snapshots. E7 is the expensive tail and should be sized once E0 is in.

Deliverables per experiment are listed inline; the paper-level figures we are aiming for are: (1) the R₀ data-collapse plot with N-invariance and the T_sys survival curves (E1); (2) the agent-vs-system decoupling under an environment source (E2); (3) the p_env × τ_mem heatmap and the closing feasibility window (E3); (4) the who-collapses-first plot on sparse and heterogeneous networks (E5); (5) alarm-time versus collapse-time for the control charts (E6); (6) an LLM/fastsim overlay at matched cells (E7).

## 9. Risks and open questions

The R₀ = K·τ_mem/Q form assumes uniform asking; the paper's `decay` strategy re-asks known questions at a rate that itself depends on ρ, which couples the birth rate to the death rate (the pilot tied ρ = 1/τ_mem for that reason). E4 should sweep ρ and θ independently to untangle "how fast I forget" from "how eagerly I re-ask". Retrieval crowding means the LLM system's effective τ_mem is state-dependent (Section 3); if E0 shows that matters, the analytic model needs a death rate that grows with insertion rate. Fastsim caps retrieval_k at 7 and treats cross-question retrieval as a probability, so retrieval-depth effects can only be studied in the LLM sim. Finally, "I don't know" is a prompt-induced behaviour in the LLM arm: we should check that the collapse endpoint is IDK rather than confabulation before claiming the phenomenon is epistemic honesty rather than epistemic failure.

## Appendix: config sketch for the extended fastsim

```yaml
experiment: {name: e1-collapse-K5-tau20-Q200-N100-s0}
data:
  total_questions: 200
  questions_per_agent: 2          # each static question seeded once (Q/N)
  n_temporal: 0
  temporal_change_probability: 0.0
  env_observation_rate: 0.0       # NEW (b)
  env_channel: sampled            # NEW (b): owners | sampled | both
agents:
  count: 100
  retrieval_k: 3
  use_llm: false
  propagation_probability: 0.0    # no adversary
  cross_question_propagation: 0.0
network: {topology: full_mesh}
simulation:
  max_iterations: 3000
  seed: 0
  questions_per_turn: 5           # K
  question_strategy: decay        # NEW (c): none | decay | uniform | oldest
  peer_selection: uniform         # NEW (c)
  forget_strategy:
    strategy: decay
    decay_coefficient: 0.05
    decay_mode: additive
    tau_mem: 20                   # NEW (a): fastsim expiry; LLM path derives it from score_threshold
    expiry: deterministic         # NEW (a): deterministic | geometric
  temporal_kernel: {enabled: true, interval: 10, n_nontemporal_sample: 20}
control_bar: {burn_in: 100, window_size: 200, k: 2}
```
