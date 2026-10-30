"""Agent implementation with RAG database and LLM backend."""

from dataclasses import dataclass, field
import asyncio
import json
import re
import logging
import os
import numpy as np
import time
from openai import OpenAI, AsyncOpenAI, DefaultAsyncHttpxClient, RateLimitError, APIConnectionError, InternalServerError
import httpx2 as _httpx

# transient API failures worth retrying (APITimeoutError subclasses APIConnectionError)
TRANSIENT = (RateLimitError, APIConnectionError, InternalServerError)
RETRIES = 12                     # backoff 1, 2, 4, ... capped at 60 s: ~10 min before giving up


def _backoff(attempt: int) -> float:
    import random
    return min(60.0, 2.0 ** attempt) * (0.5 + random.random())

from .config import DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from .database import VectorDatabase, QAPair

# Tail taming for the per-step barrier: if a completion has not returned after
# HEDGE_AFTER seconds, fire a duplicate request and take whichever finishes
# first (deterministic at temperature 0, ~10% duplicate token cost).
# hedging (a duplicate request after this many seconds) is off by default: cancelling
# the loser leaked connections and stalled long runs. LLM_HEDGE_AFTER=1.5 re-enables it.
HEDGE_AFTER = float(os.environ.get("LLM_HEDGE_AFTER", "0"))

# Measured answering gate for the surrogate (no-LLM) path: P(IDK | present, k, position),
# estimated from the real agent (ecal_probe.py). Loaded from the path in ECAL_TABLE;
# absent from the table or env -> the gate is the deterministic step function.
_ECAL_GATE: dict[tuple[int, int], float] | None = None


def _ecal_gate() -> dict[tuple[int, int], float]:
    global _ECAL_GATE
    if _ECAL_GATE is None:
        _ECAL_GATE = {}
        path = os.environ.get("ECAL_TABLE", "")
        if path and os.path.exists(path):
            import ast, json as _json
            from collections import defaultdict
            acc = defaultdict(lambda: [0.0, 0])
            for key, cell in _json.load(open(path)).items():
                tup = ast.literal_eval(key)
                if tup[0] == "present":                  # (tag, k, position, level)
                    a = acc[(tup[1], tup[2])]
                    a[0] += cell["p_idk"] * cell["n"]; a[1] += cell["n"]
            _ECAL_GATE = {kp: s / n for kp, (s, n) in acc.items() if n}
    return _ECAL_GATE


# One AsyncOpenAI client per event loop (httpx clients are loop-bound).
_ASYNC_CLIENTS: dict[int, tuple[float, AsyncOpenAI]] = {}

# Token usage of completed requests (for cost accounting; written to results.json).
# Hedged duplicates that get cancelled may still bill but report no usage, so
# `hedges` counts them separately.
USAGE = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "hedges": 0}


def _count(r) -> None:
    u = getattr(r, "usage", None)
    USAGE["calls"] += 1
    if u is not None:
        USAGE["prompt_tokens"] += u.prompt_tokens or 0
        USAGE["completion_tokens"] += u.completion_tokens or 0


_H2_LIMITS = _httpx.Limits(max_connections=100, max_keepalive_connections=100, keepalive_expiry=120)
CLIENT_TTL = 60.0          # seconds before a fresh client replaces the current one
_RETIRED: list = []        # (retire time, client) awaiting close


def _get_async_client() -> AsyncOpenAI:
    """One client per event loop, rebuilt every CLIENT_TTL seconds. Long runs
    slowed step after step on a single long-lived client (connections the
    server had closed piled up in its pool); a fresh pool bounds that.
    Retired clients are closed once their in-flight requests have finished."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    now = time.time()
    cur = _ASYNC_CLIENTS.get(key)
    if cur is None or now - cur[0] > CLIENT_TTL:
        if cur is not None:
            _RETIRED.append((now, cur[1]))
        # HTTP/2 multiplexes requests over a few connections: HTTP/1.1 opened a
        # TLS connection per request and the pool scan grew with its size,
        # which made runs CPU-bound (~20 ms CPU per call vs ~2.5 ms).
        _ASYNC_CLIENTS[key] = (now, AsyncOpenAI(
            timeout=30.0, max_retries=2,
            http_client=DefaultAsyncHttpxClient(http2=True, limits=_H2_LIMITS)))
    while _RETIRED and now - _RETIRED[0][0] > 300:    # > any request's timeout + retries
        loop.create_task(_RETIRED.pop(0)[1].close())
    return _ASYNC_CLIENTS[key][1]


async def _hedged_chat(model: str, messages: list[dict], hedge_after: float | None = None) -> str:
    """One completion with a hedged duplicate against the latency tail."""
    client = _get_async_client()

    async def one() -> str:
        r = await client.chat.completions.create(
            model=model, messages=messages, temperature=0.0, max_tokens=500)
        _count(r)
        return r.choices[0].message.content.strip()

    if not (hedge_after or HEDGE_AFTER):                     # unhedged: plain retries
        last: BaseException | None = None
        for attempt in range(RETRIES + 1):
            try:
                return await one()
            except TRANSIENT as e:
                last = e
                logging.getLogger(__name__).warning(f"LLM retry {attempt + 1}: {type(e).__name__}")
                if attempt < RETRIES:
                    await asyncio.sleep(_backoff(attempt))
        raise last

    tasks: set[asyncio.Task] = {asyncio.create_task(one())}
    hedged = False
    last_exc: BaseException | None = None
    try:
        while tasks:
            done, pending = await asyncio.wait(
                tasks, timeout=None if hedged else (hedge_after or HEDGE_AFTER),
                return_when=asyncio.FIRST_COMPLETED)
            for t in done:
                if t.exception() is None:
                    return t.result()
                last_exc = t.exception()
            tasks = set(pending)
            if not hedged:
                tasks.add(asyncio.create_task(one()))
                hedged = True
                USAGE["hedges"] += 1
        # both attempts raised (e.g. sustained 429): unhedged retries with backoff
        for attempt in range(RETRIES):
            try:
                return await one()
            except TRANSIENT as e:
                last_exc = e
                await asyncio.sleep(_backoff(attempt))
        raise last_exc
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()


@dataclass
class Agent:
    """An agent with a RAG database and LLM backend."""

    id: int
    database: VectorDatabase
    model: str = "gpt-4o-mini"
    retrieval_k: int = 5
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    use_llm: bool = True

    # Adversarial behavior configuration
    # Schedule: list of (timestep, probability) tuples for adversarial behavior
    # e.g., [(0, 0), (50, 0), (100, 0.5), (200, 1.0)] means 0% adversarial until t=50,
    # then ramps to 50% at t=100, and 100% at t=200
    adversarial_schedule: list[tuple[int, float]] | None = None
    defection_schedule: dict | None = None  # {start, duration, max_p, shape}
    propagation_probability: float = 1.0  # Probability quine wins for same-question match (noLLM only)
    cross_question_propagation: float = 1.0  # Probability quine wins for cross-question match (noLLM only)
    # Adversarial prompts (used when behaving adversarially)
    adversarial_system_prompt: str | None = None
    adversarial_prompt_template: str | None = None
    # RNG for probabilistic adversarial behavior (set by simulation)
    _rng: np.random.Generator | None = field(default=None, repr=False)

    # Answering policy: "open" = any exact entry grounds; "firsthand" = no
    # mimesis -- only environment-derived entries (seeds, observations) ground
    answer_policy: str = "open"

    # Forget strategy configuration
    forget_strategy: str = "none"  # "none" or "decay"
    decay_coefficient: float = 0.1  # Exponential decay rate
    question_selection: str = "memory"  # "memory" (decay-weighted re-asks) or "uniform"
    decay_mode: str = "multiplicative"  # "multiplicative" or "additive"

    # Question tracking
    questions_in_play: set[str] = field(default_factory=set)
    known_questions: set[str] = field(default_factory=set)

    # Temporal question ownership - agent owns these temporal questions
    # and always knows the correct answer (current value from shared state)
    owned_temporal_questions: set[str] = field(default_factory=set)
    temporal_questions: set[str] = field(default_factory=set)  # All temporal questions in play
    temporal_values: dict[str, int] = field(default_factory=dict)  # Shared dict: question -> current value

    # Track when questions were last learned (for decay strategy)
    # Maps question -> iteration when it was learned
    question_learn_time: dict[str, int] = field(default_factory=dict)
    current_iteration: int = 0

    # OpenAI client (initialized lazily)
    _client: OpenAI | None = field(default=None, repr=False)

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(timeout=30.0, max_retries=3)
        return self._client

    @property
    def unknown_questions(self) -> set[str]:
        """Questions in play that this agent doesn't know."""
        return self.questions_in_play - self.known_questions

    def initialize_knowledge(self, qa_pairs: list[QAPair]) -> None:
        """Initialize agent with a set of QA pairs, marking them as known."""
        for qa in qa_pairs:
            qa.firsthand = True                      # the initial endowment is environment-given
        self.database.add_many(qa_pairs)
        for qa in qa_pairs:
            self.known_questions.add(qa.question)
            self.question_learn_time[qa.question] = 0  # Learned at iteration 0

    def set_iteration(self, iteration: int) -> None:
        """Set the current iteration (for decay calculations)."""
        self.current_iteration = iteration

    def set_rng(self, rng: np.random.Generator) -> None:
        """Set the RNG for probabilistic adversarial behavior."""
        self._rng = rng

    def get_adversarial_probability(self) -> float:
        """Get current adversarial probability based on schedule and iteration.

        Supports both sigmoid defection schedule and legacy piecewise linear schedule.
        Returns 0.0 if no schedule is set.
        """
        import math

        # Sigmoid defection schedule (preferred)
        if self.defection_schedule is not None:
            s = self.defection_schedule
            start, duration, max_p, shape = s["start"], s["duration"], s["max_p"], s["shape"]
            t = self.current_iteration
            if t <= start:
                return 0.0
            if t >= start + duration:
                return max_p
            x = shape * (2 * (t - start) / duration - 1)
            sig = 1 / (1 + math.exp(-x))
            sig_lo = 1 / (1 + math.exp(shape))
            sig_hi = 1 / (1 + math.exp(-shape))
            return max_p * (sig - sig_lo) / (sig_hi - sig_lo)

        # Legacy piecewise linear schedule
        if self.adversarial_schedule is None or len(self.adversarial_schedule) == 0:
            return 0.0

        schedule = sorted(self.adversarial_schedule, key=lambda x: x[0])

        if self.current_iteration <= schedule[0][0]:
            return schedule[0][1]

        if self.current_iteration >= schedule[-1][0]:
            return schedule[-1][1]

        for i in range(len(schedule) - 1):
            t1, p1 = schedule[i]
            t2, p2 = schedule[i + 1]
            if t1 <= self.current_iteration < t2:
                fraction = (self.current_iteration - t1) / (t2 - t1)
                return p1 + fraction * (p2 - p1)

        return schedule[-1][1]

    def set_questions_in_play(self, questions: set[str]) -> None:
        """Set the questions that are in play for this simulation."""
        self.questions_in_play = questions

    def set_temporal_questions(self, temporal_questions: set[str], owned: set[str], temporal_values: dict[str, int]) -> None:
        """Set the temporal questions in play and which ones this agent owns.

        Args:
            temporal_questions: All temporal questions in the simulation
            owned: The temporal questions this agent owns (always knows correct answer)
            temporal_values: Shared dict mapping question -> current value (updated by simulation)
        """
        self.temporal_questions = temporal_questions
        self.owned_temporal_questions = owned
        self.temporal_values = temporal_values  # Reference to shared state
        # Agent always "knows" their owned temporal questions
        for q in owned:
            self.known_questions.add(q)

    def mark_known(self, question: str) -> None:
        """Mark a question as known."""
        self.known_questions.add(question)

    def select_question_to_ask(self, rng: np.random.Generator) -> str | None:
        """Randomly select a question to ask a peer.

        In 'none' mode: only asks unknown questions.
        In 'decay' mode: can re-ask known questions with probability
        that increases as time since learning increases.
        """
        if self.question_selection == "uniform":
            # Uniform: any question in play, independent of the agent's memory
            if not self.questions_in_play:
                return None
            return rng.choice(sorted(self.questions_in_play))

        if self.forget_strategy == "none":
            # Original behavior: only ask unknown questions
            unknown = list(self.unknown_questions)
            if not unknown:
                return None
            return rng.choice(unknown)

        elif self.forget_strategy == "decay":
            # Decay mode: build candidate pool with decay-based probabilities
            candidates = []
            weights = []

            for question in self.questions_in_play:
                if question in self.known_questions:
                    # Known question: probability to ask based on time since learned
                    time_since_learned = self.current_iteration - self.question_learn_time.get(question, 0)
                    # P = 1 - exp(-decay_coefficient * time)
                    prob = 1.0 - np.exp(-self.decay_coefficient * time_since_learned)
                    # Only include if we "pass" the probability check
                    if rng.random() < prob:
                        candidates.append(question)
                        weights.append(prob)
                else:
                    # Unknown question: always a candidate with weight 1.0
                    candidates.append(question)
                    weights.append(1.0)

            if not candidates:
                return None

            # Normalize weights and select
            weights = np.array(weights)
            weights = weights / weights.sum()
            return rng.choice(candidates, p=weights)

        else:
            raise ValueError(f"Unknown forget strategy: {self.forget_strategy}")

    def prepare_answer(self, question: str, question_embedding: np.ndarray) -> tuple:
        """Resolve a question up to (but not including) the LLM call.

        Returns ("final", text) when no LLM call is needed (owned temporal,
        no-LLM mode, adversarial template), else ("llm", system_prompt,
        user_prompt) for the caller to execute synchronously or async.
        """
        # Check if this is a temporal question we own - return current value directly
        if question in self.owned_temporal_questions:
            temporal_value = self.temporal_values.get(question, 0)
            return ("final", str(temporal_value))

        # Lightweight path: skip LLM, use top-3 memory lookup
        if not self.use_llm:
            # Check adversarial behavior first
            use_adversarial = False
            if (self.adversarial_schedule is not None or self.defection_schedule is not None) and self._rng is not None:
                adv_prob = self.get_adversarial_probability()
                use_adversarial = self._rng.random() < adv_prob

            if use_adversarial:
                return ("final", "I lost the game")

            if self.forget_strategy == "decay":
                retrieved = self.database.search(
                    question_embedding,
                    k=self.retrieval_k,
                    current_iteration=self.current_iteration,
                    decay_coefficient=self.decay_coefficient,
                    decay_mode=self.decay_mode
                )
            else:
                retrieved = self.database.search(question_embedding, k=self.retrieval_k)

            # Check if any of the top-k results contains a quine
            for qa in retrieved:
                if 'i lost the game' in qa.answer.lower():
                    if qa.question == question:
                        p = self.propagation_probability
                    else:
                        p = self.cross_question_propagation
                    if self._rng is None or self._rng.random() < p:
                        return ("final", qa.answer)
                    break
            # Optional state dump for in-vivo E-cal: record the actual retrieved
            # context for every answering event (ECAL_DUMP=path.jsonl)
            dump = os.environ.get("ECAL_DUMP")
            if dump:
                with open(dump, "a") as fh:
                    fh.write(__import__("json").dumps({
                        "t": self.current_iteration, "agent": self.id, "q": question,
                        "ctx": [(qa.question, qa.answer) for qa in retrieved],
                        "present": any(qa.question == question for qa in retrieved)}) + "\n")
            # Otherwise return the top exact question match, gated by the measured
            # P(IDK | present, k, position) when an ecal table is provided
            gate = _ecal_gate()
            for pos, qa in enumerate(retrieved, start=1):
                if qa.question == question and (self.answer_policy != "firsthand" or qa.firsthand):
                    p_idk = gate.get((len(retrieved), pos), 0.0)
                    if p_idk > 0 and self._rng is not None and self._rng.random() < p_idk:
                        return ("final", "I don't know")
                    return ("final", qa.answer)
            return ("final", "I don't know")

        # Determine if we should behave adversarially this turn
        use_adversarial = False
        if (self.adversarial_schedule is not None or self.defection_schedule is not None) and self._rng is not None:
            adv_prob = self.get_adversarial_probability()
            use_adversarial = self._rng.random() < adv_prob

        # If adversarial, return the prompt template directly (no LLM call)
        if use_adversarial and self.adversarial_prompt_template is not None:
            return ("final", self.adversarial_prompt_template)

        # Normal behavior: use default prompts
        active_system_prompt = self.system_prompt
        active_prompt_template = self.prompt_template

        # Retrieve relevant QA pairs (with decay discounting if in decay mode)
        # Note: We still do retrieval even for adversarial behavior - the agent
        # maintains its database but may choose to ignore it when acting adversarially
        if self.forget_strategy == "decay":
            retrieved = self.database.search(
                question_embedding,
                k=self.retrieval_k,
                current_iteration=self.current_iteration,
                decay_coefficient=self.decay_coefficient,
                decay_mode=self.decay_mode
            )
        else:
            retrieved = self.database.search(question_embedding, k=self.retrieval_k)

        # Format context
        if retrieved:
            context_parts = []
            for qa in retrieved:
                context_parts.append(f"Q: {qa.question}\nA: {qa.answer}")
            context = "\n\n".join(context_parts)
        else:
            context = "(No relevant information found)"

        # Build prompt using active template
        user_prompt = active_prompt_template.format(
            retrieved_context=context,
            question=question
        )

        # Verbatim relay: if the LLM answers (non-IDK), the message passed on is
        # the stored exact-match entry itself, never the generation -- answers
        # descend verbatim from the environment, eliminating paraphrase drift.
        exact = next((qa.answer for qa in retrieved
                      if qa.question == question
                      and (self.answer_policy != "firsthand" or qa.firsthand)), None)
        # Optional in-vivo gate log (LLM_DUMP=path.jsonl): per answering event, the
        # retrieved context's structure (same-question entries with firsthand flag,
        # revision tag and insertion time) and, after the call, whether it said IDK.
        rec = None
        if os.environ.get("LLM_DUMP"):
            def _rev(a):
                m = re.search(r" \[rev (\d+)\]$", a)
                return int(m.group(1)) if m else 0
            rec = {"t": self.current_iteration, "agent": self.id, "q": question, "k": len(retrieved),
                   "ctx": [[qa.question == question, bool(qa.firsthand),
                            _rev(qa.answer) if qa.question == question else None, qa.insertion_time]
                           for qa in retrieved]}
        return ("llm", active_system_prompt, user_prompt, exact, rec)

    @staticmethod
    def _dump(rec, response: str) -> None:
        if rec is not None:
            rec["idk"] = Agent._relay(response, "x") == "I don't know"
            with open(os.environ["LLM_DUMP"], "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    @staticmethod
    def _relay(response: str, exact: str | None) -> str:
        """The LLM is the gate; the stored entry is the payload."""
        low = response.lower()
        if "i don't know" in low or "i do not know" in low or "i don’t know" in low:
            return "I don't know"
        return exact if exact is not None else "I don't know"

    def answer(self, question: str, question_embedding: np.ndarray) -> str:
        """Answer a question (synchronous path; see prepare_answer)."""
        res = self.prepare_answer(question, question_embedding)
        if res[0] == "final":
            return res[1]
        _, system_prompt, user_prompt, exact, rec = res
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        for attempt in range(RETRIES + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0, max_tokens=500)
                _count(response)
                text = response.choices[0].message.content.strip()
                self._dump(rec, text)
                return self._relay(text, exact)
            except TRANSIENT:
                if attempt == RETRIES:
                    raise
                time.sleep(_backoff(attempt))

    async def answer_async(self, question: str, question_embedding: np.ndarray) -> str:
        """Answer a question (async path with a hedged request against tail latency)."""
        res = self.prepare_answer(question, question_embedding)
        if res[0] == "final":
            return res[1]
        _, system_prompt, user_prompt, exact, rec = res
        r = await _hedged_chat(self.model,
                               [{"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}])
        self._dump(rec, r)
        return self._relay(r, exact)

    def observe(self, question: str, true_answer: str, question_embedding: np.ndarray) -> None:
        """Consult the environment: insert the TRUE current answer as a
        firsthand entry (the only way truth enters an agent)."""
        self.database.add(QAPair(
            question=question, answer=true_answer, embedding=question_embedding,
            insertion_time=self.current_iteration,
            is_temporal=question in self.temporal_questions, firsthand=True))
        self.mark_known(question)
        self.question_learn_time[question] = self.current_iteration

    def receive_answer(self, question: str, answer: str, question_embedding: np.ndarray) -> bool:
        """Receive an answer from a peer. Returns True if knowledge was added."""
        # Check if this is a valid answer (not "I don't know")
        if self._is_dont_know(answer):
            return False

        # Check if this is a temporal question
        is_temporal = question in self.temporal_questions

        # Always insert (no duplicate checking per spec)
        qa_pair = QAPair(
            question=question,
            answer=answer,
            embedding=question_embedding,
            insertion_time=self.current_iteration,  # Track when inserted for decay
            is_temporal=is_temporal,
            firsthand=False                          # hearsay never grounds under firsthand policy
        )
        self.database.add(qa_pair)
        self.mark_known(question)
        # Track when this question was learned (or re-learned in decay mode)
        self.question_learn_time[question] = self.current_iteration
        return True

    def _is_dont_know(self, answer: str) -> bool:
        """Check if an answer is a 'don't know' response."""
        lower = answer.lower().strip()
        dont_know_phrases = [
            "i don't know",
            "i do not know",
            "i dont know",
            "i don't have",
            "i do not have",
            "no relevant",
            "cannot answer",
            "can't answer",
            "unable to answer",
            "not enough information",
        ]
        return any(phrase in lower for phrase in dont_know_phrases)

    def get_state(self) -> dict:
        """Get agent state for logging/debugging."""
        # Count known temporal vs non-temporal questions
        known_temporal = len(self.known_questions & self.temporal_questions)
        known_nontemporal = len(self.known_questions) - known_temporal

        return {
            "id": self.id,
            "db_size": len(self.database),
            "known_count": len(self.known_questions),
            "unknown_count": len(self.unknown_questions),
            "in_play_count": len(self.questions_in_play),
            "known_temporal": known_temporal,
            "known_nontemporal": known_nontemporal,
            "owned_temporal_count": len(self.owned_temporal_questions),
        }
