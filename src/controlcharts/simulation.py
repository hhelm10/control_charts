"""Main simulation loop with parallel agent communication."""

import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .agent import Agent
from .network import Network
from .hooks import IterationHook, noop_hook

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    """Result of a single agent query."""
    querying_agent_id: int
    responding_agent_id: int
    question: str
    question_embedding: np.ndarray
    answer: str
    knowledge_added: bool = False


@dataclass
class Simulation:
    """Multi-agent information flow simulation."""

    network: Network
    question_embeddings: dict[str, np.ndarray]  # question -> embedding
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(42))
    iteration_hook: IterationHook = field(default=noop_hook)
    max_workers: int = 50
    questions_per_turn: int = 1  # Number of questions each agent asks per turn (K)
    env_queries_per_turn: int = 0  # Environment consultations per agent per turn (E); budget B = K + E
    static_answers: dict | None = None  # question -> base true answer (the environment store)
    question_lambdas: dict | None = None  # question -> per-step change probability (all-temporal mode)
    revisions: dict = field(default_factory=dict)  # question -> current revision counter

    # Temporal question state (shared with agents)
    temporal_values: dict[str, int] = field(default_factory=dict)  # question -> current value
    temporal_change_probability: float = 0.0  # p(change) per question per step

    def run(self, max_iterations: int = 100) -> list[dict]:
        """Run the simulation for max_iterations steps.

        Returns:
            List of step results with query outcomes.
        """
        results = []

        # Calculate progress milestones
        milestones = {int(max_iterations * p): p for p in [0.25, 0.5, 0.75, 1.0]}

        for step in range(max_iterations):
            step_result = self._run_step(step)
            results.append(step_result)

            # Run iteration hook
            self.iteration_hook(step, self.network.agents, self.network)

            # Log progress at milestones
            completed = step + 1
            if completed in milestones:
                pct = int(milestones[completed] * 100)
                total_known = sum(len(a.known_questions) for a in self.network.agents)
                total_questions = len(self.network.agents) * len(self.network.agents[0].questions_in_play)
                logger.info(f"Progress: {pct}% ({completed}/{max_iterations} iterations) - "
                           f"Knowledge: {total_known}/{total_questions}")

        return results

    def current_truth(self, q: str) -> str | None:
        """The environment's current answer for q (revision-versioned in all-temporal mode)."""
        if self.temporal_values and q in self.temporal_values:
            return str(self.temporal_values[q])
        if self.static_answers and q in self.static_answers:
            base = self.static_answers[q]
            rev = self.revisions.get(q, 0)
            return base if rev == 0 else f"{base} [rev {rev}]"
        return None

    def _run_step(self, step: int) -> dict:
        """Run a single simulation step with parallel queries."""
        # Update current iteration for all agents (needed for decay calculations)
        # Also set RNG for probabilistic adversarial behavior
        for agent in self.network.agents:
            agent.set_iteration(step)
            agent.set_rng(self.rng)

        # Probabilistically update temporal question values
        if self.temporal_values and self.temporal_change_probability > 0:
            for question in self.temporal_values:
                if self.rng.random() < self.temporal_change_probability:
                    self.temporal_values[question] += 1

        # All-temporal mode: every question's truth revises at its own rate
        if self.question_lambdas:
            for q, lam in self.question_lambdas.items():
                if self.rng.random() < lam:
                    self.revisions[q] = self.revisions.get(q, 0) + 1

        # Phase 1: Each agent selects questions and peers to ask
        # Phase 0: environment consultations (mechanical; the only source of truth)
        if self.env_queries_per_turn > 0:
            for agent in self.network.agents:
                for _ in range(self.env_queries_per_turn):
                    q = agent.select_question_to_ask(self.rng)
                    if q is None:
                        continue
                    truth = self.current_truth(q)
                    if truth is None:
                        continue
                    agent.observe(q, truth, self.question_embeddings[q])

        # Each agent can ask up to questions_per_turn questions
        queries = []
        for agent in self.network.agents:
            for _ in range(self.questions_per_turn):
                question = agent.select_question_to_ask(self.rng)
                if question is None:
                    continue

                peer_id = self.network.select_peer(agent.id, self.rng)
                if peer_id is None:
                    continue

                queries.append({
                    "querying_agent": agent,
                    "responding_agent": self.network.get_agent(peer_id),
                    "question": question,
                    "question_embedding": self.question_embeddings[question]
                })

        # Phase 2: Execute all queries in parallel (async, hedged against tail latency)
        import asyncio

        async def _gather():
            return await asyncio.gather(*[
                q["responding_agent"].answer_async(q["question"], q["question_embedding"])
                for q in queries])

        if getattr(self, "_aloop", None) is None:
            self._aloop = asyncio.new_event_loop()
        answers = self._aloop.run_until_complete(_gather())
        query_results = [
            QueryResult(
                querying_agent_id=q["querying_agent"].id,
                responding_agent_id=q["responding_agent"].id,
                question=q["question"],
                question_embedding=q["question_embedding"],
                answer=a
            ) for q, a in zip(queries, answers)
        ]

        # Phase 3: Process answers and update agent databases
        for result in query_results:
            querying_agent = self.network.get_agent(result.querying_agent_id)
            result.knowledge_added = querying_agent.receive_answer(
                result.question,
                result.answer,
                result.question_embedding
            )

        return {
            "step": step,
            "num_queries": len(queries),
            "num_knowledge_added": sum(1 for r in query_results if r.knowledge_added),
            "queries": [
                {
                    "from": r.querying_agent_id,
                    "to": r.responding_agent_id,
                    "question": r.question[:50] + "..." if len(r.question) > 50 else r.question,
                    "knowledge_added": r.knowledge_added
                }
                for r in query_results
            ]
        }

    def _execute_query(
        self,
        responding_agent: Agent,
        question: str,
        question_embedding: np.ndarray
    ) -> str:
        """Execute a single query (called in parallel)."""
        return responding_agent.answer(question, question_embedding)


def create_simulation(
    network: Network,
    questions_in_play: list[str],
    question_embeddings: dict[str, np.ndarray],
    seed: int = 42,
    iteration_hook: IterationHook | None = None,
    questions_per_turn: int = 1,
    temporal_values: dict[str, int] | None = None,
    temporal_change_probability: float = 0.0,
    env_queries_per_turn: int = 0,
    static_answers: dict | None = None,
    question_lambdas: dict | None = None,
) -> Simulation:
    """Create a simulation with the given configuration."""
    # Set questions in play for all agents
    questions_set = set(questions_in_play)
    for agent in network.agents:
        agent.set_questions_in_play(questions_set)

    return Simulation(
        network=network,
        env_queries_per_turn=env_queries_per_turn,
        static_answers=static_answers,
        question_lambdas=question_lambdas,
        question_embeddings=question_embeddings,
        rng=np.random.default_rng(seed),
        iteration_hook=iteration_hook or noop_hook,
        questions_per_turn=questions_per_turn,
        temporal_values=temporal_values or {},
        temporal_change_probability=temporal_change_probability
    )
