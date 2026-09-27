"""Temporal data kernel hook for capturing agent response embeddings over time."""

import json
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .agent import Agent
    from .network import Network

logger = logging.getLogger(__name__)


class TemporalKernelHook:
    """Hook that captures embeddings of agent responses at regular intervals.

    Fires every `interval` iterations, samples `sample_size` questions,
    queries each agent. Responses are collected in memory during the simulation,
    then batch-embedded via Modal at the end to avoid repeated cold starts.

    When temporal questions are provided, ALL temporal questions are always
    included in each snapshot, plus a sample of non-temporal questions.
    """

    def __init__(
        self,
        interval: int,
        sample_size: int,
        questions_in_play: list[str],
        question_embeddings: dict[str, np.ndarray],
        output_dir: Path,
        seed: int = 42,
        max_workers: int = 200,
        temporal_questions: list[str] = None,  # Temporal questions to always include
        temporal_values: dict[str, int] = None,  # Shared dict: question -> current value
    ):
        self.interval = interval
        self.sample_size = sample_size
        self.questions_in_play = questions_in_play
        self.question_embeddings = question_embeddings
        self.output_dir = Path(output_dir)
        self.rng = np.random.default_rng(seed)
        self.max_workers = max_workers
        self.temporal_questions = set(temporal_questions or [])
        self.temporal_values = temporal_values or {}  # Reference to shared state

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track metadata for all snapshots
        self.snapshots_metadata: list[dict] = []
        self.truth_fn = None  # set post-construction: question -> current true answer

        # Store collected responses in memory (embedded at end)
        self.pending_snapshots: list[dict] = []

        # Sample questions ONCE at initialization - same sample for all snapshots
        # If we have temporal questions, sample EQUAL numbers of temporal and non-temporal
        if self.temporal_questions:
            # Separate non-temporal questions
            non_temporal_questions = [q for q in self.questions_in_play if q not in self.temporal_questions]

            # Sample EQUAL numbers: use number of temporal questions as the count
            n_temporal = len(self.temporal_questions)
            n_nontemporal = min(n_temporal, len(non_temporal_questions))
            indices = self.rng.choice(len(non_temporal_questions), size=n_nontemporal, replace=False)
            sampled_nontemporal = [non_temporal_questions[i] for i in indices]

            # Combine: ALL temporal + equal number of sampled non-temporal
            self.sampled_questions = list(self.temporal_questions) + sampled_nontemporal

            # Track which sampled questions are temporal (for metadata)
            self.sampled_temporal_mask = [q in self.temporal_questions for q in self.sampled_questions]

            logger.info(f"Temporal kernel: {len(self.temporal_questions)} temporal + "
                       f"{len(sampled_nontemporal)} non-temporal = {len(self.sampled_questions)} total questions (equal sampling)")
        else:
            # No temporal questions - sample from all questions
            n = min(self.sample_size, len(self.questions_in_play))
            indices = self.rng.choice(len(self.questions_in_play), size=n, replace=False)
            self.sampled_questions = [self.questions_in_play[i] for i in indices]
            self.sampled_temporal_mask = [False] * len(self.sampled_questions)
            logger.info(f"Temporal kernel: sampled {len(self.sampled_questions)} questions for all snapshots")

    def _get_sample_questions(self) -> list[str]:
        """Get the sampled questions (same for all snapshots)."""
        return self.sampled_questions

    def _query_agent(self, agent: "Agent", question: str, question_embedding: np.ndarray) -> str:
        """Query a single agent for an answer."""
        return agent.answer(question, question_embedding)

    def _collect_responses(
        self,
        agents: list["Agent"],
        questions: list[str]
    ) -> list[list[str]]:
        """Collect responses from all agents for all questions.

        Returns:
            List of shape [num_agents][num_questions] with response strings.
        """
        import asyncio

        pairs = [(ai, qi, agent, q) for ai, agent in enumerate(agents)
                 for qi, q in enumerate(questions)]

        async def _gather():
            return await asyncio.gather(*[
                agent.answer_async(q, self.question_embeddings[q])
                for _, _, agent, q in pairs])

        if getattr(self, "_aloop", None) is None:
            self._aloop = asyncio.new_event_loop()
        flat = self._aloop.run_until_complete(_gather())
        responses = [[None] * len(questions) for _ in agents]
        for (ai, qi, _, _), r in zip(pairs, flat):
            responses[ai][qi] = r
        return responses

    def _embed_responses(self, responses: list[list[str]]) -> np.ndarray:
        """Embed all responses using Modal.

        Args:
            responses: List of shape [num_agents][num_questions]

        Returns:
            Array of shape [num_agents, num_questions, embedding_dim]
        """
        from .embedding import embed_texts

        # Flatten for batch embedding
        flat_responses = []
        for agent_responses in responses:
            flat_responses.extend(agent_responses)

        # Embed all responses at once
        flat_embeddings = embed_texts(flat_responses)

        # Reshape back to [agents, questions, dim]
        num_agents = len(responses)
        num_questions = len(responses[0])
        embedding_dim = flat_embeddings.shape[1]

        embeddings = flat_embeddings.reshape(num_agents, num_questions, embedding_dim)
        return embeddings

    def _save_snapshot(
        self,
        step: int,
        questions: list[str],
        responses: list[list[str]],
        embeddings: np.ndarray,
        agents: list["Agent"] | None = None,
        agent_ids: list[int] | None = None,
        temporal_values_snapshot: dict[str, int] | None = None,
    ) -> Path:
        """Save a snapshot to disk."""
        snapshot_path = self.output_dir / f"snapshot_step_{step:04d}.npz"

        # Save embeddings as compressed numpy
        np.savez_compressed(
            snapshot_path,
            embeddings=embeddings,
            step=step
        )

        # Get agent IDs from either parameter
        if agent_ids is None:
            agent_ids = [a.id for a in agents]

        # Save metadata separately as JSON
        metadata = {
            "step": step,
            "questions": questions,
            "responses": responses,
            "agent_ids": agent_ids,
            "shape": list(embeddings.shape),
            "temporal_mask": self.sampled_temporal_mask,  # Which questions are temporal
            "temporal_values": temporal_values_snapshot or {},  # Current values for each temporal question
            "truths": ({q: self.truth_fn(q) for q in questions} if self.truth_fn else {}),
        }

        metadata_path = self.output_dir / f"snapshot_step_{step:04d}_meta.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.snapshots_metadata.append({
            "step": step,
            "path": str(snapshot_path),
            "metadata_path": str(metadata_path),
            "shape": list(embeddings.shape)
        })

        return snapshot_path

    def __call__(self, step: int, agents: list["Agent"], network: "Network") -> None:
        """Execute hook if it's time to fire. Collects responses but defers embedding."""
        # Check if we should fire this iteration
        if step % self.interval != 0:
            return

        logger.info(f"Temporal kernel hook firing at step {step}")

        # Get the fixed sample of questions (same for all snapshots)
        questions = self._get_sample_questions()
        logger.info(f"Querying {len(questions)} sampled questions")

        # Collect responses from all agents
        logger.info(f"Querying {len(agents)} agents...")
        responses = self._collect_responses(agents, questions)

        # Store for later batch embedding (capture current temporal values)
        self.pending_snapshots.append({
            "step": step,
            "questions": questions,
            "responses": responses,
            "agent_ids": [a.id for a in agents],
            "temporal_values_snapshot": dict(self.temporal_values),  # Copy current values
        })
        logger.info(f"Stored snapshot for step {step} (will embed at end)")

    def finalize(self) -> None:
        """Batch embed all collected responses and save snapshots.

        Call this after the simulation ends to process all pending snapshots
        with a single Modal invocation (avoiding repeated cold starts).
        """
        if not self.pending_snapshots:
            logger.info("No pending snapshots to finalize")
            return

        logger.info(f"Finalizing {len(self.pending_snapshots)} snapshots...")

        # Flatten all responses for batch embedding
        all_responses = []
        for snapshot in self.pending_snapshots:
            for agent_responses in snapshot["responses"]:
                all_responses.extend(agent_responses)

        logger.info(f"Batch embedding {len(all_responses)} responses via Modal...")

        # Single Modal call for all responses
        from .embedding import embed_texts
        all_embeddings = embed_texts(all_responses)

        logger.info(f"Batch embedding complete. Shape: {all_embeddings.shape}")

        # Reshape and save each snapshot
        embedding_dim = all_embeddings.shape[1]
        offset = 0

        for snapshot in self.pending_snapshots:
            num_agents = len(snapshot["responses"])
            num_questions = len(snapshot["responses"][0])
            snapshot_size = num_agents * num_questions

            # Extract this snapshot's embeddings
            snapshot_embeddings = all_embeddings[offset:offset + snapshot_size]
            offset += snapshot_size

            # Reshape to [agents, questions, embedding_dim]
            embeddings = snapshot_embeddings.reshape(num_agents, num_questions, embedding_dim)

            # Save snapshot
            path = self._save_snapshot(
                step=snapshot["step"],
                questions=snapshot["questions"],
                responses=snapshot["responses"],
                embeddings=embeddings,
                agents=None,  # We stored agent_ids separately
                agent_ids=snapshot["agent_ids"],
                temporal_values_snapshot=snapshot.get("temporal_values_snapshot"),
            )
            logger.info(f"Saved snapshot for step {snapshot['step']} to {path}")

        logger.info(f"Finalized all {len(self.pending_snapshots)} snapshots")
        self.pending_snapshots = []  # Clear pending

    def save_index(self) -> Path:
        """Save an index of all snapshots."""
        index_path = self.output_dir / "snapshots_index.json"
        with open(index_path, "w") as f:
            json.dump(self.snapshots_metadata, f, indent=2)
        return index_path
