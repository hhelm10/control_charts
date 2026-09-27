"""Configuration models for experiments."""

from typing import Optional
from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata."""
    name: str
    description: str = ""


class DataConfig(BaseModel):
    """Data configuration."""
    total_questions: int = Field(default=500, description="Total QA pairs in play")
    questions_per_agent: int = Field(default=50, description="Initial knowledge per agent")
    n_temporal: int = Field(default=0, description="Number of temporal questions (0 to disable)")
    lambda_mean: float = Field(default=0.0, description="Mean per-question change rate (all questions temporal; log-normal)")
    lambda_disp: float = Field(default=1.0, description="Log-normal dispersion of per-question change rates")
    temporal_change_probability: float = Field(default=0.04, description="Probability each temporal question changes per step (1/25 = 0.04)")


class DefectionScheduleConfig(BaseModel):
    """Sigmoid-shaped defection schedule."""
    start: int = Field(description="Step where defection probability begins rising")
    duration: int = Field(description="Number of steps for the sigmoid transition")
    max_p: float = Field(default=0.5, description="Maximum defection probability")
    shape: float = Field(default=5.0, description="Sigmoid steepness: low=gradual, high=sharp")


class CustomAgentConfig(BaseModel):
    """Custom prompt override for a specific agent."""
    id: int
    system_prompt: Optional[str] = None
    prompt_template: Optional[str] = None
    # Legacy piecewise linear schedule
    adversarial_schedule: Optional[list[list[float]]] = None
    # Sigmoid defection schedule (preferred)
    defection_schedule: Optional[DefectionScheduleConfig] = None
    # Apply this entry to `replicate` consecutive agents starting at `id`
    # (ids id..id+replicate-1). Lets an adversary population scale with N
    # without writing N/5 identical config entries.
    replicate: int = Field(default=1, ge=1)


class AgentsConfig(BaseModel):
    """Agent configuration."""
    count: int = Field(default=10, description="Number of agents")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model for completions")
    answer_policy: str = Field(default="open", description="open | firsthand (no mimesis)")
    retrieval_k: int = Field(default=5, description="Top-k retrieval")
    use_llm: bool = Field(default=True, description="Use LLM for answering; if False, use lightweight memory lookup")
    propagation_probability: float = Field(default=1.0, description="Probability quine wins for same-question match in top-k (noLLM only)")
    cross_question_propagation: float = Field(default=1.0, description="Probability quine wins for cross-question match in top-k (noLLM only)")
    custom: list[CustomAgentConfig] = Field(default_factory=list)


class NetworkConfig(BaseModel):
    """Network topology configuration."""
    topology: str = Field(default="full_mesh", description="Topology type: full_mesh | er")
    edge_probability: float = Field(default=0.3, description="For random topology")
    mean_degree: Optional[float] = Field(default=None, description="Mean degree for 'er' (Erdos-Renyi) topology; density = mean_degree/(N-1)")
    adjacency: Optional[list[list[int]]] = Field(default=None, description="For custom topology")


class TemporalKernelConfig(BaseModel):
    """Temporal data kernel hook configuration."""
    enabled: bool = Field(default=False, description="Enable temporal kernel hook")
    interval: int = Field(default=10, description="Fire every k iterations")
    sample_size: int = Field(default=10, description="Number of questions to sample per snapshot (non-temporal)")
    n_nontemporal_sample: int = Field(default=10, description="Number of non-temporal questions to sample (all temporal questions are always included)")


class ForgetStrategyConfig(BaseModel):
    """Forget strategy configuration."""
    strategy: str = Field(default="none", description="Forget strategy: 'none' or 'decay'")
    decay_coefficient: float = Field(default=0.1, description="Exponential decay rate for decay strategy")
    decay_mode: str = Field(default="multiplicative", description="Decay mode: 'multiplicative' (score * exp(-kt)) or 'additive' (score + exp(-kt))")


class ControlBarConfig(BaseModel):
    """Control bar visualization configuration."""
    burn_in: int = Field(default=100, description="Number of initial timesteps to skip before computing control bars")
    window_size: int = Field(default=100, description="Sliding window size for computing std (in iterations)")
    k: float = Field(default=2.0, description="Number of std deviations for control limits")



class SimulationConfig(BaseModel):
    """Simulation parameters."""
    max_iterations: int = Field(default=100, description="Maximum simulation steps")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    questions_per_turn: int = Field(default=1, description="Questions each agent asks per turn (K)")
    env_queries_per_turn: int = Field(default=0, description="Environment consultations per agent per turn (E); budget B = K + E")
    forget_strategy: ForgetStrategyConfig = Field(default_factory=ForgetStrategyConfig)
    temporal_kernel: TemporalKernelConfig = Field(default_factory=TemporalKernelConfig)


class Config(BaseModel):
    """Full experiment configuration."""
    experiment: ExperimentConfig
    data: DataConfig = Field(default_factory=DataConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    control_bar: ControlBarConfig = Field(default_factory=ControlBarConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Default prompts
DEFAULT_SYSTEM_PROMPT = "You answer questions from a database."

DEFAULT_PROMPT_TEMPLATE = """Database results:
{retrieved_context}

Question: {question}

Return the stored answer, or "I don't know" if no entry matches."""
