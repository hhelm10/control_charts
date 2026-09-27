"""Command-line interface for controlcharts."""

import json
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime

import click
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

import matplotlib.pyplot as plt

from .config import Config, DEFAULT_SYSTEM_PROMPT, DEFAULT_PROMPT_TEMPLATE
from .database import VectorDatabase, QAPair, load_embeddings_cache
from .agent import Agent
from .network import Network
from .simulation import create_simulation
from .hooks import LoggingHook, CompositeHook
from .temporal_kernel import TemporalKernelHook
from .tdkps_analysis import run_tdkps_analysis, plot_perspective_variance, plot_combined_analysis
from .temporal_questions import TEMPORAL_QUESTIONS, get_temporal_answer

console = Console()
logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).parent.parent.parent.parent / "data" / "nq_embedded.parquet"


def plot_knowledge(hook_history: list[dict], output_path: Path, experiment_name: str) -> None:
    """Plot knowledge acquisition over time for each agent."""
    num_agents = len(hook_history[0]["agents"])

    # Extract known counts per agent over time
    steps = []
    agent_known = {i: [] for i in range(num_agents)}

    for entry in hook_history:
        steps.append(entry["step"])
        for agent_state in entry["agents"]:
            agent_known[agent_state["id"]].append(agent_state["known_count"])

    # Plot
    plt.figure(figsize=(10, 6))
    for agent_id in range(num_agents):
        plt.plot(steps, agent_known[agent_id], marker='o', markersize=3, label=f'Agent {agent_id}')

    plt.xlabel('Iteration')
    plt.ylabel('# Questions Known')
    plt.title(f'Knowledge Acquisition Over Time - {experiment_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


@click.group()
def main():
    """Control Charts: Multi-agent information flow simulation."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Suppress noisy HTTP request logging from openai/httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


@main.command()
@click.option("--remote/--local", default=True, help="Use Modal (remote) or local embedding")
@click.option("--max-samples", type=int, default=None, help="Limit number of samples")
@click.option("--output", type=click.Path(), default=None, help="Output parquet path")
def setup(remote: bool, max_samples: int | None, output: str | None):
    """Download and embed Natural Questions dataset."""
    from .setup import main as setup_main
    ctx = click.Context(setup_main)
    ctx.invoke(setup_main, remote=remote, max_samples=max_samples, output=output)


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--agents", type=int, default=None, help="Override agent count")
@click.option("--questions", type=int, default=None, help="Override total questions")
@click.option("--k", type=int, default=None, help="Override retrieval k")
@click.option("--topology", type=str, default=None, help="Override topology")
@click.option("--max-iterations", type=int, default=None, help="Override max iterations")
@click.option("--seed", type=int, default=None, help="Override random seed")
@click.option("--data-path", type=click.Path(exists=True), default=None, help="Path to embedded data")
@click.option("--output", type=click.Path(), default=None, help="Output JSON path for results")
@click.option("--verbose/--quiet", default=True, help="Verbose output")
def run(
    config_path: str,
    agents: int | None,
    questions: int | None,
    k: int | None,
    topology: str | None,
    max_iterations: int | None,
    seed: int | None,
    data_path: str | None,
    output: str | None,
    verbose: bool
):
    """Run an experiment from a config file."""
    # Load config
    config = Config.from_yaml(config_path)

    # Apply CLI overrides
    if agents is not None:
        config.agents.count = agents
    if questions is not None:
        config.data.total_questions = questions
    if k is not None:
        config.agents.retrieval_k = k
    if topology is not None:
        config.network.topology = topology
    if max_iterations is not None:
        config.simulation.max_iterations = max_iterations
    if seed is not None:
        config.simulation.seed = seed

    data_path = Path(data_path) if data_path else DEFAULT_DATA_PATH

    # Generate timestamp for this run
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create output directory with timestamp
    if output:
        # User specified output path - use its parent and create timestamp subfolder
        output_base = Path(output).parent
    else:
        output_base = Path("experiments/results")

    run_dir = output_base / f"{config.experiment.name}_{run_timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Copy the config file to the run directory
    config_source = Path(config_path)
    config_dest = run_dir / f"config_{config_source.name}"
    shutil.copy2(config_source, config_dest)

    console.print(f"\n[bold]Control Charts Experiment: {config.experiment.name}[/bold]")
    console.print(f"Description: {config.experiment.description}")
    console.print(f"Run directory: {run_dir}\n")

    # Load embedded data
    with console.status("Loading embedded QA pairs..."):
        all_questions, all_answers, all_embeddings = load_embeddings_cache(str(data_path))

    console.print(f"✓ Loaded {len(all_questions)} embedded QA pairs")

    # Initialize RNG
    rng = np.random.default_rng(config.simulation.seed)

    # Select questions in play (non-temporal)
    n_temporal = min(config.data.n_temporal, len(TEMPORAL_QUESTIONS))
    n_nontemporal = min(config.data.total_questions - n_temporal, len(all_questions))
    indices = rng.choice(len(all_questions), size=n_nontemporal, replace=False)

    questions_in_play = [all_questions[i] for i in indices]
    answers_in_play = [all_answers[i] for i in indices]
    embeddings_in_play = all_embeddings[indices]

    # Build question -> embedding mapping
    question_embeddings = {q: embeddings_in_play[i] for i, q in enumerate(questions_in_play)}

    console.print(f"✓ Selected {n_nontemporal} non-temporal questions in play")

    # Handle temporal questions if enabled
    temporal_questions_in_play = []
    temporal_embeddings = {}
    if n_temporal > 0:
        # Select temporal questions
        temporal_questions_in_play = TEMPORAL_QUESTIONS[:n_temporal]

        # Embed temporal questions using Modal
        console.print(f"Embedding {n_temporal} temporal questions...")
        from .embedding import embed_texts
        temporal_embs = embed_texts(temporal_questions_in_play)

        # Add temporal questions to questions_in_play and embeddings
        for i, tq in enumerate(temporal_questions_in_play):
            questions_in_play.append(tq)
            answers_in_play.append("0")  # Initial answer (iteration 0)
            question_embeddings[tq] = temporal_embs[i]
            temporal_embeddings[tq] = temporal_embs[i]

        console.print(f"✓ Added {n_temporal} temporal questions (total: {len(questions_in_play)} questions)")

    temporal_questions_set = set(temporal_questions_in_play)

    # Initialize temporal values dict (shared state for all agents and simulation)
    # Each temporal question starts with value 0, updated probabilistically each step
    temporal_values = {tq: 0 for tq in temporal_questions_in_play}

    # Get embedding dimension
    embedding_dim = embeddings_in_play.shape[1]

    # Create agents
    agents_list = []
    questions_per_agent = config.data.questions_per_agent

    # Build custom prompts lookup
    custom_prompts = {c.id: c for c in config.agents.custom}

    # Distribute temporal question ownership - each agent owns some temporal questions
    # Round-robin assignment: agent i owns temporal questions where index % n_agents == i
    temporal_ownership = {i: set() for i in range(config.agents.count)}
    for tq_idx, tq in enumerate(temporal_questions_in_play):
        owner_agent = tq_idx % config.agents.count
        temporal_ownership[owner_agent].add(tq)

    # Shuffle and distribute non-temporal questions to agents
    n_total_questions = len(questions_in_play)
    shuffled_indices = rng.permutation(n_nontemporal)  # Only shuffle non-temporal indices

    for i in range(config.agents.count):
        # Get adversarial config if specified
        adversarial_schedule = None
        defection_schedule = None
        adversarial_system_prompt = None
        adversarial_prompt_template = None

        if i in custom_prompts:
            custom = custom_prompts[i]
            # Parse adversarial schedule: [[t, p], ...] -> [(t, p), ...]
            if custom.adversarial_schedule:
                adversarial_schedule = [
                    (int(entry[0]), float(entry[1]))
                    for entry in custom.adversarial_schedule
                ]
            # Parse sigmoid defection schedule
            defection_schedule = None
            if custom.defection_schedule:
                defection_schedule = {
                    "start": custom.defection_schedule.start,
                    "duration": custom.defection_schedule.duration,
                    "max_p": custom.defection_schedule.max_p,
                    "shape": custom.defection_schedule.shape,
                }
            # Adversarial prompts (used when behaving adversarially)
            if custom.system_prompt:
                adversarial_system_prompt = custom.system_prompt
            if custom.prompt_template:
                adversarial_prompt_template = custom.prompt_template

        # Create agent with empty database
        # Normal behavior uses default prompts; adversarial behavior uses custom prompts
        agent = Agent(
            id=i,
            database=VectorDatabase(dimension=embedding_dim),
            model=config.agents.model,
            retrieval_k=config.agents.retrieval_k,
            answer_policy=config.agents.answer_policy,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            prompt_template=DEFAULT_PROMPT_TEMPLATE,
            adversarial_schedule=adversarial_schedule,
            defection_schedule=defection_schedule,
            adversarial_system_prompt=adversarial_system_prompt,
            adversarial_prompt_template=adversarial_prompt_template,
            forget_strategy=config.simulation.forget_strategy.strategy,
            decay_coefficient=config.simulation.forget_strategy.decay_coefficient,
            decay_mode=config.simulation.forget_strategy.decay_mode,
            use_llm=config.agents.use_llm,
            propagation_probability=config.agents.propagation_probability,
            cross_question_propagation=config.agents.cross_question_propagation
        )

        # Assign initial knowledge (only non-temporal questions)
        start_idx = i * questions_per_agent
        end_idx = min(start_idx + questions_per_agent, n_nontemporal)
        agent_indices = shuffled_indices[start_idx:end_idx]

        qa_pairs = [
            QAPair(
                question=questions_in_play[j],
                answer=answers_in_play[j],
                embedding=embeddings_in_play[j],
                is_temporal=False
            )
            for j in agent_indices
        ]
        agent.initialize_knowledge(qa_pairs)

        # Set temporal questions and ownership (pass shared temporal_values dict).
        # With the environment-query channel on, ownership is disabled: no agent
        # knows the truth without consulting the environment.
        agent.set_temporal_questions(
            temporal_questions_set,
            set() if config.simulation.env_queries_per_turn > 0 else temporal_ownership[i],
            temporal_values=temporal_values
        )

        agents_list.append(agent)

    console.print(f"✓ Created {len(agents_list)} agents with ~{questions_per_agent} non-temporal QA pairs each")
    if n_temporal > 0:
        console.print(f"✓ Distributed {n_temporal} temporal questions across agents (each agent owns ~{n_temporal // config.agents.count} temporal questions)")

    # Create network
    network = Network.create(
        agents=agents_list,
        topology=config.network.topology,
        edge_probability=config.network.edge_probability,
        adjacency_matrix=config.network.adjacency,
        rng=rng
    )
    console.print(f"✓ Created {config.network.topology} network topology")

    # Create iteration hooks
    logging_hook = LoggingHook(verbose=verbose)
    hooks = [logging_hook]

    # Add temporal kernel hook if enabled
    temporal_kernel_hook = None
    if config.simulation.temporal_kernel.enabled:
        # Snapshots go in the run directory
        snapshots_dir = run_dir / "snapshots"

        # Determine sample size for non-temporal questions
        n_nontemporal_sample = config.simulation.temporal_kernel.n_nontemporal_sample
        if n_nontemporal_sample == 0:
            # Fallback to sample_size for backwards compatibility
            n_nontemporal_sample = config.simulation.temporal_kernel.sample_size

        temporal_kernel_hook = TemporalKernelHook(
            interval=config.simulation.temporal_kernel.interval,
            sample_size=n_nontemporal_sample,  # This is now for non-temporal only
            questions_in_play=questions_in_play,
            question_embeddings=question_embeddings,
            output_dir=snapshots_dir,
            seed=config.simulation.seed,
            temporal_questions=temporal_questions_in_play,  # Pass temporal questions
            temporal_values=temporal_values,  # Shared dict for tracking current values
        )
        hooks.append(temporal_kernel_hook)
        if n_temporal > 0:
            console.print(f"✓ Temporal kernel hook enabled (interval={config.simulation.temporal_kernel.interval}, "
                         f"all {n_temporal} temporal + {n_nontemporal_sample} non-temporal questions)")
        else:
            console.print(f"✓ Temporal kernel hook enabled (interval={config.simulation.temporal_kernel.interval}, "
                         f"sample_size={n_nontemporal_sample})")

    # Combine hooks
    hook = CompositeHook(hooks)

    # Create simulation
    sim = create_simulation(
        network=network,
        questions_in_play=questions_in_play,
        question_embeddings=question_embeddings,
        seed=config.simulation.seed,
        iteration_hook=hook,
        questions_per_turn=config.simulation.questions_per_turn,
        temporal_values=temporal_values,
        temporal_change_probability=config.data.temporal_change_probability,
        env_queries_per_turn=config.simulation.env_queries_per_turn,
        static_answers=dict(zip(questions_in_play, answers_in_play)),
        question_lambdas=(
            {q: float(config.data.lambda_mean *
                      np.exp(config.data.lambda_disp * rng.standard_normal()
                             - config.data.lambda_disp ** 2 / 2))
             for q in questions_in_play}
            if config.data.lambda_mean > 0 else None),
    )

    # Give the snapshot hook access to the environment's current truth
    for h in hooks:
        if hasattr(h, "truth_fn"):
            h.truth_fn = sim.current_truth

    # Run simulation
    console.print(f"\n[bold]Running simulation for {config.simulation.max_iterations} iterations...[/bold]\n")

    start_time = time.time()
    results = sim.run(max_iterations=config.simulation.max_iterations)
    elapsed_time = time.time() - start_time

    # Summary
    console.print("\n[bold]Simulation Complete![/bold]")
    console.print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}m)")

    # Finalize temporal kernel (batch embed all responses) and run analysis
    if temporal_kernel_hook is not None:
        console.print("\n[bold]Finalizing temporal kernel snapshots...[/bold]")
        temporal_kernel_hook.finalize()
        console.print(f"✓ Temporal kernel snapshots finalized")

        index_path = temporal_kernel_hook.save_index()
        console.print(f"✓ Temporal kernel snapshots index saved to {index_path}")

        # Run TDKPS analysis and generate combined figure
        try:
            console.print("\n[bold]Running TDKPS analysis...[/bold]")
            run_tdkps_analysis(
                snapshots_dir=snapshots_dir,
                output_dir=run_dir,
                experiment_name=config.experiment.name,
            )
            console.print(f"✓ TDKPS analysis complete")

            # Generate combined analysis figure
            console.print("\n[bold]Generating combined analysis figure...[/bold]")
            plot_combined_analysis(
                output_dir=run_dir,
                experiment_name=config.experiment.name,
                snapshots_dir=snapshots_dir,
                config_path=config_dest,
            )
            console.print(f"✓ Combined analysis figure complete")
        except Exception as e:
            console.print(f"[yellow]⚠ TDKPS analysis failed: {e}[/yellow]")
            console.print("[yellow]Continuing to save results...[/yellow]")

    # Always save results to the run directory
    output_path = run_dir / "results.json"

    output_data = {
        "config": config.model_dump(),
        "timestamp": datetime.now().isoformat(),
        "elapsed_time_seconds": elapsed_time,
        "run_directory": str(run_dir),
        "results": results,
        "final_states": [agent.get_state() for agent in agents_list],
        "hook_history": logging_hook.history
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)

    console.print(f"✓ Results saved to {output_path}")

    # Generate knowledge plot
    knowledge_plot_path = run_dir / f"{config.experiment.name}_knowledge.png"
    plot_knowledge(logging_hook.history, knowledge_plot_path, config.experiment.name)
    console.print(f"✓ Knowledge plot saved to {knowledge_plot_path}")

    console.print(f"\n[bold green]All outputs saved to: {run_dir}[/bold green]")


@main.command("regenerate-figures")
@click.argument("results_dir", type=click.Path(exists=True))
@click.option("--skip-tdkps", is_flag=True, help="Skip TDKPS analysis (requires maps package)")
@click.option("--skip-knowledge", is_flag=True, help="Skip knowledge acquisition plot")
@click.option("--skip-combined", is_flag=True, help="Skip combined analysis figure")
def regenerate_figures(
    results_dir: str,
    skip_tdkps: bool,
    skip_knowledge: bool,
    skip_combined: bool,
):
    """Regenerate figures from cached experiment results.

    RESULTS_DIR is the path to an experiment results directory
    (e.g., experiments/results/adversarial-game_2026-01-06_13-21-26)
    """
    results_path = Path(results_dir)

    # Load results.json
    results_json_path = results_path / "results.json"
    if not results_json_path.exists():
        console.print(f"[red]Error: No results.json found in {results_path}[/red]")
        raise SystemExit(1)

    with open(results_json_path) as f:
        data = json.load(f)

    experiment_name = data["config"]["experiment"]["name"]
    console.print(f"\n[bold]Regenerating figures for: {experiment_name}[/bold]")
    console.print(f"Results directory: {results_path}")

    # Show elapsed time if available
    if "elapsed_time_seconds" in data:
        elapsed = data["elapsed_time_seconds"]
        console.print(f"Original run time: {elapsed:.1f}s ({elapsed/60:.1f}m)\n")
    else:
        console.print()

    # Regenerate knowledge plot from hook_history
    if not skip_knowledge:
        if "hook_history" in data and data["hook_history"]:
            console.print("[bold]Regenerating knowledge acquisition plot...[/bold]")
            knowledge_plot_path = results_path / f"{experiment_name}_knowledge.png"
            plot_knowledge(data["hook_history"], knowledge_plot_path, experiment_name)
            console.print(f"✓ Knowledge plot saved to {knowledge_plot_path}")
        else:
            console.print("[yellow]⚠ No hook_history in results.json, skipping knowledge plot[/yellow]")

    # Check for snapshots directory
    snapshots_dir = results_path / "snapshots"
    if not snapshots_dir.exists():
        console.print("[yellow]⚠ No snapshots directory found, skipping TDKPS analysis[/yellow]")
    else:
        # Regenerate TDKPS analysis (computes embeddings needed for combined figure)
        if not skip_tdkps:
            console.print("\n[bold]Regenerating TDKPS analysis...[/bold]")
            try:
                run_tdkps_analysis(
                    snapshots_dir=snapshots_dir,
                    output_dir=results_path,
                    experiment_name=experiment_name,
                )
                console.print(f"✓ TDKPS analysis complete")
            except Exception as e:
                console.print(f"[yellow]⚠ TDKPS analysis failed: {e}[/yellow]")

        # Regenerate combined analysis figure
        if not skip_combined:
            console.print("\n[bold]Regenerating combined analysis figure...[/bold]")
            # Find the config file in the results directory
            config_files = list(results_path.glob("config_*.yaml"))
            config_path = config_files[0] if config_files else None
            plot_combined_analysis(
                output_dir=results_path,
                experiment_name=experiment_name,
                snapshots_dir=snapshots_dir,
                config_path=config_path,
            )
            console.print(f"✓ Combined analysis figure complete")

    console.print(f"\n[bold green]Figure regeneration complete![/bold green]")


if __name__ == "__main__":
    main()
