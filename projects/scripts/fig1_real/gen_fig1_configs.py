"""Figure 1 (real system) spine grid: N=10, paper mechanics, one knob per setting.

Settings: baseline / Q in {200, 500} / decay in {0.2, 0.5} / K=1 / retrieval_k=1 /
mean degree in {2, 6} (ER, edge_probability = d/(N-1)). 3 seeds each, T=1000.
"""
from pathlib import Path
import yaml

HERE = Path(__file__).parent
CFG = HERE.parents[2] / "data" / "configs"
N = 10
T = 1000
SEEDS = [42, 43, 44]


def base_cfg():
    return {
        "experiment": {"name": "", "description": "Figure 1 real-system spine grid"},
        "data": {"total_questions": 50, "questions_per_agent": 4, "n_temporal": 10,
                 "temporal_change_probability": 0.04},
        "agents": {"count": N, "model": "gpt-4o-mini", "retrieval_k": 3, "use_llm": True},
        "network": {"topology": "full_mesh"},
        "simulation": {"max_iterations": T, "seed": 0, "questions_per_turn": 5,
                       "forget_strategy": {"strategy": "decay", "decay_coefficient": 0.05,
                                           "decay_mode": "multiplicative"},
                       "temporal_kernel": {"enabled": True, "interval": 10,
                                           "sample_size": 10, "n_nontemporal_sample": 10}},
        "control_bar": {"burn_in": 100, "window_size": 100, "k": 2},
    }


def settings():
    out = {"baseline": {}}
    for q in (200, 500):
        out[f"q{q}"] = {("data", "total_questions"): q,
                        ("data", "questions_per_agent"): int(0.8 * q / N),
                        ("data", "n_temporal"): q // 5}
    for dc in (0.2, 0.5):
        out[f"decay{dc}"] = {("simulation", "forget_strategy", "decay_coefficient"): dc}
    out["k1bw"] = {("simulation", "questions_per_turn"): 1}
    out["ret1"] = {("agents", "retrieval_k"): 1}
    for d in (2, 6):
        out[f"deg{d}"] = {("network", "topology"): "random",
                          ("network", "edge_probability"): round(d / (N - 1), 4)}
    return out


def main():
    CFG.mkdir(exist_ok=True)
    n = 0
    for name, mods in settings().items():
        for seed in SEEDS:
            c = base_cfg()
            c["experiment"]["name"] = f"fig1-{name}-s{seed}"
            c["simulation"]["seed"] = seed
            for path, v in mods.items():
                node = c
                for k in path[:-1]:
                    node = node[k]
                node[path[-1]] = v
            (CFG / f"fig1-{name}-s{seed}.yaml").write_text(yaml.safe_dump(c, sort_keys=False))
            n += 1
    print(f"wrote {n} configs to {CFG}")


if __name__ == "__main__":
    main()
