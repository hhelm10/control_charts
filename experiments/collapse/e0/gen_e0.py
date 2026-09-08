"""E0 setup: synthetic QA parquet + per-cell fastsim configs (design doc §7 E0).

Q sweep in the UNMODIFIED simulator. Cells: Q x N x seed, K=5, decay forgetting,
n_temporal = 0.2*Q, questions_per_agent = 0.8*Q/N (each static question seeded
exactly once network-wide), T=2000. N=100 cells with Q<=100 are dropped:
questions_per_agent would be <1 and nothing is ever seeded.
"""
import sys
from pathlib import Path

import pandas as pd
import yaml

HERE = Path(__file__).parent
CFG_DIR = HERE / "configs"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "synthetic_qa.parquet"

QS = [50, 100, 200, 500, 1000, 2000]
NS = [5, 100]
SEEDS = range(42, 72)  # 30 seeds
T = 2000


def make_parquet():
    n = 1600  # >= max static count: 0.8 * 2000
    df = pd.DataFrame({
        "question": [f"synthetic question {i:04d}?" for i in range(n)],
        "answer": [f"synthetic answer {i:04d}" for i in range(n)],
    })
    df.to_parquet(DATA)
    print(f"wrote {DATA} ({n} rows)")


def cell_cfg(N, Q, seed):
    return {
        "experiment": {"name": f"e0-N{N}-q{Q}-s{seed}", "description": "E0 Q sweep, unmodified fastsim"},
        "data": {
            "total_questions": Q,
            "questions_per_agent": int(0.8 * Q / N),
            "n_temporal": Q // 5,
            "temporal_change_probability": 0.04,
        },
        "agents": {
            "count": N,
            "model": "gpt-4o-mini",
            "retrieval_k": 3,
            "use_llm": False,
            "propagation_probability": 0.0,
            "cross_question_propagation": 0.0,
            "custom": [],
        },
        "network": {"topology": "full_mesh"},
        "simulation": {
            "max_iterations": T,
            "seed": seed,
            "questions_per_turn": 5,
            "forget_strategy": {"strategy": "decay", "decay_coefficient": 0.05, "decay_mode": "additive"},
            "temporal_kernel": {"enabled": True, "interval": 10, "sample_size": 10, "n_nontemporal_sample": 10},
        },
        "control_bar": {"burn_in": 100, "window_size": 200, "k": 2},
    }


def main():
    make_parquet()
    CFG_DIR.mkdir(exist_ok=True)
    n = 0
    for N in NS:
        for Q in QS:
            if int(0.8 * Q / N) < 1:
                print(f"skip N={N} Q={Q}: questions_per_agent < 1 (nothing would be seeded)")
                continue
            for seed in SEEDS:
                cfg = cell_cfg(N, Q, seed)
                (CFG_DIR / f"{cfg['experiment']['name']}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
                n += 1
    print(f"wrote {n} configs to {CFG_DIR}")


if __name__ == "__main__":
    main()
