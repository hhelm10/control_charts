"""Run one E0 fastsim cell, condense its outputs to a compact summary, delete the bulky run dir.

Usage: run_cell.py <config.yaml> <output_base> <summary_dir>
Summary JSON per cell: mean_known trajectory (every 10 steps), final per-agent known counts,
and per-snapshot probe-panel stats: IDK fraction and alive fraction (question answered
substantively by >=1 panel agent), each split static/temporal.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

IDK_TEXT = "I don't know"


def main():
    cfg, out_base, summary_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    name = Path(cfg).stem
    summary_dir.mkdir(parents=True, exist_ok=True)
    dest = summary_dir / f"{name}.json"
    if dest.exists():
        return

    repo = Path(__file__).resolve().parents[3]
    r = subprocess.run(
        [sys.executable, "-m", "controlcharts.fastsim", "run", cfg,
         "--data-path", str(Path(__file__).parent / "synthetic_qa.parquet"),
         "--output-base", str(out_base), "--panel-size", "100",
         "--skip-embed", "--skip-analysis"],
        cwd=repo, env={"PYTHONPATH": "src", "OMP_NUM_THREADS": "2", "PATH": "/usr/bin:/bin"},
        capture_output=True, text=True)
    if r.returncode != 0:
        print(f"FAIL {name}: {r.stderr[-2000:]}", file=sys.stderr)
        sys.exit(1)

    run_dirs = sorted(out_base.glob(f"{name}_fast_*"))
    run_dir = run_dirs[-1]
    rs = json.loads((run_dir / "results_summary.json").read_text())

    snaps = []
    for f in sorted(run_dir.glob("snapshots/snapshot_step_*_meta.json")):
        m = json.loads(f.read_text())
        tmask = m["temporal_mask"]
        resp = m["responses"]  # [panel][probe] strings
        n_panel, n_probe = len(resp), len(tmask)
        stats = {"step": m["step"]}
        for label, sel in (("all", [True] * n_probe), ("static", [not t for t in tmask]),
                           ("temporal", list(tmask))):
            idx = [j for j in range(n_probe) if sel[j]]
            if not idx:
                stats[f"idk_{label}"] = stats[f"alive_{label}"] = None
                continue
            idk = sum(resp[i][j] == IDK_TEXT for i in range(n_panel) for j in idx)
            alive = sum(any(resp[i][j] != IDK_TEXT for i in range(n_panel)) for j in idx)
            stats[f"idk_{label}"] = idk / (n_panel * len(idx))
            stats[f"alive_{label}"] = alive / len(idx)
        snaps.append(stats)

    hist = rs["history"]
    dest.write_text(json.dumps({
        "name": name, "config": rs["config"],
        "mean_known": [h["mean_known"] for h in hist[::10]],
        "mean_known_final": hist[-1]["mean_known"],
        "final_known_count": rs["final_known_count"],
        "n_probes": len(json.loads((run_dir / "snapshots" / "snapshot_step_0000_meta.json").read_text())["temporal_mask"])
                    if (run_dir / "snapshots" / "snapshot_step_0000_meta.json").exists() else None,
        "snapshots": snaps,
    }))
    shutil.rmtree(run_dir)


if __name__ == "__main__":
    main()
