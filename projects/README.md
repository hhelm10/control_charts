# Epistemic collapse — project layout

Working set for the epistemic-collapse paper (Helivan × Calcifer Computing),
organized out of `experiments/collapse/`. Run all scripts **from the repo
root** (e.g. `python projects/scripts/plot_books_costly.py`).

## artifacts/ — figures

- `figure1_final.png` — Fig 1: collapse in a real gpt-4o-mini network, 2×4 by factor (real data only).
- `figure2_final.png` — Fig 2: simulation ≈ real, per-factor calibration (zero fitted parameters).
- `figure_internal_policy_{overlay,agentlevel}.png` — internal: no-mimesis vs mimesis on shared panels.
- `figure1_mimesis_books.png`, `figure3_hyperparams_policies.png` — Fig 3/4 candidates (toy; pre-B=11 vocabulary).
- `fig_books_costly.png` — costly-written books vs environment speed, labeled in expected writes ω = p/W.
- `fig_books_W.png` — internal: hold p, vary W (freshness dial at constant tax).
- `fig_books_omega.png` — internal: ω-invariance certification (fixed ω, p varied 4–10×).
- `figure1_h2h_goodness.png` — head-to-head surrogate vs real at the baseline cell.

## scripts/

- `toy_v4.py` — the toy platform (crowding, answer policies, books incl. costly writing).
- `sweep_books_{W,omega}.py`, `plot_books_{costly,W,omega}.py` — books experiments (free compute).
- `plot_figure1_books.py`, `plot_figure3.py` — toy Fig 3/4 candidates.
- `fig1_real/` — real-system pipeline: `gen_fig1_configs.py` (run grids), `ecal_probe.py`
  (measured answering-gate table + m_profile), `ecal_invivo.py` (in-vivo validation),
  `analyze_fig1.py`, `compare_surrogate.py`, `plot_h2h.py`, `plot_figure1_final.py`,
  `plot_figure2.py`. Raw run snapshots stay in `experiments/results/` (~16 GB, not moved).
  LLM runs cost real money — quote and get sign-off before launching.

## data/

- `books_*_results.json`, `figure3_results.json` — toy sweep outputs.
- `ecal_table.json`, `m_profile.json` — measured gate P(response | context state) and
  pool-size → bucket-occupancy profile (inputs to the surrogate via `ECAL_TABLE`).
- `ecal_invivo_results*.json` — in-vivo gate validation.
- `configs_fig1v6/` (no-mimesis grid), `configs_fig1v7/` (mimesis grid) — the run configs
  behind Figures 1–2.
- Large external inputs live outside the repo in `~/helivan-chat-a100/projects/data/`:
  `nq_embedded.parquet` (question pool), `answer_alphabet.npz` (lookup embedding).

## writing/

- `summary.tex` / `summary.pdf` — two-page summary.
- `epistemic_collapse_experiment_design.md` — experiment design doc.
- `README.md` — collaborator handoff (paths inside predate this reorganization).

Legacy material (toy v1–v3, pilot, factor sweeps, logs, superseded figures) remains in
`experiments/collapse/`.
