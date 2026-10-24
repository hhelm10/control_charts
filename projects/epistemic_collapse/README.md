# epistemic_collapse

Paper: epistemic collapse and epistemic capacity in multi-agent systems
(Helivan × Calcifer Computing). Follows the conventions in `../README.md`.

## Artifacts

| Artifact | Script | Caches (`data/`) |
|---|---|---|
| Figure 1 — failure over time, failure vs. size, epistemic capacity | `epistemic_collapse_figure1_capacity.py` | `figure1_trajectories.json`, `figure1_mirror.json`, `figure1_theory_size.json`, `figure1_theory_capacity.json` |
| Appendix A — proxy reproduces real gpt-4o-mini networks | `epistemic_collapse_appendixA_proxy_validation.py` | `appendixA_proxy_validation.json` |

Regenerate any artifact from the repo root (or anywhere):

```
.venv/bin/python projects/epistemic_collapse/scripts/epistemic_collapse_figure1_capacity.py
```

A script recomputes a cache only if it is missing (delete the JSON to force a
recompute); otherwise it only renders.

## Shared model code (`scripts/`)

- `epistemic_collapse_theory.py` — analytic model: renewal retention with the
  measured similarity distribution, birth–death chain per question, collapse
  and misalignment, capacity $M^\star(\varphi;\bar\lambda)$, mitigation strategies.
- `epistemic_collapse_mirror.py` — fast reimplementation of the repo pipeline's
  mechanics with the measured answering gate (the "proxy"), for sweeps.
- `epistemic_collapse_gate.py` — measures the answering gate (gpt-4o-mini;
  costs API credits) → `data/ecal_table.json`, `data/m_profile.json`.
- `epistemic_collapse_common.py` — paths, `save_artifact`, channel grammar.

## Channel grammar (this paper)

| Method | `hv_style` role | Color | Linestyle |
|---|---|---|---|
| no mitigation | `baseline_gray` | `#6b7280` | dashed |
| sharing secondhand (mimesis) | `comparator` | `#93aacc` | solid |
| books | `focus` | `#3596ff` | solid |
| mimesis + books | `anchor` | `#114471`, 3.6 pt | solid |

- **Linestyle = information source**: dashed = firsthand only; solid = uses
  secondhand information (peers' answers or books). Declared in the Fig. 1
  caption. Figure 1 contains only firsthand systems, so it uses a figure-wide
  solid/dashed pair for fast/slow forgetting, declared in its legend.
- **Violet (reserved concept) = epistemic failure φ**: all failures `#6d5bd0`,
  collapse `#43307e`, misalignment `#a795e3`.

## Data

- `nq_embedded.parquet`, `answer_alphabet.npz` — question pool and verbatim
  response alphabet. Raw inputs, kept committed here (an exception to the
  repo-level `data/` rule) so a clone runs without extra downloads.
- `ecal_table.json`, `m_profile.json` — measured answering gate.
- `validation_three_way.json` — proxy vs. mirror vs. theory (uniform selection).
- `mimesis_scan_mirror.json` — mimesis vs. none scan (Figure 2 design input).
- `configs/` — run configs behind Appendix A (`real_grid_firsthand`,
  `real_grid_mimesis`) and the three-way validation (`validation_uniform`).
  Their raw run directories live in the gitignored `experiments/results/`.
