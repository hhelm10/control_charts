# epistemic_collapse

Paper: epistemic collapse and epistemic capacity in multi-agent systems
(Helivan × Calcifer Computing). Follows the conventions in `../README.md`.

## Artifacts

| Artifact | Script | Caches (`data/`) |
|---|---|---|
| Figure 1 — example dynamics, failure vs. size, epistemic capacity vs. tolerance φ | `epistemic_collapse_figure1_capacity.py` | `figure1_trajectories.json`, `figure1_mirror.json`, `figure1_theory_size.json`, `figure1_theory_capacity.json` |
| Figure 2 — sensitivity of capacity to λ̄, c, E, N, k, peer degree (no mitigation); real-system points | `epistemic_collapse_figure2_sensitivity.py` | `figure2_theory.json`, `figure2_mirror.json`, `figure2_real.json` |
| Figure 3 — relative capacity Γ of mimesis across the Figure 2 parameters (simulation) | `epistemic_collapse_figure3_mimesis.py` | `figure3_mirror.json` |
| Figure 4 — relative capacity Γ of books (simulation) | `epistemic_collapse_figure4_books.py` | `figure4_mirror.json` |
| Matched validation — proxy replays each real run on its own question set (Figs 1–2) | `epistemic_collapse_matched_validation.py` | `figure2_proxy_matched.json`, `figure1_proxy_matched_ss.json` |
| Appendix A — proxy vs real gpt-4o-mini on four metrics, all 790 Fig 1–2 runs (matched) | `epistemic_collapse_appendixA_proxy_validation.py` | reads the matched-validation caches |

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
  mechanics (the "proxy"), for sweeps, with the measured context-dependent
  refusal (`refusal_mode="nother"`, default).
- `epistemic_collapse_real.py` — reads real runs (`experiments/results/`).
- Refusal measurements (gpt-4o-mini; cost API credits):
  `epistemic_collapse_measure_refusal.py` (whole pool, k = 3) and
  `epistemic_collapse_measure_refusal_k.py K` (k = 1, 7) → `data/refusal_oai*.json`;
  `epistemic_collapse_gate_invivo.py [k]` and
  `epistemic_collapse_refusal_context.py` fit the in-vivo model from LLM logs of
  real runs (`LLM_DUMP`; `data/llm_gate_dumps/`, not committed) →
  `data/gate_invivo*.json`, `data/refusal_context.json` (refusal table per
  retrieval depth k = 1, 3; other k use k = 3).
- `epistemic_collapse_gate.py` — the earlier synthetic-context gate
  (→ `data/ecal_table.json`, `data/m_profile.json`; still used for k ∉ {1, 3}
  by the legacy refusal modes).
- `epistemic_collapse_common.py` — paths, `save_artifact`, channel grammar.

## Channel grammar (this paper)

| Method | `hv_style` role | Color | Linestyle |
|---|---|---|---|
| no mitigation | `baseline_gray` | `#6b7280` | solid |
| sharing secondhand (mimesis) | `comparator` | `#93aacc` | dashed |
| books | `focus` | `#3596ff` | dashed |
| mimesis + books | `anchor` | `#114471`, 3.6 pt | dashed |

- **Linestyle = information source**: solid = firsthand only; dashed = uses
  secondhand information (peers' answers or books). Declared in the Fig. 1
  caption and honored everywhere (Figures 1–2 are all solid). This overrides
  the `hv_style` role linestyles (`LINESTYLE` in `epistemic_collapse_common.py`).
- **Line vs. marker = source of the number**: line = theory, marker =
  simulation (mirror).
- **Failure components (project exception to the canon)**: matplotlib Set1, chosen
  so the components "add" — collapse red `#e41a1c` + misalignment blue `#377eb8`
  = all failures purple `#984ea3`. Violet (`reserve`) is unassigned.

## Data

- `nq_embedded_oai.parquet` — the question pool (25k NQ, OpenAI
  text-embedding-3-small), used by every model since 2026-10-28; built by
  `epistemic_collapse_build_pool_oai.py`. Kept committed (an exception to the
  repo-level `data/` rule) so a clone runs without extra downloads. Its verbatim
  response alphabet `answer_alphabet_oai.npz` (89 MB, needed only for real runs)
  is not committed: rebuild it with the same script.
- `nq_embedded.parquet`, `answer_alphabet.npz` — the earlier nomic pool (`EC_POOL=nomic`).
- `figure1_real.json`, `figure2_real.json` — real-system (gpt-4o-mini) failure,
  per run: [collapse, misalignment], steady state t ≥ 50 of T = 100.
- `refusal_oai.json`, `refusal_oai_k1.json`, `refusal_oai_k7.json`,
  `gate_invivo*.json`, `refusal_context.json` — measured refusal.
- `ecal_table.json`, `m_profile.json` — measured answering gate.
- `validation_three_way.json` — proxy vs. mirror vs. theory (uniform selection).
- `mimesis_scan_mirror.json` — mimesis vs. none scan (Figure 3 design input).
- `configs/` — run configs of the earlier surrogate validation (`real_grid_firsthand`,
  `real_grid_mimesis`), the three-way validation (`validation_uniform`), the
  real Figure 1 and 2 runs (`real_fig1`, `real_fig2`) and the logged k = 1
  runs (`real_gate`).
  Their raw run directories live in the gitignored `experiments/results/`.
