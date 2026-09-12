# notebooks/

Eleven working notebooks, numbered in the order the project actually happened.
They are the primary record of *how* every result was produced; the reusable
core of their code is factored into `src/hyperbolic_icd10/`, and the analysis
that a reader most wants (regenerating tables, re-evaluating checkpoints) is in
`scripts/`.

**Portability.** Each notebook's first cell (tagged `portable-setup`, added
2026-09-12) resolves all paths from `HME_ROOT`. On Colab, mount Drive and set
`HME_ROOT` to your clone. Locally, run Jupyter from the repo root or
`export HME_ROOT=/path/to/repo`. Outputs are preserved as originally run — they
are a record, not a guarantee the cell reruns unchanged.

| # | Notebook | Role | Status |
|---|---|---|---|
| 01 | `01_phase1_data_pipeline.ipynb` | Download CMS zip, parse tabular XML into the 46,817-node tree, compute per-node features, save the processed artifacts. | Run-once, complete. Superseded for headless use by `scripts/build_tree.py`. |
| 02 | `02_phase2_baselines.ipynb` | Euclidean, Poincaré K=−1, learnable-K, mixed-curvature baselines at d=5/10/50, K=50. | Complete. Baselines, not the paper headline. |
| 03 | `03_phase3_novel_methods.ipynb` | Method 1 (tree-anchored curvature, line-integral distance) and Method 2 (alternating optimization). | Complete but best-epoch-selected — see `docs/known_issues.md` #1. |
| 04 | `04_phase3_levers.ipynb` | The "levers" explored to push Method 1's MAP up (ranking loss, base cloud, coupling). | Mostly negative results; kept for provenance. |
| 05 | `05_phase4_fair_comparison.ipynb` | Each method at its own best hyperparameters across d=2/5/10. | Complete. |
| 06 | `06_sandbox_convergence_and_dag.ipynb` | Convergence sweeps (the discovery that 0.775 was under-training) and DAG experiments. | Exploratory sandbox. |
| 07 | `07_position_dependent_curvature_and_sarkar.ipynb` | Curvature-in-the-metric, the structural audit of why it can't help here, **and the Sarkar construction + high-precision evaluation** (Tables 6–7). | Complete. Construction ported to `src/…/sarkar.py`. |
| 08 | `08_temperature_experiments.ipynb` | The clean temperature-confound stack (`train3`/`eval3`): source of Tables 2 & 4. | Canonical trainer; ported to `src/…/train.py`. |
| 09 | `09_paper_experiments.ipynb` | The paper experiments with a VAL/TEST split and a single evaluator: E1 (Table 1), E2 (Table 3), E3 (Table 5), Figure 1. | Canonical. Evaluator ported to `src/…/metrics.py`. |
| 10 | `10_paper_audit.ipynb` | Verifies every headline number reconciles from the archive; recomputes the τ threshold; lists text inconsistencies. | Audit only; logic ported to `scripts/make_tables.py` and `audit_threshold.py`. |
| 11 | `11_v2_closure_contest.ipynb` | Pre-registered test of whether position-dependent curvature earns an *unconfounded* closure win. | Overnight run harness. |

`archive/00_early_phase1_draft_2026-05-13.ipynb` is the original single-file
Phase-1 draft, kept for history.

## Which notebook produced which table

- Table 1 → 09 (`E1_selection.json`)
- Tables 2, 4 → 08 (`dim_runs/ms_const_*`, `msld_*`, `temp_*`)
- Table 3 → 09 (`E2_euclidean.json`, `E2_lr.json`)
- Table 5 → 09 (`E3_wordnet.json`)
- Tables 6, 7 → 07 (Sarkar construction; rerun via `scripts/run_sarkar.py`)

Regenerate all of them with `python scripts/make_tables.py`.
