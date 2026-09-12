# results/

Every experiment output. The **canonical** numbers — the ones behind the paper
draft — are the `paper_experiments/` JSONs and the seeded `dim_runs/*.pt`
checkpoints; everything else is either a control, an input, or a superseded run
kept for provenance. `checkpoint_manifest.csv` indexes every `.pt`/`.npy` with
its stored metrics and the notebook that produced it. Regenerate it with
`scripts/make_checkpoint_manifest.py`.

`tables/` holds the paper tables regenerated from these files by
`scripts/make_tables.py`; delete and rebuild them any time.

## Canonical (paper) sources

| Folder / file | Contents | Notebook |
|---|---|---|
| `paper_experiments/E1_selection.json` | Table 1: ICD-10 temperature sweep, const curvature, d=10, 5 τ × 3 seeds, full trajectories | 09 |
| `paper_experiments/E2_euclidean.json` | Table 3: Euclidean control (temperature has no effect) | 09 |
| `paper_experiments/E2_lr.json` | Table 3: learning-rate controls (effect is not an lr artifact) | 09 |
| `paper_experiments/E3_wordnet.json` | Table 5: replication on the WordNet noun forest | 09 |
| `paper_experiments/B_beta0.json` | β=0 ablation (effect survives dropping the hierarchy term), n=1 | 09 |
| `paper_experiments/A_fullgraph_map.json` | Full-graph vs sampled MAP — the gap the paper must disclose (see `docs/known_issues.md` #6) | 09 |
| `paper_experiments/D_geometry_stats.json` | Boundary fraction, sibling angle vs τ (mechanism check) | 09 |
| `paper_experiments/figure1_decoupling.*` | Figure 1 (also copied to `figures/`) | 09 |
| `dim_runs/ms_const_x*_seed*.pt` | d=10 temperature sweep, 3 seeds — Tables 2 & 4 | 08 |
| `dim_runs/msld_{const,graded}_d{2,5}_x*_seed*.pt` | d=2 and d=5 sweeps, 3 seeds — Table 4 | 08 |
| `dim_runs/temp_{euclid,graded}_x1.0.pt` | Euclidean & position-dependent rows of Table 2 (n=1) | 08 |
| `dim_runs/sarkar_tau*_{r,th}.npy` | Sarkar 2D construction, polar coords — Table 6 | 07 |
| `dim_runs/sark_d{2,5}_t*_{r,D}.npy` | Sarkar/De Sa d-dimensional construction | 07 |

Table 6 has no persisted metrics JSON in the original archive; regenerate the
numbers (and a `paper_experiments/table6_sarkar_sweep.json`) with
`scripts/run_sarkar.py`.

## Controls, ablations, and secondary datasets

| Folder | Contents |
|---|---|
| `phase2_baselines/` | Phase-2 baseline sweep JSONs (Euclidean, Poincaré K=−1, learnable K, mixed-curvature) at d=5/10/50, K=50. Superseded as the paper headline but the recipe's provenance. |
| `phase3_method1/`, `phase3_method1_exp/`, `phase3_method1_exp_v4/` | Tree-anchored curvature (Method 1), α sweeps on the 2,000-edge fixed eval — **best-epoch-selected; see `docs/known_issues.md` #1**. |
| `phase3_method1_ablation/` | Feature-anchoring ablation (real vs shuffled vs random vs binary κ). |
| `phase3_method1_desat/`, `phase3_method2/` | κ desaturation sweep; Method 2 (alternating optimization). |
| `phase4_*` | Fair-comparison grids across d and methods (revised and standardized versions). |
| `dim_runs/go_*`, `dim_runs/gp_*`, `dim_runs/flip_results.json`, `dim_runs/go_pos_results.json` | Gene Ontology (second graph): temperature sweep and positional-field arms. |
| `dim_runs/normpen_*`, `dim_runs/norm_const_*` | Norm-penalty (boundary-escape) causal test — the mechanism that was tested and *falsified*. |
| `dim_runs/lr_*`, `dim_runs/negK*`, `dim_runs/sub_*` | lr, negative-count, and sub-unit-temperature sensitivity. |
| `v2_closure_contest/` | The pre-registered "does curvature earn an unconfounded closure win" contest (Phase A checkpoint reports). |
| `eval_sets/` | The frozen Phase-3 fixed evaluation edge sets. |

## Inputs, not results

`data/processed/curvature_node_features_altorder.csv` (per-node curvature
features — **in a different node order than the checkpoints**; the canonical
order is `data/processed/icd10_nodes.csv`), `dim_runs/kappa_*.npy`, `dim_runs/c_node.npy`,
`dim_runs/ktest_*.npy` are curvature field vectors fed into training.

## File-naming scheme

`{family}_{mode}_[d{dim}]_x{scale}_[seed{n}].pt`, where `x{scale}` is the loss
temperature τ (the paper's scale hyperparameter) and `mode ∈ {const, graded,
euclid}`. `graded` = per-node multiplicative curvature (a scale change, see
known issues). `msld` = multi-seed low-dim; `ms` = multi-seed d=10; `temp` =
single-seed; `d5hi`/`hi` = high-τ runs; `sarkar`/`sark` = combinatorial
construction. Each `.pt` carries its own `metrics` dict computed by the notebook
08 / 09 evaluator, which `src/hyperbolic_icd10/metrics.py` reproduces to ~1e-7.
