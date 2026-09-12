# Hyperbolic Embeddings of ICD-10-CM, and the Loss-Temperature Confound in Reconstruction MAP

This repository embeds the ICD-10-CM diagnostic ontology (46,817 nodes, 46,816
edges) into hyperbolic space and studies how the field's standard reconstruction
metric behaves. The headline result is a **measurement**, not a new
state-of-the-art embedding method:

> On this 46,817-node medical ontology, a single scale hyperparameter (the
> temperature of the training objective) moves reconstruction MAP from **0.78 to
> 0.99** while distortion stays flat near **0.11**, and over the same range
> accuracy on hierarchical relations *withheld from training* falls (closure MRR
> 0.134 → 0.042). A Sarkar combinatorial construction on the same graph reaches
> MAP **1.00** at distortion **0.019** in two dimensions. Reconstruction MAP,
> reported without the temperature at which it was obtained, therefore cannot
> support the cross-method comparisons the hyperbolic-embedding literature
> routinely draws from it.

This finding **extends** Sala, De Sa, Gu & Ré (ICML 2018), who already published
the scale/precision tradeoff in the *constructive* setting. See
[`docs/related_work_positioning.md`](docs/related_work_positioning.md) for exactly
what is theirs and what is new here, and read
[`docs/known_issues.md`](docs/known_issues.md) before citing any number.

## What's in here

```
src/hyperbolic_icd10/   Installable package: data loading, fixed eval sets,
                        metrics, trainer, Sarkar construction, checkpoint I/O.
scripts/                Command-line entry points (download, build, train,
                        regenerate every paper table, re-evaluate a checkpoint).
notebooks/              The 11 working notebooks, chronological, Colab-portable.
data/                   Processed node table (tracked); raw CMS zip + pickle
                        (git-ignored, regenerable — see below).
results/                All experiment outputs (JSON + saved coordinates) and a
                        checkpoint manifest. Large tensors are release assets.
models/                 Trained checkpoints (git-ignored; shipped as a release).
figures/                Figure 1 (the decoupling plot) and the branching histogram.
docs/                   Paper draft, proposals, related-work positioning, the
                        error ledger, and the known-issues list.
tests/                  Reproduction checks (pytest).
```

## Quickstart

```bash
git clone <this repo> && cd hyperbolic-icd10
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # core: numpy, pandas, mpmath (CPU-only)

# The processed node table is tracked, so this works immediately:
python -c "from hyperbolic_icd10 import load_tree; t=load_tree(); print(t.N, len(t.edges_idx))"
# -> 46817 46816

# Regenerate every paper table from the archived results (no GPU, no torch):
python scripts/make_tables.py

# Reproduce the construction reference point (Table 6), ~20 s per tau on CPU:
python scripts/run_sarkar.py --taus 5.0 8.0 12.0

# Re-evaluate any saved embedding with the exact archived protocol:
python scripts/reevaluate_checkpoint.py results/dim_runs/ms_const_x1.0_seed0.pt
```

Training needs a GPU and the optional extras (`pip install -e ".[train]"`, which
pulls `torch` and `geoopt==0.5.1`). Every archived run used an A100 on Colab
Pro+; a single run is ~1,500 epochs at ~30 min.

## Reproducing from raw data

The processed node table (`data/processed/icd10_nodes.csv`) is tracked, so most
of the repo runs without the raw files. To rebuild everything from source:

```bash
python scripts/download_icd10.py --extract     # FY2025 CMS zip -> data/raw/
python scripts/build_tree.py --pickle          # XML -> icd10_nodes.csv + pickle
python scripts/make_legacy_pickle.py           # (alternative: pickle from the CSV, no XML)
```

`build_tree.py` asserts the tree comes out at 46,817 / 46,816; a different count
means the CMS file version changed.

## Large files (checkpoints, raw data)

To keep the git history usable, ~420 MB of trained checkpoints (`models/`,
`results/dim_runs/*.pt`, `results/**/*.npy`) and the 137 MB raw ICD-10 download
are **not** committed. They are published as a GitHub Release / Zenodo archive;
run `scripts/package_release.py` to rebuild that archive from a full working
tree. Everything needed to *reproduce* the checkpoints (code, seeds, configs) is
in git; the checkpoints themselves are a convenience.

## The dataset

FY2025 ICD-10-CM Code Tables, parsed from the CMS tabular XML
(<https://www.cms.gov/files/zip/2025-code-tables-tabular-and-index.zip>, U.S.
government work, public domain). Structure, verified in
`notebooks/10_paper_audit.ipynb`: 46,817 nodes, 46,816 edges (N−1 confirms it is
a tree), 10,775 internal / 36,042 leaves, max depth 7, max branching 34, mean
branching 4.34 over internal nodes. The node **index order** used by every
checkpoint is fixed by the Phase-1 parser and stored in the `idx` column of the
node table — do not re-sort it.

## Citing

See [`CITATION.cff`](CITATION.cff). This is a preprint-stage research artifact;
the paper draft lives in [`docs/paper/`](docs/paper/) and is not peer-reviewed.

## License

Code: MIT ([`LICENSE`](LICENSE)). The ICD-10-CM source data is a U.S. government
work (public domain). Text documents under `docs/` are the author's and are
shared for transparency, not licensed for reuse.
