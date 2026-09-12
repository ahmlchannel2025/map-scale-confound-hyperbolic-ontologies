#!/usr/bin/env python3
"""Index every checkpoint (.pt) and saved embedding (.npy) with its stored metrics.

Writes results/checkpoint_manifest.csv.  Needs no torch.  Re-run after adding runs.
"""
import csv, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10.config import ROOT, RESULTS, MODELS
from hyperbolic_icd10.checkpoints import peek

FAMILIES = [  # (regex on file stem, family, produced by)
    (r"^msld_(const|graded)_d\d+_x", "msld: multi-seed low-dim temperature sweep (Table 4, d=2/5)", "notebooks/08"),
    (r"^ms_const_x", "ms: multi-seed d=10 temperature sweep (Tables 2, 4)", "notebooks/08"),
    (r"^mt_const", "mt: duplicate d=10 tau=1 seeds (superseded by ms_*)", "notebooks/08"),
    (r"^temp_(const|graded|euclid)_x", "temp: single-seed d=10 temperature sweep (Table 2 Euclid / pos-dep rows)", "notebooks/08"),
    (r"^d5hi_const", "d5hi: d=5 at high tau (8, 12)", "notebooks/08"),
    (r"^hi_const", "hi: d=10 at high tau", "notebooks/08"),
    (r"^dimtemp_", "dimtemp: early single-seed dimension x temperature grid", "notebooks/07"),
    (r"^(const|graded|euclid|learned|kernelfield|kernelreweight|deeperlearned_lineint)_d\d+", "phase-4 converged runs (lr 50, long training)", "notebooks/06"),
    (r"^gp_(const|flip|ctrl)_x", "gp: Gene Ontology positional-field arms (const / flipped / control), 3 seeds", "notebooks/07"),
    (r"^go_const_x", "go: Gene Ontology temperature sweep", "notebooks/08"),
    (r"^pos_d\d+_x", "pos: per-node curvature in the metric (position_dependent_curvature)", "notebooks/07"),
    (r"^sub_const_x", "sub: sub-unit temperatures", "notebooks/08"),
    (r"^negK\d+", "negK: negative-count sensitivity", "notebooks/08"),
    (r"^normpen_b", "normpen: norm-penalty (boundary-escape) causal test", "notebooks/08"),
    (r"^norm_const", "norm: norm-penalty control", "notebooks/08"),
    (r"^lr_", "lr: learning-rate controls (Table 3)", "notebooks/08"),
    (r"^sarkar_tau", "Sarkar 2-D construction, polar coords (Table 6)", "notebooks/07"),
    (r"^sark_d\d+_t", "Sarkar/De Sa d-dimensional construction", "notebooks/07"),
    (r"^kappa_|^ktest_|^c_node", "curvature feature vectors (inputs, not results)", "notebooks/07"),
    (r"^pos_A_full", "Block A: positions for full-graph per-node MAP", "notebooks/09"),
    (r"^(euclidean|poincare|mixed_curvature)_", "Phase 2 baseline (K=50, 300 epochs)", "notebooks/02"),
    (r"^alpha_sweep", "Phase 3 Method 1 (tree-anchored, line-integral) alpha sweep", "notebooks/03"),
    (r"^exp_sweep|^exp_fixed|^sweep_phase2ref|^phase2ref", "Phase 3 Method 1, exp coupling / Phase-2 reference cloud", "notebooks/03-04"),
    (r"^method2", "Phase 3 Method 2 (alternating optimisation)", "notebooks/03"),
    (r"^sarkar_d2_reference", "Sarkar 2-D reference embedding, tau=1.5 (first, float64 implementation — superseded)", "notebooks/05"),
]

def family(stem):
    for rx, fam, nb in FAMILIES:
        if re.search(rx, stem): return fam, nb
    return "unclassified", ""

def main():
    rows = []
    for base in [RESULTS / "dim_runs", RESULTS / "paper_experiments", MODELS]:
        for p in sorted(base.rglob("*")):
            if p.suffix not in (".pt", ".npy"): continue
            rel = p.relative_to(ROOT).as_posix()
            fam, nb = family(p.stem)
            row = {"path": rel, "size_mb": round(p.stat().st_size / 1e6, 2), "family": fam, "notebook": nb,
                   "MAP": "", "mean_rank": "", "median_rank": "", "distortion": "", "dim": "", "extra": ""}
            if p.suffix == ".pt":
                try:
                    ck = peek(p)
                    m = ck.get("metrics") or {}
                    row.update({k: m.get(k, "") for k in ("MAP", "mean_rank", "median_rank", "distortion")})
                    row["dim"] = ck.get("dim", "")
                    extra = {k: v for k, v in ck.items() if k not in ("emb", "model", "model_state_dict", "opt", "metrics", "positions", "ref_positions", "kappa_raw", "kappa_weight_state", "curve", "loss_history", "embeddings", "tag")}
                    row["extra"] = json.dumps(extra, default=str)[:300]
                except Exception as e:
                    row["extra"] = f"unreadable: {e!r}"[:200]
            rows.append(row)
    out = RESULTS / "checkpoint_manifest.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} files)")
    un = [r["path"] for r in rows if r["family"] == "unclassified"]
    if un: print("unclassified:", *un, sep="\n  ")

if __name__ == "__main__":
    main()
