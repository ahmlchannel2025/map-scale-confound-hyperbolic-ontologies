#!/usr/bin/env python3
"""Re-evaluate any saved embedding with the two archived protocols (no torch needed).

  python scripts/reevaluate_checkpoint.py results/dim_runs/ms_const_x1.0_seed0.pt
  python scripts/reevaluate_checkpoint.py results/paper_experiments/pos_A_full_x8.0_s0.npy --protocol paper

``eval3`` (notebook 08): MAP/mean rank/median rank on the 1,000 EVAL edges + distortion — matches the
``metrics`` dict stored in results/dim_runs/*.pt to ~1e-7.
``paper`` (notebook 09): MAP on VAL/TEST halves, closure MRR/hits@10, distortion.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10 import load_tree, build_eval_sets, eval3, eval_all
from hyperbolic_icd10.checkpoints import load_arrays

def load_positions(path):
    p = Path(path)
    if p.suffix == ".npy": return np.load(p), {}
    ck = load_arrays(p)
    for k in ("emb", "positions"):
        if k in ck: return np.asarray(ck[k]), ck.get("metrics", {})
    for k in ("model_state_dict", "model"):
        if k in ck:
            sd = ck[k]
            for kk in ("embeddings", "embeddings.weight"):
                if kk in sd: return np.asarray(sd[kk]), ck.get("metrics", {})
    if "embeddings" in ck: return np.asarray(ck["embeddings"]), ck.get("metrics", {})
    raise SystemExit(f"no embedding found in {path}; keys={list(ck)}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--protocol", choices=["eval3", "paper"], default="eval3")
    ap.add_argument("--euclidean", action="store_true"); a = ap.parse_args()
    pos, stored = load_positions(a.path); pos = pos.astype(np.float64)
    tree = load_tree(); sets = build_eval_sets(tree)
    euc = a.euclidean or "euclid" in Path(a.path).stem
    m = eval3(pos, sets, tree.nbrs, euclidean=euc) if a.protocol == "eval3" else eval_all(pos, sets, tree.nbrs, euclidean=euc)
    print(json.dumps({"path": a.path, "shape": list(pos.shape), "euclidean": euc, "protocol": a.protocol, "recomputed": m, "stored": stored}, indent=1))

if __name__ == "__main__":
    main()
