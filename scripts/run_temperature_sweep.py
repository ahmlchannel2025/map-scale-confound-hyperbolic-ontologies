#!/usr/bin/env python3
"""E1: temperature sweep, constant curvature, d=10, 3 seeds — the source of Table 1.  Needs a GPU.

  python scripts/run_temperature_sweep.py --taus 1.0 2.23 3.0 5.0 8.0 --seeds 0 1 2 --out results/paper_experiments/E1_rerun.json
  python scripts/run_temperature_sweep.py --quick        # 200 epochs, 1 seed, smoke test

Each run logs the full metric trajectory every 300 epochs; the paper reports the FINAL epoch
(no checkpoint selection).  Save positions with --save-positions to re-evaluate later.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10 import load_tree, build_eval_sets
from hyperbolic_icd10.config import RESULTS, DIM_RUNS
from hyperbolic_icd10.train import train_logged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="const", choices=["const", "graded", "euclid"])
    ap.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.23, 3.0, 5.0, 8.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dim", type=int, default=10); ap.add_argument("--alpha", type=float, default=-2.0)
    ap.add_argument("--epochs", type=int, default=1500); ap.add_argument("--lr", type=float, default=50)
    ap.add_argument("--K", type=int, default=50); ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--save-positions", action="store_true")
    ap.add_argument("--out", default=str(RESULTS / "paper_experiments" / "E1_rerun.json"))
    a = ap.parse_args()
    if a.quick: a.epochs, a.seeds = 200, a.seeds[:1]
    tree = load_tree(); sets = build_eval_sets(tree)
    runs = []
    for tau in a.taus:
        for seed in a.seeds:
            tag = f"{a.mode}_x{tau}_d{a.dim}_s{seed}"
            saver = None
            if a.save_positions:
                DIM_RUNS.mkdir(parents=True, exist_ok=True)
                saver = lambda pos, tag=tag: np.save(DIM_RUNS / f"{tag}_final.npy", pos)
            runs.append(train_logged(tree, sets, mode=a.mode, scale=tau, dim=a.dim, alpha=a.alpha, epochs=a.epochs,
                                     lr=a.lr, K=a.K, seed=seed, beta=a.beta, tag=tag, save_positions=saver))
            json.dump(runs, open(a.out, "w"), indent=1)
    print("wrote", a.out)

if __name__ == "__main__":
    main()
