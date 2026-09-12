#!/usr/bin/env python3
"""Table 6: Sarkar construction of ICD-10-CM in the Poincaré disk, tau sweep (CPU, ~20 s per tau).

Writes polar coordinates to results/dim_runs/sarkar_tau{tau}_{r,th}.npy and a metrics JSON to
results/paper_experiments/table6_sarkar_sweep.json — the persisted source the August draft lacked.

  python scripts/run_sarkar.py                         # taus 0.5 1 2 3 5 8 12 16 20, float64 ranking on 1000 queries
  python scripts/run_sarkar.py --hp --n-query 1000     # arbitrary-precision ranking (slow: ~3 s/query at tau=8)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10 import load_tree, build_eval_sets
from hyperbolic_icd10.config import DIM_RUNS, RESULTS
from hyperbolic_icd10.sarkar import sarkar_2d_mp, eval_polar, eval_hp, default_dps, separation_threshold

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--taus", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 16.0, 20.0])
    ap.add_argument("--hp", action="store_true", help="also rank in arbitrary precision")
    ap.add_argument("--n-query", type=int, default=40, help="queries for the --hp column (archived: 40)")
    ap.add_argument("--no-save", action="store_true")
    a = ap.parse_args()
    tree = load_tree(); sets = build_eval_sets(tree)
    m = int(tree.branching_factor.max())
    print(f"N={tree.N}  max branching {m}  separation threshold tau > {separation_threshold(m):.4f} (d=2, exact)")
    out = []
    print(f"{'tau':>6}{'dps':>5}{'MAP':>9}{'mean_rk':>9}{'med_rk':>8}{'distort':>9}{'uniq%':>7}{'time':>6}")
    for tau in a.taus:
        t0 = time.time(); dps = default_dps(tau)
        R, T = sarkar_2d_mp(tree, tau, dps)
        r_h = np.array([float(r) for r in R]); th = np.array([float(t) for t in T])
        met = eval_polar(r_h, th, tree, sets.EVAL, sets.DPAIRS)
        uniq = len(np.unique(np.stack([r_h, th], 1), axis=0)) / tree.N
        g = np.diff(np.sort(th)); g = g[g > 0]
        row = {"tau": tau, "dps": dps, "n_query_f64": len(sets.EVAL), **met,
               "unique_coord_frac": uniq, "min_theta_gap": float(g.min()) if len(g) else None,
               "float64_eps": float(np.spacing(1.0)), "max_r_hyp": float(np.nanmax(r_h))}
        if a.hp and np.isfinite(r_h).all():
            hp = eval_hp(R, T, tree, sets.EVAL, a.n_query)
            row.update({"MAP_hp": hp["MAP"], "mean_rank_hp": hp["mean_rank"], "median_rank_hp": hp["median_rank"], "n_query_hp": a.n_query})
        out.append(row)
        print(f"{tau:>6}{dps:>5}{met.get('MAP', float('nan')):>9.4f}{met.get('mean_rank', float('nan')):>9.0f}"
              f"{met.get('median_rank', float('nan')):>8.0f}{met.get('distortion', float('nan')):>9.4f}{100*uniq:>7.1f}{time.time()-t0:>6.0f}s"
              + (f"   hp MAP {row['MAP_hp']:.4f} (n={a.n_query})" if 'MAP_hp' in row else ""))
        if not a.no_save:
            DIM_RUNS.mkdir(parents=True, exist_ok=True)
            np.save(DIM_RUNS / f"sarkar_tau{tau}_r.npy", r_h); np.save(DIM_RUNS / f"sarkar_tau{tau}_th.npy", th)
    if not a.no_save:
        p = RESULTS / "paper_experiments" / "table6_sarkar_sweep.json"
        json.dump(out, open(p, "w"), indent=1); print("wrote", p)

if __name__ == "__main__":
    main()
