#!/usr/bin/env python3
"""Sibling-separation threshold: exact (d=2) value vs. the unconverged solver value in the draft.

Notebook 10 §1.  The draft quotes tau > 5.02 at d=2 from a 1,500-iteration repulsion solver
that had not converged; the exact value for max branching 34 is 4.8239.  Both lie inside the
empirically bracketed interval (tau=3 fails, tau=5 succeeds), so the qualitative claim survives;
the number in the text does not.
"""
import math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10 import load_tree
from hyperbolic_icd10.sarkar import separation_threshold

def spherical_code(m, dim, iters=1500, seed=0):
    """Verbatim solver from notebook 07 (repulsion on the sphere, lr decays as 0.999^t)."""
    n = m + 1; rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)); X /= np.linalg.norm(X, axis=1, keepdims=True); lr = 0.15
    for _ in range(iters):
        F = np.zeros_like(X)
        for i in range(n):
            diff = X[i] - X; dist = np.linalg.norm(diff, axis=1) + 1e-9
            w = 1.0 / dist ** 3; w[i] = 0; F[i] = (diff * w[:, None]).sum(0)
        F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-12)
        X += lr * F; X /= np.linalg.norm(X, axis=1, keepdims=True); lr *= 0.999
    return X

def min_angle(X):
    D = X @ X.T; np.fill_diagonal(D, -2.0); return float(np.arccos(np.clip(D.max(), -1, 1)))

def tau_from_angle(theta): return -2.0 * math.log(math.sin(theta / 2.0))

if __name__ == "__main__":
    m = int(load_tree().branching_factor.max()); print("max branching", m)
    print("as in the draft (iters=1500, seed 0):")
    for d in (2, 5, 10):
        a = min_angle(spherical_code(m, d, 1500)); print(f"  d={d:<3} theta={a:.4f}  tau>{tau_from_angle(a):.2f}")
    print("converged (iters=20000, best of 5 restarts):")
    for d in (2, 5, 10):
        a = max(min_angle(spherical_code(m, d, 20000, s)) for s in range(5)); print(f"  d={d:<3} theta={a:.4f}  tau>{tau_from_angle(a):.2f}")
    print(f"d=2 exact: theta=2pi/{m+1}={2*math.pi/(m+1):.4f} -> tau > {separation_threshold(m):.4f}")
