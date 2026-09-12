"""Reconstruction and generalisation metrics (numpy, CPU).

Definitions — state these exactly when reporting:

* **MAP** (reconstruction): for each query node ``u`` (the parent endpoint of a
  sampled edge), rank all other nodes by embedded distance and compute average
  precision against ``u``'s full neighbour set (parent + children).  Mean over
  queries.  Rank-only: invariant to any monotone rescaling of distances.
* **Closure MRR / hits@10**: for each (node, non-parent ancestor) pair, rank all
  nodes by distance from the node with its *direct neighbours filtered out*;
  reciprocal rank / indicator(rank ≤ 10) of the ancestor.
* **Distortion** (sampled, scale-fitted): over ``DPAIRS``, fit one global
  scalar ``c`` by least squares to map embedded distance onto graph distance,
  then ``mean |c·d_emb − d_graph| / d_graph``.  This is **not** Sala et al.'s
  all-pairs, unscaled ``D(f)``; the two must not be compared (paper §7, blocker
  B7).
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .evalsets import EvalSets


def dist_poincare(a: np.ndarray, allp: np.ndarray) -> np.ndarray:
    """Poincaré-ball (c=1) distance from one point to every row of ``allp``."""
    diff2 = np.sum((allp - a) ** 2, axis=1)
    nu = 1 - np.sum(a ** 2)
    nv = 1 - np.sum(allp ** 2, axis=1)
    return np.arccosh(np.maximum(1 + 2 * diff2 / (nu * nv + 1e-12), 1.0))


def dist_euclid(a: np.ndarray, allp: np.ndarray) -> np.ndarray:
    return np.linalg.norm(allp - a, axis=1)


def _ap(order: np.ndarray, truth: set) -> Optional[float]:
    hit, precs = 0, []
    for j, node in enumerate(order):
        if node in truth:
            hit += 1
            precs.append(hit / (j + 1))
            if hit == len(truth):
                break
    return float(np.mean(precs)) if precs else None


def map_on(pos: np.ndarray, pairs: Sequence[Tuple[int, int]], nbrs: Dict[int, set],
           dfn=dist_poincare, return_ranks: bool = False):
    aps, ranks = [], []
    for u, v in pairs:
        d = dfn(pos[u], pos)
        d[u] = np.inf
        order = np.argsort(d)
        ranks.append(int(np.where(order == v)[0][0]) + 1)
        ap = _ap(order, nbrs[u])
        if ap is not None:
            aps.append(ap)
    m = float(np.mean(aps))
    return (m, np.array(ranks)) if return_ranks else m


def distortion(pos: np.ndarray, dpairs, dfn=dist_poincare) -> float:
    ed = np.array([float(dfn(pos[u], pos[v:v + 1])[0]) for (u, v, g) in dpairs])
    gd = np.array([g for (_, _, g) in dpairs], float)
    c = np.dot(ed, gd) / (np.dot(ed, ed) + 1e-12)
    return float(np.mean(np.abs(c * ed - gd) / gd))


def closure(pos: np.ndarray, pairs, nbrs, dfn=dist_poincare) -> Dict[str, float]:
    rr, h10, ranks = [], [], []
    for v, a in pairs:
        d = dfn(pos[v], pos)
        d[v] = np.inf
        for w in nbrs[v]:
            d[w] = np.inf
        r = int(np.where(np.argsort(d) == a)[0][0]) + 1
        rr.append(1.0 / r)
        h10.append(r <= 10)
        ranks.append(r)
    return {"MRR": float(np.mean(rr)), "hits@10": float(np.mean(h10)),
            "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks))}


def eval_all(pos: np.ndarray, sets: EvalSets, nbrs, euclidean: bool = False) -> Optional[dict]:
    """The paper-protocol evaluator (notebook 09): MAP on VAL and TEST, closure, distortion."""
    if np.isnan(pos).any():
        return None
    dfn = dist_euclid if euclidean else dist_poincare
    clo = closure(pos, sets.CLOSURE, nbrs, dfn)
    return {"MAP_val": map_on(pos, sets.VAL, nbrs, dfn),
            "MAP_test": map_on(pos, sets.TEST, nbrs, dfn),
            "closMRR": clo["MRR"], "hits10": clo["hits@10"],
            "distortion": distortion(pos, sets.DPAIRS, dfn)}


def eval3(pos: np.ndarray, sets: EvalSets, nbrs, euclidean: bool = False) -> dict:
    """The notebook-08 evaluator: MAP / mean rank / median rank on all 1,000 EVAL edges + distortion.

    This is what every ``metrics`` dict inside ``results/dim_runs/*.pt`` was
    computed with (Tables 2 and 4).
    """
    if np.isnan(pos).any():
        nan = float("nan")
        return {"MAP": nan, "mean_rank": nan, "median_rank": nan, "distortion": nan}
    dfn = dist_euclid if euclidean else dist_poincare
    m, ranks = map_on(pos, sets.EVAL, nbrs, dfn, return_ranks=True)
    return {"MAP": m, "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "distortion": distortion(pos, sets.DPAIRS, dfn)}
