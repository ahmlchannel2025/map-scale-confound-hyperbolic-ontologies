"""Sarkar (2011) combinatorial embedding of the ICD-10-CM tree in the Poincaré disk.

Port of notebook 07 (cells "SARKAR CONSTRUCTION ON ICD-10", "high-precision
evaluation").  The construction is exact in ``mpmath`` arbitrary precision;
coordinates are then stored in polar form ``(r_hyp, θ)`` and ranked either in
float64 (``eval_polar``) or in arbitrary precision (``eval_hp``).

``tau`` is the hyperbolic edge length (Sala et al.'s scaling factor).  The
sibling-separation threshold for the largest fan-out ``m`` in the tree is

    τ > −2·ln(sin(θ/2)),   θ = 2π/(m+1)   (exact at d = 2)

which for ``m = 34`` gives τ > 4.82 — **not** the 5.02 that an unconverged
repulsion solver produced and that the August draft still quotes (see
notebook 10, §1).
"""
from __future__ import annotations

from collections import deque
from typing import Dict, Tuple

import numpy as np

from .data import Tree


def separation_threshold(m: int) -> float:
    """τ above which two of ``m`` siblings at radius τ around a parent are ≥ τ apart (d=2, exact)."""
    import math
    theta = 2 * math.pi / (m + 1)
    return -2.0 * math.log(math.sin(theta / 2.0))


def default_dps(tau: float) -> int:
    """Decimal digits used in the archived runs: int(7·τ/ln 10) + 50."""
    return int(7 * tau / np.log(10)) + 50


def sarkar_2d_mp(tree: Tree, tau: float, dps: int):
    """Return mpmath lists ``(R, T)`` of hyperbolic radius and angle for every node."""
    from mpmath import mp, mpf, mpc
    mp.dps = dps
    N, ROOT, kids, parent = tree.N, tree.ROOT, tree.kids, tree.parent
    coord = [None] * N
    coord[ROOT] = mpc(0, 0)
    r_e = mp.tanh(mpf(tau) / 2)
    k = len(kids.get(ROOT, []))
    for j, c in enumerate(kids.get(ROOT, [])):
        a = 2 * mp.pi * j / k
        coord[c] = r_e * mpc(mp.cos(a), mp.sin(a))
    q = deque(kids.get(ROOT, []))
    while q:
        v = q.popleft()
        ch = kids.get(v, [])
        if not ch:
            continue
        x = coord[v]
        xc = mp.conj(x)
        pl = (coord[parent[v]] - x) / (1 - xc * coord[parent[v]])   # parent seen from v at origin
        th_p = mp.atan2(pl.imag, pl.real)
        m = len(ch)
        for j, c in enumerate(ch):
            th = th_p + 2 * mp.pi * (j + 1) / (m + 1)                 # skip the parent's slot
            loc = r_e * mpc(mp.cos(th), mp.sin(th))
            coord[c] = (loc + x) / (1 + xc * loc)                     # Möbius-translate back
            q.append(c)
    R, T = [], []
    for i in range(N):
        z = coord[i]
        a = mp.sqrt(z.real ** 2 + z.imag ** 2)
        R.append(2 * mp.atanh(a))
        T.append(mp.atan2(z.imag, z.real))
    return R, T


def sarkar_2d(tree: Tree, tau: float, dps: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Construct and return float64 polar coordinates ``(r_hyp, theta)``."""
    dps = dps or default_dps(tau)
    R, T = sarkar_2d_mp(tree, tau, dps)
    r_h = np.array([float(r) for r in R])
    th = np.array([float(t) for t in T])
    return r_h, th


def _logsinh(x):
    return np.where(x > 20, x - np.log(2), np.log(np.sinh(np.clip(x, 1e-300, None))))


def dist_from_polar(i: int, r_h: np.ndarray, th: np.ndarray) -> np.ndarray:
    """Stable hyperbolic distance from node ``i`` to all nodes, in polar form.

    cosh d = cosh(Δr) + 2·sinh(r_i)·sinh(r_j)·sin²(Δθ/2)
    """
    n = len(r_h)
    d0 = np.abs(th - th[i])
    d0 = np.minimum(d0, 2 * np.pi - d0)
    dr = np.abs(r_h - r_h[i])
    lg = (np.log(2) + _logsinh(np.full(n, r_h[i])) + _logsinh(r_h)
          + 2 * np.log(np.clip(np.sin(d0 / 2), 1e-300, None)))
    out = np.empty(n)
    far = lg > 500
    out[far] = np.log(2) + lg[far]
    ok = ~far
    out[ok] = np.arccosh(np.maximum(np.cosh(np.clip(dr[ok], 0, 500))
                                    + np.exp(np.clip(lg[ok], -700, 500)), 1.0))
    return out


def eval_polar(r_h, th, tree: Tree, EVAL, DPAIRS) -> Dict[str, float]:
    """float64 MAP / mean rank / median rank / distortion, same protocol as ``metrics.eval3``."""
    if not np.isfinite(r_h).all():
        return {"MAP": float("nan"), "note": f"{(~np.isfinite(r_h)).sum()} non-finite"}
    ranks, aps = [], []
    for u, v in EVAL:
        d = dist_from_polar(u, r_h, th)
        d[u] = np.inf
        o = np.argsort(d)
        ranks.append(int(np.where(o == v)[0][0]) + 1)
        tr = tree.nbrs[u]
        h, pr = 0, []
        for j, nd in enumerate(o):
            if nd in tr:
                h += 1
                pr.append(h / (j + 1))
                if h == len(tr):
                    break
        if pr:
            aps.append(np.mean(pr))
    ed = np.array([dist_from_polar(u, r_h, th)[v] for (u, v, _) in DPAIRS])
    gd = np.array([g for (_, _, g) in DPAIRS], float)
    c = np.dot(ed, gd) / (np.dot(ed, ed) + 1e-12)
    ranks = np.array(ranks)
    return {"MAP": float(np.mean(aps)), "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "distortion": float(np.mean(np.abs(c * ed - gd) / gd))}


def eval_hp(R, T, tree: Tree, EVAL, n_query: int = 40):
    """Arbitrary-precision ranking by cosh(d) (monotone in d). Slow: O(n_query · N) mp ops.

    The archived numbers (MAP 1.0000 at τ = 8, 12, 20) used ``n_query = 40``.
    Forty queries is a demonstration, not a reportable sample; rerun with
    ``n_query = 1000`` before this column carries weight in a table.
    """
    from mpmath import mp
    N = tree.N
    SH = [mp.sinh(r) for r in R]
    ranks, aps = [], []
    for u, v in EVAL[:n_query]:
        ru, su, tu = R[u], SH[u], T[u]
        key = []
        for j in range(N):
            if j == u:
                key.append(mp.inf)
                continue
            s = mp.sin((T[j] - tu) / 2)
            key.append(mp.cosh(ru - R[j]) + 2 * su * SH[j] * s * s)
        o = sorted(range(N), key=lambda j: key[j])
        ranks.append(o.index(v) + 1)
        tr = tree.nbrs[u]
        h, pr = 0, []
        for jj, nd in enumerate(o):
            if nd in tr:
                h += 1
                pr.append(h / (jj + 1))
                if h == len(tr):
                    break
        if pr:
            aps.append(float(np.mean(pr)))
    return {"MAP": float(np.mean(aps)), "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)), "n_query": n_query}
