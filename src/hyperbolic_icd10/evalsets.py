"""Fixed evaluation sets.

These reproduce, bit for bit, the sets used in ``notebooks/09_paper_experiments``
(Tables 1, 3, 5 and the β=0 ablation) and ``notebooks/08_temperature_experiments``
/ ``11_v2_closure_contest`` (Tables 2, 4).  Do not change the seeds or the
sampling order: any change silently makes new numbers incomparable with the
archived ones.

* ``EVAL``     1,000 parent–child edges, ``default_rng(42)``, drawn from
               ``edges_idx`` in Phase-1 order.  Reconstruction MAP is computed
               on the *parent* endpoint of each edge (a per-node MAP over a
               node sample, not over all 46,817 nodes — see README "Known
               issues").
* ``VAL/TEST`` first / second 500 of ``EVAL`` (paper protocol; selection on
               VAL, reporting on TEST).
* ``DPAIRS``   ≤2,000 (source, target, graph distance) pairs, ``default_rng(0)``,
               BFS shortest paths, for distortion.
* ``CLOSURE``  1,000 (node, non-parent ancestor) pairs, ``default_rng(11)``.
               ``rng(12)`` gives the disjoint *selection* closure set used by
               the V2 contest.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .data import Tree


@dataclass
class EvalSets:
    EVAL: List[Tuple[int, int]]
    VAL: List[Tuple[int, int]]
    TEST: List[Tuple[int, int]]
    DPAIRS: List[Tuple[int, int, int]]
    CLOSURE: List[Tuple[int, int]]


def edge_sample(tree: Tree, n: int = 1000, seed: int = 42) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    E = tree.edges_idx
    return [E[i] for i in rng.choice(len(E), size=min(n, len(E)), replace=False)]


def distortion_pairs(tree: Tree, n_sources: int = 2000, seed: int = 0):
    adj = defaultdict(list)
    for u, v in tree.edges_idx:
        adj[u].append(v)
        adj[v].append(u)
    rng = np.random.default_rng(seed)
    N = tree.N
    out = []
    for s in rng.integers(0, N, size=n_sources):
        dd = {int(s): 0}
        q = deque([int(s)])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in dd:
                    dd[y] = dd[x] + 1
                    q.append(y)
        t = int(rng.integers(0, N))
        if t in dd and t != s:
            out.append((int(s), t, dd[t]))
    return out


def closure_pairs(tree: Tree, n: int = 1000, seed: int = 11):
    """(node, random non-parent ancestor) pairs never used as training positives."""
    rng = np.random.default_rng(seed)
    cands = [i for i in range(tree.N) if i in tree.parent and len(tree.ancestors(i)) >= 2]
    out = []
    for v in rng.choice(cands, size=min(n, len(cands)), replace=False):
        anc = tree.ancestors(int(v))[1:]
        if anc:
            out.append((int(v), int(rng.choice(anc))))
    return out


def build_eval_sets(tree: Tree, eval_size: int = 1000, closure_seed: int = 11) -> EvalSets:
    EVAL = edge_sample(tree, eval_size, 42)
    half = eval_size // 2
    return EvalSets(EVAL, EVAL[:half], EVAL[half:], distortion_pairs(tree),
                    closure_pairs(tree, 1000, closure_seed))
