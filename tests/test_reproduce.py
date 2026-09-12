"""Reproduction checks. Run: pytest -q  (numpy-only; skips what the checkout lacks)."""
from pathlib import Path
import numpy as np, pytest
from hyperbolic_icd10 import load_tree, build_eval_sets, eval3
from hyperbolic_icd10.config import NODES_CSV, TREE_PICKLE, DIM_RUNS
from hyperbolic_icd10.sarkar import separation_threshold

def test_tree_shape():
    t = load_tree(NODES_CSV)
    assert t.N == 46817 and len(t.edges_idx) == 46816 and t.ROOT == 0
    assert int(t.branching_factor.max()) == 34 and int(t.depth.max()) == 7
    assert int((t.branching_factor == 0).sum()) == 36042

@pytest.mark.skipif(not TREE_PICKLE.exists(), reason="legacy pickle not present")
def test_csv_matches_pickle():
    a, b = load_tree(NODES_CSV), load_tree(TREE_PICKLE)
    assert a.codes == b.codes and a.edges_idx == b.edges_idx
    assert np.array_equal(a.branching_factor, b.branching_factor)

def test_eval_sets_match_notebook_construction():
    t = load_tree(NODES_CSV)
    rng = np.random.default_rng(42)
    expected = [t.edges_idx[i] for i in rng.choice(len(t.edges_idx), size=1000, replace=False)]
    s = build_eval_sets(t)
    assert s.EVAL == expected and len(s.VAL) == 500 and len(s.CLOSURE) == 1000

def test_threshold():
    assert abs(separation_threshold(34) - 4.8239) < 1e-3

@pytest.mark.skipif(not (DIM_RUNS / "ms_const_x1.0_seed0.pt").exists(), reason="checkpoint not present (release asset)")
def test_checkpoint_metrics_reproduce():
    from hyperbolic_icd10.checkpoints import load_arrays
    ck = load_arrays(DIM_RUNS / "ms_const_x1.0_seed0.pt")
    t = load_tree(NODES_CSV); s = build_eval_sets(t)
    m = eval3(ck["emb"].astype(np.float64), s, t.nbrs)
    for k in ("MAP", "mean_rank", "median_rank", "distortion"):
        assert abs(m[k] - ck["metrics"][k]) < 1e-5, k
