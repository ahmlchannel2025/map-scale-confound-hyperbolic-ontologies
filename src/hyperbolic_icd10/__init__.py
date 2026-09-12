"""Hyperbolic embeddings of ICD-10-CM: data, fixed evaluation sets, metrics, trainer, Sarkar construction."""
from .config import ROOT, DATA_PROCESSED, RESULTS, MODELS, DIM_RUNS
from .data import Tree, load_tree, parse_tabular_xml, build_features
from .evalsets import EvalSets, build_eval_sets
from .metrics import eval_all, eval3, map_on, closure, distortion, dist_poincare, dist_euclid

__version__ = "0.1.0"
__all__ = ["ROOT", "DATA_PROCESSED", "RESULTS", "MODELS", "DIM_RUNS", "Tree", "load_tree",
           "parse_tabular_xml", "build_features", "EvalSets", "build_eval_sets", "eval_all",
           "eval3", "map_on", "closure", "distortion", "dist_poincare", "dist_euclid"]
