#!/usr/bin/env python3
"""Rebuild data/processed/icd10_tree_with_features.pkl (the file every notebook loads) from the
tracked icd10_nodes.csv — no raw XML needed.  Node order and every feature are identical to the
original Phase-1 pickle (verified by tests/test_reproduce.py::test_csv_matches_pickle).
"""
import csv, pickle, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10.config import NODES_CSV, TREE_PICKLE, DATA_PROCESSED
import json

rows = sorted(csv.DictReader(open(NODES_CSV, newline="", encoding="utf-8")), key=lambda r: int(r["idx"]))
nodes, feats, edges = {}, {}, []
for r in rows:
    c = r["code"]
    nodes[c] = {"code": c, "description": r["description"] or None, "type": r["type"],
                "chapter": r["chapter"] or None, "depth": int(r["depth"])}
    b = int(r["branching_factor"])
    feats[c] = {**nodes[c], "section": r["section"] or None, "branching_factor": b,
                "subtree_size": int(r["subtree_size"]), "is_leaf": b == 0, "is_internal": b > 0}
    if r["parent_code"]:
        edges.append((r["parent_code"], c))
children_of = defaultdict(list); parent_of = {}
for p, c in edges:
    children_of[p].append(c); parent_of[c] = p
meta = json.load(open(DATA_PROCESSED / "icd10_metadata.json"))
pickle.dump({"nodes": nodes, "edges": edges, "features": feats, "children_of": dict(children_of),
             "parent_of": parent_of, "metadata": meta}, open(TREE_PICKLE, "wb"))
print("wrote", TREE_PICKLE, len(nodes), "nodes", len(edges), "edges")
