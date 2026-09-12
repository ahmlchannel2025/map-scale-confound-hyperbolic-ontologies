#!/usr/bin/env python3
"""Phase 1: parse icd10cm_tabular_2025.xml into data/processed/icd10_nodes.csv (+ metadata).

This regenerates the tracked node table from the raw XML.  The ``idx`` column
is the canonical node order used by every checkpoint.  Run after
``download_icd10.py --extract``.  Pass ``--pickle`` to also write the legacy
``icd10_tree_with_features.pkl`` that the original notebooks load.
"""
import argparse, json, pickle, sys
from collections import defaultdict
from datetime import date
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10.config import DATA_RAW, DATA_PROCESSED
from hyperbolic_icd10.data import parse_tabular_xml, build_features, write_nodes_csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(DATA_RAW / "icd10cm_extracted" / "icd10cm_tabular_2025.xml"))
    ap.add_argument("--pickle", action="store_true")
    a = ap.parse_args()
    nodes, edges = parse_tabular_xml(Path(a.xml))
    feats = build_features(nodes, edges)
    parent_of = {c: p for p, c in edges}
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    write_nodes_csv(nodes, feats, parent_of, DATA_PROCESSED / "icd10_nodes.csv")
    meta = {"source_xml": Path(a.xml).name, "icd10cm_version": "FY2025",
            "total_nodes": len(nodes), "total_edges": len(edges),
            "total_internal_nodes": sum(f["is_internal"] for f in feats.values()),
            "total_leaves": sum(f["is_leaf"] for f in feats.values()),
            "max_depth": max(f["depth"] for f in feats.values()),
            "max_branching": max(f["branching_factor"] for f in feats.values()),
            "max_subtree_size": max(f["subtree_size"] for f in feats.values()),
            "date_processed": str(date.today())}
    json.dump(meta, open(DATA_PROCESSED / "icd10_metadata.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))
    assert meta["total_nodes"] == 46817 and meta["total_edges"] == 46816, "tree changed — check the XML version"
    if a.pickle:
        children_of = defaultdict(list)
        for p, c in edges: children_of[p].append(c)
        pickle.dump({"nodes": nodes, "edges": edges, "features": feats,
                     "children_of": dict(children_of), "parent_of": parent_of, "metadata": meta},
                    open(DATA_PROCESSED / "icd10_tree_with_features.pkl", "wb"))
        print("wrote icd10_tree_with_features.pkl")

if __name__ == "__main__":
    main()
