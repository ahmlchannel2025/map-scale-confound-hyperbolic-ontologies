"""ICD-10-CM tree: parsing (Phase 1) and loading.

The integer index of every node is fixed by *insertion order of the Phase-1
parser* (ROOT, then chapter 1, its first section, that section's first
diagnosis, ...).  Every checkpoint in ``models/`` and ``results/dim_runs/`` is
indexed in that order, and every evaluation set (see :mod:`graph`) is drawn
from ``edges_idx`` in that order with a fixed seed.  ``data/processed/
icd10_nodes.csv`` stores the order explicitly in its ``idx`` column, so the
repository no longer depends on the original pickle.
"""
from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import NODES_CSV, TREE_PICKLE

ROOT_CODE = "ROOT"


# ----------------------------------------------------------------------------
# Phase 1 parser — verbatim logic from notebooks/01_phase1_data_pipeline.ipynb
# ----------------------------------------------------------------------------
def _get_text(element, tag):
    sub = element.find(tag)
    if sub is not None and sub.text is not None:
        return sub.text.strip()
    return None


def parse_tabular_xml(xml_path: Path) -> Tuple[Dict[str, dict], List[Tuple[str, str]]]:
    """Parse ``icd10cm_tabular_2025.xml`` into (nodes, edges).

    Returns ``nodes`` as an insertion-ordered dict ``code -> metadata`` and
    ``edges`` as ``(parent_code, child_code)`` pairs.  A synthetic ``ROOT``
    node sits above the 22 chapters; sections are ``SEC_<id>``.
    """
    root = ET.parse(str(xml_path)).getroot()
    nodes: Dict[str, dict] = {}
    edges: List[Tuple[str, str]] = []
    nodes[ROOT_CODE] = {"code": ROOT_CODE, "description": "ICD-10-CM root",
                        "type": "root", "chapter": None, "depth": 0}

    def walk_diag(diag, parent_code, chapter_num, depth):
        code = _get_text(diag, "name")
        if code is None:
            return
        nodes[code] = {"code": code, "description": _get_text(diag, "desc"),
                       "type": "diagnosis", "chapter": chapter_num, "depth": depth}
        edges.append((parent_code, code))
        for child in diag.findall("diag"):
            walk_diag(child, code, chapter_num, depth + 1)

    for chapter in root.findall("chapter"):
        chapter_num = _get_text(chapter, "name")
        chapter_code = f"CH{chapter_num}"
        nodes[chapter_code] = {"code": chapter_code, "description": _get_text(chapter, "desc"),
                               "type": "chapter", "chapter": chapter_num, "depth": 1}
        edges.append((ROOT_CODE, chapter_code))
        for section in chapter.findall("section"):
            section_id = section.get("id")
            if section_id is None:
                continue
            section_code = f"SEC_{section_id}"
            nodes[section_code] = {"code": section_code, "description": _get_text(section, "desc"),
                                   "type": "section", "chapter": chapter_num, "depth": 2}
            edges.append((chapter_code, section_code))
            for diag in section.findall("diag"):
                walk_diag(diag, section_code, chapter_num, 3)
    return nodes, edges


def build_features(nodes: Dict[str, dict], edges: List[Tuple[str, str]]) -> Dict[str, dict]:
    """Per-node features exactly as Phase 1 defined them."""
    children_of = defaultdict(list)
    parent_of: Dict[str, str] = {}
    for p, c in edges:
        children_of[p].append(c)
        parent_of[c] = p

    subtree: Dict[str, int] = {}

    def size(code):
        if code in subtree:
            return subtree[code]
        n = 0
        for ch in children_of[code]:
            n += 1 + size(ch)
        subtree[code] = n
        return n

    def section_of(code):
        while code in parent_of:
            if nodes[code]["type"] == "section":
                return code
            code = parent_of[code]
        return None

    feats = {}
    for code, node in nodes.items():
        b = len(children_of[code])
        feats[code] = {"code": code, "description": node["description"], "type": node["type"],
                       "chapter": node["chapter"], "section": section_of(code),
                       "depth": node["depth"], "branching_factor": b,
                       "subtree_size": size(code), "is_leaf": b == 0, "is_internal": b > 0}
    return feats


def write_nodes_csv(nodes, feats, parent_of, path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "code", "parent_code", "type", "chapter", "section", "depth",
                    "branching_factor", "subtree_size", "is_leaf", "description"])
        for i, c in enumerate(nodes):
            ft = feats[c]
            w.writerow([i, c, parent_of.get(c, ""), ft["type"], ft["chapter"] or "",
                        ft["section"] or "", ft["depth"], ft["branching_factor"],
                        ft["subtree_size"], int(ft["is_leaf"]), nodes[c].get("description", "")])


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------
@dataclass
class Tree:
    codes: List[str]
    code_to_idx: Dict[str, int]
    edges_idx: List[Tuple[int, int]]          # (parent, child), Phase-1 order
    parent: Dict[int, int]
    kids: Dict[int, List[int]]
    nbrs: Dict[int, Set[int]]
    connected: Set[Tuple[int, int]]
    branching_factor: np.ndarray               # int, per node
    subtree_size: np.ndarray
    depth: np.ndarray
    description: List[str] = field(default_factory=list)

    @property
    def N(self) -> int:
        return len(self.codes)

    @property
    def ROOT(self) -> int:
        return self.code_to_idx[ROOT_CODE]

    def ancestors(self, v: int) -> List[int]:
        out = []
        while v in self.parent:
            v = self.parent[v]
            out.append(v)
        return out


def load_tree(path: Optional[Path] = None) -> Tree:
    """Load the tree from ``icd10_nodes.csv`` (preferred) or the Phase-1 pickle.

    Both give identical indices; the CSV is the tracked, format-stable copy.
    """
    path = Path(path) if path else (NODES_CSV if NODES_CSV.exists() else TREE_PICKLE)
    if path.suffix == ".pkl":
        import pickle
        d = pickle.load(open(path, "rb"))
        codes = list(d["nodes"].keys())
        edges = list(d["edges"])
        f = d["features"]
        bf = np.array([f[c]["branching_factor"] for c in codes])
        ss = np.array([f[c]["subtree_size"] for c in codes])
        dp = np.array([f[c]["depth"] for c in codes])
        desc = [d["nodes"][c].get("description") or "" for c in codes]
    else:
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        rows.sort(key=lambda r: int(r["idx"]))
        codes = [r["code"] for r in rows]
        edges = [(r["parent_code"], r["code"]) for r in rows if r["parent_code"]]
        bf = np.array([int(r["branching_factor"]) for r in rows])
        ss = np.array([int(r["subtree_size"]) for r in rows])
        dp = np.array([int(r["depth"]) for r in rows])
        desc = [r["description"] for r in rows]
        # Phase-1 edge order is the child's insertion order; rows are already
        # in that order, so the list comprehension above reproduces it.

    c2i = {c: i for i, c in enumerate(codes)}
    edges_idx = [(c2i[p], c2i[c]) for p, c in edges]
    parent, kids, nbrs = {}, defaultdict(list), defaultdict(set)
    for u, v in edges_idx:
        parent[v] = u
        kids[u].append(v)
        nbrs[u].add(v)
        nbrs[v].add(u)
    connected = set(edges_idx) | {(v, u) for u, v in edges_idx}
    return Tree(codes, c2i, edges_idx, parent, dict(kids), dict(nbrs), connected,
                bf, ss, dp, desc)
