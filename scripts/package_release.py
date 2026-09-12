#!/usr/bin/env python3
"""Bundle the git-ignored large artifacts (models, checkpoints, raw data) into a single
archive for a GitHub Release / Zenodo deposit.  Run from a *full* working tree (one that
still has models/ and results/dim_runs/*.pt populated).

  python scripts/package_release.py --out ../hyperbolic-icd10-artifacts-v0.1.0.tar.gz
"""
import argparse, sys, tarfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10.config import ROOT

PATTERNS = ["models/**/*.pt", "results/dim_runs/*.pt", "results/dim_runs/*.npy",
            "results/**/*.pt", "results/**/*.npy", "results/paper_experiments/*.npy",
            "data/raw/icd10cm_2025_code_tables.zip", "data/processed/*.pkl"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT.parent / "hyperbolic-icd10-artifacts.tar.gz"))
    a = ap.parse_args()
    files, seen = [], set()
    for pat in PATTERNS:
        for p in ROOT.glob(pat):
            if p.is_file() and p not in seen:
                seen.add(p); files.append(p)
    total = sum(p.stat().st_size for p in files) / 1e6
    print(f"{len(files)} files, {total:.0f} MB -> {a.out}")
    with tarfile.open(a.out, "w:gz") as tf:
        for p in sorted(files):
            tf.add(p, arcname=str(p.relative_to(ROOT)))
    print("done. Attach this to the GitHub Release; unpack at the repo root to restore.")

if __name__ == "__main__":
    main()
