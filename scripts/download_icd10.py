#!/usr/bin/env python3
"""Download the FY2025 ICD-10-CM Code Tables (CMS, public domain) into data/raw/.

Usage:  python scripts/download_icd10.py [--extract]
"""
import argparse, sys, zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10.config import DATA_RAW, ICD10_URL

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true", help="unzip into data/raw/icd10cm_extracted/")
    a = ap.parse_args()
    import requests
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    out = DATA_RAW / "icd10cm_2025_code_tables.zip"
    if out.exists():
        print(f"exists: {out} ({out.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"downloading {ICD10_URL}")
        r = requests.get(ICD10_URL, stream=True, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        print(f"saved {out} ({out.stat().st_size/1e6:.1f} MB)")
    if a.extract:
        d = DATA_RAW / "icd10cm_extracted"
        d.mkdir(exist_ok=True)
        with zipfile.ZipFile(out) as zf:
            zf.extractall(d)
        print("extracted:", sorted(p.name for p in d.iterdir()))

if __name__ == "__main__":
    main()
