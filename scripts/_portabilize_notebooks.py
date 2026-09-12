#!/usr/bin/env python3
"""One-off transform applied 2026-09-12 when the project moved from Google Drive to GitHub.

Kept for provenance. It (1) prepends a bootstrap cell defining HME_ROOT, (2) rewrites every
hard-coded Drive path to HME_ROOT, (3) maps the old results/models sub-folder names (which
contained spaces) to the new ones, (4) removes drive.mount() calls. Cell outputs are untouched.
"""
import json, re, sys
from pathlib import Path

OLD = "/content/drive/MyDrive/Position-Dependent_Curvature_for_Hyperbolic_Embeddings_of_Medical_Ontologies/hyperbolic_medical_embedding"
OLD_CKPT = "/content/drive/MyDrive/dim_runs"
FOLDERS = [("results/V2_closure_contest", "results/v2_closure_contest"),
           ("'V2_closure_contest'", "'v2_closure_contest'"),
           ("Method 1 ablation", "phase3_method1_ablation"), ("Method 1 desat", "phase3_method1_desat"),
           ("Method 1 exp v4", "phase3_method1_exp_v4"), ("Method 1 exp", "phase3_method1_exp"),
           ("'Method 1'", "'phase3_method1'"), ("\"Method 1\"", "\"phase3_method1\""),
           ("'Method 2'", "'phase3_method2'"), ("\"Method 2\"", "\"phase3_method2\""),
           ("Fair comparison v2", "phase4_fair_comparison_v2"), ("Fair comparison", "phase4_fair_comparison_v2"),
           ("Standardized grid", "phase4_standardized_grid"), ("Phase4 standardization", "phase4_standardization"),
           ("Phase4 revised", "phase4_revised"), ("'Baselines'", "'phase2_baselines'"),
           ("results/fixed_eval_d10_seed42.json", "results/eval_sets/phase3_fixed_eval_d10_seed42.json"),
           ("'fixed_eval_d10_seed42.json'", "'eval_sets/phase3_fixed_eval_d10_seed42.json'"),
           ("'fixed_eval_set.json'", "'eval_sets/phase3_fixed_eval_set.json'"),
           ("'all_results_so_far.json'", "'phase3_summary_2026-06-30.json'")]
PHASE2_REF = ["poincare_d10_permanent_freeze_hier.pt", "poincare_d10_freeze_hier.pt", "euclidean_d10.pt", "euclidean_d10_K50.pt",
              "poincare_learnable_K_d10.pt", "poincare_learnable_K_d5.pt", "poincare_learnable_K_d50.pt",
              "poincare_d5_permanent_freeze_hier.pt", "poincare_d50_permanent_freeze_hier.pt", "mixed_curvature_d5.pt",
              "mixed_curvature_d10.pt", "mixed_curvature_d50.pt", "euclidean_d5.pt", "euclidean_d50.pt",
              "phase2_all_baselines.json", "dimension_sweep_d5_d10_d50.json", "learnable_K_sweep.json",
              "euclidean_d10_results.json", "euclidean_d10_K50_results.json", "poincare_d10_attempts_full.json"]

BOOT = '''# --- portable setup (added 2026-09-12 when the project moved to GitHub) ---------------------
# All paths below resolve from HME_ROOT, the repository root.  On Colab: mount Drive and point
# HME_ROOT at your clone.  Locally: run jupyter from the repo, or export HME_ROOT=/path/to/repo.
import os
try:
    from google.colab import drive; drive.mount('/content/drive')
    HME_ROOT = os.environ.get('HME_ROOT', '/content/drive/MyDrive/hyperbolic-icd10')   # <-- edit
except ImportError:
    HME_ROOT = os.environ.get('HME_ROOT', os.path.abspath(os.path.join(os.getcwd(), '..')))
assert os.path.isdir(os.path.join(HME_ROOT, 'data')), f'HME_ROOT={HME_ROOT!r} is not the repo root'
print('HME_ROOT =', HME_ROOT)
'''

def transform(path: Path, phase2_flat: bool):
    nb = json.loads(path.read_text())
    n_sub = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        s = src
        s = re.sub(r"^\s*from google\.colab import drive.*\n?", "", s, flags=re.M)
        s = re.sub(r"^\s*drive\.mount\([^)]*\).*\n?", "", s, flags=re.M)
        s = re.sub(r"^(\s*)drive\.mount\([^)]*\);?\s*", r"\1", s, flags=re.M)
        s = s.replace("'" + OLD, "HME_ROOT + '").replace('"' + OLD, 'HME_ROOT + "')
        s = s.replace("'" + OLD_CKPT + "'", "HME_ROOT + '/results/dim_runs'").replace('"' + OLD_CKPT + '"', 'HME_ROOT + "/results/dim_runs"')
        s = s.replace("HME_ROOT + ''", "HME_ROOT").replace('HME_ROOT + ""', "HME_ROOT")
        for a, b in FOLDERS:
            s = s.replace(a, b)
        if phase2_flat:
            s = s.replace("os.path.join(PROJECT_ROOT, 'results')", "os.path.join(PROJECT_ROOT, 'results', 'phase2_baselines')")
            s = s.replace("os.path.join(PROJECT_ROOT, 'models')", "os.path.join(PROJECT_ROOT, 'models', 'phase2_baselines')")
            s = s.replace("HME_ROOT + '/models'", "HME_ROOT + '/models/phase2_baselines'")
            s = s.replace("HME_ROOT + '/results'", "HME_ROOT + '/results/phase2_baselines'")
        else:
            for f in PHASE2_REF:
                s = s.replace(f"os.path.join(MODELS_DIR, '{f}')", f"os.path.join(MODELS_DIR, 'phase2_baselines', '{f}')")
                s = s.replace(f"os.path.join(RESULTS_DIR, '{f}')", f"os.path.join(RESULTS_DIR, 'phase2_baselines', '{f}')")
        if s != src:
            n_sub += 1
            c["source"] = s.splitlines(keepends=True)
    boot = {"cell_type": "code", "execution_count": None, "metadata": {"tags": ["portable-setup"]}, "outputs": [],
            "source": BOOT.splitlines(keepends=True)}
    nb["cells"].insert(0, boot)
    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    return n_sub

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1] / "notebooks"
    for p in sorted(root.glob("*.ipynb")) + sorted((root / "archive").glob("*.ipynb")):
        n = transform(p, phase2_flat=p.name.startswith("02_"))
        print(f"{p.name}: {n} cells rewritten")
    leftovers = []
    for p in root.rglob("*.ipynb"):
        t = p.read_text()
        if "/content/drive" in t.replace("drive.mount('/content/drive')", ""):
            leftovers.append(p.name)
    print("notebooks still mentioning /content/drive (outside the bootstrap):", leftovers or "none")
