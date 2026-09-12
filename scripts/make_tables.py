#!/usr/bin/env python3
"""Regenerate the paper's tables from the archived results (no torch needed).

Port of ``notebooks/10_paper_audit.ipynb`` §3–5.  Writes
``results/tables/*.md`` and prints them.  Every number in the August draft's
Tables 1–5 should reconcile from here; anything that does not is listed at the
end as "unsourced".

  Table 1  ICD-10 temperature sweep, d=10, 3 seeds, final epoch  <- paper_experiments/E1_selection.json
  Table 2  three families at d=10, tau=1                           <- dim_runs/{temp_euclid,ms_const,temp_graded}_x1.0*.pt
  Table 3  Euclidean + learning-rate controls                      <- paper_experiments/E2_euclidean.json, E2_lr.json
  Table 4  MAP vs tau at d=2/5/10, 3 seeds                         <- dim_runs/{msld_*,ms_const}_*.pt
  Table 5  WordNet replication                                     <- paper_experiments/E3_wordnet.json
  beta=0   hierarchy-term ablation (n=1)                            <- paper_experiments/B_beta0.json
  Table 6  Sarkar construction sweep (no persisted metrics JSON: run scripts/run_sarkar.py)

Standard deviations: ``--ddof 0`` reproduces the draft's ± values (population
sd); ``--ddof 1`` is the conventional sample sd and is the default here.
"""
import argparse, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hyperbolic_icd10.config import RESULTS
from hyperbolic_icd10.checkpoints import peek

PE = RESULTS / "paper_experiments"
DR = RESULTS / "dim_runs"


def agg(vals, ddof):
    vals = np.asarray(vals, float)
    sd = float(np.std(vals, ddof=ddof)) if len(vals) > 1 else 0.0
    return float(np.mean(vals)), sd


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def fmt(m, s, n):
    return f"{m:.4f} ± {s:.4f}" if n > 1 else f"{m:.4f} (n=1)"


def from_trajectories(fname, ddof, key=lambda r: r["scale"], label="tau"):
    runs = json.loads((PE / fname).read_text())
    g = defaultdict(list)
    for r in runs:
        g[key(r)].append(r["trajectory"][-1])          # final epoch — the paper protocol
    rows = []
    for k in sorted(g, key=lambda x: (isinstance(x, str), x)):
        fin = g[k]
        n = len(fin)
        M = agg([x["MAP_test"] for x in fin], ddof)
        C = agg([x["closMRR"] for x in fin], ddof)
        H = agg([x["hits10"] for x in fin], ddof)
        D = agg([x["distortion"] for x in fin], ddof)
        rows.append([k, n, fmt(*M, n), fmt(*C, n), f"{H[0]:.3f}", f"{D[0]:.4f}"])
    return md_table([label, "n", "recon. MAP (test)", "closure MRR", "hits@10", "distortion"], rows)


def dim_runs_metrics():
    """tag -> metrics for every dim_runs checkpoint carrying a metrics dict."""
    out = {}
    for p in DR.glob("*.pt"):
        try:
            ck = peek(p)
        except Exception:
            continue
        if isinstance(ck, dict) and isinstance(ck.get("metrics"), dict) and "MAP" in ck["metrics"]:
            out[p.stem] = ck["metrics"]
    return out


def table4(metrics, ddof):
    rows = []
    for d in (2, 5, 10):
        for tau in (1.0, 2.23, 5.0):
            if d == 10:
                tags = [f"ms_const_x{tau}_seed{s}" for s in range(3)]
            else:
                tags = [f"msld_const_d{d}_x{tau}_seed{s}" for s in range(3)]
            vals = [metrics[t]["MAP"] for t in tags if t in metrics]
            if not vals:
                rows.append([d, tau, 0, "missing"]); continue
            m, s = agg(vals, ddof)
            rows.append([d, tau, len(vals), fmt(m, s, len(vals))])
    return md_table(["d", "tau", "n", "recon. MAP"], rows)


def table2(metrics, ddof):
    rows = []
    for label, tags in [("Euclidean, tau=1", ["temp_euclid_x1.0"]),
                        ("Poincaré constant, tau=1", [f"ms_const_x1.0_seed{s}" for s in range(3)]),
                        ("Poincaré position-dependent (graded), tau=1", ["temp_graded_x1.0"])]:
        vals = [metrics[t]["MAP"] for t in tags if t in metrics]
        dist = [metrics[t]["distortion"] for t in tags if t in metrics]
        if not vals:
            rows.append([label, 0, "missing", ""]); continue
        m, s = agg(vals, ddof)
        rows.append([label, len(vals), fmt(m, s, len(vals)), f"{np.mean(dist):.4f}"])
    return md_table(["method (d=10)", "n", "recon. MAP", "distortion"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddof", type=int, default=1)
    a = ap.parse_args()
    out_dir = RESULTS / "tables"
    out_dir.mkdir(exist_ok=True)
    metrics = dim_runs_metrics()
    tables = {
        "table1_icd10_temperature_sweep.md": from_trajectories("E1_selection.json", a.ddof),
        "table3a_euclidean_control.md": from_trajectories("E2_euclidean.json", a.ddof),
        "table3b_learning_rate_control.md": from_trajectories("E2_lr.json", a.ddof, key=lambda r: r["tag"].rsplit("_s", 1)[0], label="condition"),
        "table5_wordnet_replication.md": from_trajectories("E3_wordnet.json", a.ddof),
        "beta0_hierarchy_term_ablation.md": from_trajectories("B_beta0.json", a.ddof),
        "table2_three_families_d10_tau1.md": table2(metrics, a.ddof),
        "table4_map_vs_tau_by_dimension.md": table4(metrics, a.ddof),
    }
    for name, tbl in tables.items():
        hdr = f"# {name[:-3]}\n\nGenerated by `scripts/make_tables.py --ddof {a.ddof}` from the archived results. " \
              f"± is {'sample' if a.ddof == 1 else 'population'} standard deviation over seeds.\n\n"
        (out_dir / name).write_text(hdr + tbl + "\n")
        print(f"\n### {name}\n{tbl}")

    # the two n=3 runs of the same condition (d=10, tau=1)
    e1 = json.loads((PE / "E1_selection.json").read_text())
    e1_tau1 = [r["trajectory"][-1]["MAP_test"] for r in e1 if r["scale"] == 1.0]
    ms_tau1 = [metrics[f"ms_const_x1.0_seed{s}"]["MAP"] for s in range(3) if f"ms_const_x1.0_seed{s}" in metrics]
    note = (f"\nNOTE: d=10, tau=1 exists as two independent 3-seed runs: E1_selection (Table 1) "
            f"mean {np.mean(e1_tau1):.4f} and ms_const (Tables 2, 4) mean {np.mean(ms_tau1):.4f}. "
            f"The draft mixes them (0.776 in the abstract, 0.780 elsewhere). Pick one per cell.\n"
            f"NOTE: Table 6 (construction) has no persisted metrics JSON; regenerate with scripts/run_sarkar.py.\n"
            f"NOTE: Table 7 (angular resolution) has no structured source.")
    print(note)
    (out_dir / "NOTES.md").write_text("# Notes from make_tables.py\n" + note)


if __name__ == "__main__":
    main()
