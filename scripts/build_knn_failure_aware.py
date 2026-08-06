#!/usr/bin/env python3
"""k-NN contact baseline under two locked scorings (S11 neighbor definition).

Writes:
  TableS14a_knn_contact_transfer_by_case.csv  (renamed from S14 k-NN)
  TableS14a_knn_failure_aware_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper")
TBL = ROOT / "tables"
SRC = TBL / "TableS14_knn_contact_transfer_by_case.csv"
CASES = Path("/home/wangxindi/RNA_Protein/fusai/train/data/rnp_real/val_protenix_inputs/cases.tsv")
S11 = TBL / "TableS11_partner_sequence_audit.tsv"
if not S11.exists():
    S11 = ROOT / "data/stage12_raw/audit/TableS11_partner_sequence_audit.tsv"


def main():
    # Prefer existing S14 knn file; also accept already-renamed
    src = SRC if SRC.exists() else TBL / "TableS14a_knn_contact_transfer_by_case.csv"
    rows = list(csv.DictReader(src.open()))
    by = {r["val_case"]: r for r in rows}

    # Full-cohort FA: every val50 case; no neighbor / skipped / missing → 0
    case_names = [r["name"] for r in csv.DictReader(CASES.open(), delimiter="\t")]
    fa_vals = []
    out_rows = []
    n_neighbor = n_scored = 0
    for name in case_names:
        r = by.get(name, {})
        status = r.get("status") or "missing_from_knn_table"
        f1 = float(r["knn_f1"]) if r.get("knn_f1") not in (None, "") and status == "scored" else 0.0
        if status == "scored":
            n_scored += 1
        if status in ("scored", "skipped_low_identity") or float(r.get("protein_identity") or 0) > 0:
            n_neighbor += 1
        fa_vals.append(f1)
        out_rows.append(
            {
                **{k: r.get(k, "") for k in (rows[0].keys() if rows else [])},
                "val_case": name,
                "knn_f1_fa": f1,
                "neighbor_definition": "Table S11 MMseqs2 protein nearest-train identity; thr=0.40 for transfer attempt",
            }
        )

    # Conditional mean among scored neighbors only
    scored_f1 = [float(r["knn_f1"]) for r in rows if r.get("status") == "scored" and r.get("knn_f1") not in (None, "")]

    summary = {
        "method": "knn_contact_transfer",
        "neighbor_definition": "S11 MMseqs2 protein-partner nearest train; transfer attempted at protein_identity>=0.40",
        "n_val50": len(case_names),
        "F1_failure_aware_mean_full_cohort": sum(fa_vals) / len(fa_vals) if fa_vals else None,
        "n_scored_neighbors": len(scored_f1),
        "F1_conditional_mean_scored_neighbors_only": (sum(scored_f1) / len(scored_f1)) if scored_f1 else None,
        "note": (
            "Full-cohort FA is the only quantity comparable to Table 2. "
            "Conditional scored-neighbor mean is descriptive and must not be compared to Table 2 FA."
        ),
    }

    out_case = TBL / "TableS14a_knn_contact_transfer_by_case.csv"
    with out_case.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    (TBL / "TableS14a_knn_failure_aware_summary.json").write_text(json.dumps(summary, indent=2))
    # keep legacy filename as pointer
    if SRC.exists() and SRC.name != out_case.name:
        SRC.write_text(
            "# DEPRECATED filename: use TableS14a_knn_contact_transfer_by_case.csv\n"
            + out_case.read_text()
        )
    print(json.dumps(summary, indent=2))
    print("wrote", out_case)


if __name__ == "__main__":
    main()
