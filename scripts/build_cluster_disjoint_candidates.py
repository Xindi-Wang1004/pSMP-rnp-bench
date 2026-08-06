#!/usr/bin/env python3
"""Freeze a cluster-disjoint companion split candidate list from MMseqs partner audit.

Rule (v0 draft): val cases with protein_identity < 0.40 AND rna_identity < 0.40
to any train neighbor (using Table S11). Not a final confirmatory test — list freeze only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper_audit/out/TableS11_partner_sequence_audit.tsv")
CASES = Path("/home/wangxindi/RNA_Protein/fusai/train/data/rnp_real/val_protenix_inputs/cases.tsv")
OUT_TAB = ROOT / "tables"
OUT_DATA = ROOT / "data"
THR = 0.40


def main() -> None:
    if not AUDIT.exists():
        AUDIT2 = ROOT / "data/stage12_raw/audit/TableS11_partner_sequence_audit.tsv"
        audit_path = AUDIT2 if AUDIT2.exists() else AUDIT
    else:
        audit_path = AUDIT

    cases = {r["name"]: r for r in csv.DictReader(CASES.open(), delimiter="\t")}
    rows = []
    hard = []
    soft = []
    for a in csv.DictReader(audit_path.open(), delimiter="\t"):
        name = a["val_case"]
        pid = float(a.get("protein_identity") or 0)
        rid = float(a.get("rna_identity") or 0)
        both40 = int(float(a.get("both_ge40") or 0))
        is_hard = (pid < THR) and (rid < THR)
        rec = {
            "val_case": name,
            "pdb_id": cases.get(name, {}).get("pdb_id", a.get("val_pdb")),
            "protein_identity": pid,
            "rna_identity": rid,
            "both_ge40": both40,
            "cluster_disjoint_candidate": int(is_hard),
            "nearest_train_protein": a.get("nearest_train_protein"),
            "nearest_train_rna": a.get("nearest_train_rna"),
        }
        rows.append(rec)
        (hard if is_hard else soft).append(name)

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_TAB / "TableS15_cluster_disjoint_candidates.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    manifest = {
        "rule": f"protein_identity<{THR} AND rna_identity<{THR} (Table S11)",
        "n_val": len(rows),
        "n_cluster_disjoint_candidates": len(hard),
        "candidate_ids": hard,
        "note": "List freeze only; do not tune recipe on this set. Score after recipe lock.",
    }
    (OUT_DATA / "cluster_disjoint_candidates.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
