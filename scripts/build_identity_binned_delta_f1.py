#!/usr/bin/env python3
"""Identity-binned ΔF1 with definitions, full bins, Wilcoxon p and bootstrap CI."""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "tables").exists():
    ROOT = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper")
TBL = ROOT / "tables"
S6C = TBL / "TableS6c_fivesample_status_by_case.csv"
S11 = TBL / "TableS11_partner_sequence_audit.tsv"
if not S11.exists():
    S11 = ROOT / "data/stage12_raw/audit/TableS11_partner_sequence_audit.tsv"
if not S11.exists():
    S11 = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper_audit/out/TableS11_partner_sequence_audit.tsv")

# Primary axis: max(protein_identity, rna_identity) from Table S11 (same MMseqs2 audit)
BINS = [
    ("[0,0.3)", 0.0, 0.3),
    ("[0.3,0.4)", 0.3, 0.4),
    ("[0.4,0.7)", 0.4, 0.7),
    ("[0.3,0.7)", 0.3, 0.7),  # pooled mid bin for sparse cells
    ("[0.7,1.0]", 0.7, 1.0001),
]


def bin_of(x: float, bins=None):
    bins = bins or [b for b in BINS if b[0] != "[0.3,0.7)"]
    for lab, lo, hi in bins:
        if lo <= x < hi:
            return lab
    return "NA"


def wilcoxon_greater(deltas):
    try:
        from scipy.stats import wilcoxon

        if not any(abs(d) > 0 for d in deltas):
            return None
        return float(wilcoxon(deltas, alternative="greater").pvalue)
    except Exception:
        return None


def bootstrap_ci(deltas, n_boot=2000, seed=42):
    if not deltas:
        return None, None, None
    rng = random.Random(seed)
    means = []
    n = len(deltas)
    for _ in range(n_boot):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[min(n_boot - 1, int(0.975 * n_boot))]
    return sum(deltas) / n, lo, hi


def main():
    audit = {}
    with S11.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pid = float(r.get("protein_identity") or 0.0)
            rid = float(r.get("rna_identity") or 0.0)
            audit[r["val_case"]] = {
                "protein_identity": pid,
                "rna_identity": rid,
                "max_partner_identity": max(pid, rid),
            }

    by = defaultdict(list)
    case_rows = []
    with S6C.open() as f:
        for r in csv.DictReader(f):
            if r.get("in_paired_intersection") not in ("1", "true", "True"):
                continue
            name = r["case"]
            if name not in audit:
                continue
            bf, pf = r.get("base_f1"), r.get("psmp_f1")
            if bf in ("", None) or pf in ("", None):
                continue
            d = float(pf) - float(bf)
            pct = r["Train_fraction"]
            a = audit[name]
            for axis in ("max_partner_identity", "protein_identity", "rna_identity"):
                val = a[axis]
                # atomic bins
                for lab, lo, hi in BINS:
                    if lab == "[0.3,0.7)":
                        continue
                    if lo <= val < hi:
                        by[(pct, axis, lab)].append(d)
                # pooled mid
                if 0.3 <= val < 0.7:
                    by[(pct, axis, "[0.3,0.7)")].append(d)
            case_rows.append(
                {
                    "Train_fraction": pct,
                    "case": name,
                    "delta_f1": d,
                    "base_f1": float(bf),
                    "psmp_f1": float(pf),
                    **a,
                    "bin_max_partner": bin_of(a["max_partner_identity"]),
                    "identity_definition": (
                        "Table S11 MMseqs2 partner identities; primary axis = "
                        "max(protein_identity, rna_identity)"
                    ),
                }
            )

    rows = []
    for pct in ("10%", "25%", "50%", "100%"):
        for axis in ("max_partner_identity", "protein_identity", "rna_identity"):
            for lab, _, _ in BINS:
                xs = by.get((pct, axis, lab), [])
                m, lo, hi = bootstrap_ci(xs)
                rows.append(
                    {
                        "Train_fraction": pct,
                        "identity_axis": axis,
                        "bin": lab,
                        "n_pair": len(xs),
                        "mean_delta_f1": m if m is not None else "",
                        "bootstrap_ci95_lo": lo if lo is not None else "",
                        "bootstrap_ci95_hi": hi if hi is not None else "",
                        "wilcoxon_p_onesided_greater": wilcoxon_greater(xs) if xs else "",
                        "n_improve": sum(1 for x in xs if x > 0),
                        "n_worse": sum(1 for x in xs if x < 0),
                        "n_tie": sum(1 for x in xs if x == 0),
                        "identity_definition": (
                            "S11 MMseqs2; protein_identity / rna_identity as deposited; "
                            "max_partner_identity = max(protein, rna)"
                        ),
                    }
                )

    out = TBL / "TableS13b_identity_binned_delta_f1.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    outc = TBL / "TableS13c_identity_binned_delta_f1_by_case.csv"
    with outc.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(case_rows[0].keys()))
        w.writeheader()
        w.writerows(sorted(case_rows, key=lambda x: (x["Train_fraction"], x["case"])))

    note = TBL / "Note_S13b_identity_bin_definition.md"
    note.write_text(
        "# Identity-binned ΔF1 (Table S13b)\n\n"
        "- **Source identities**: Table S11 partner-level MMseqs2 audit (same as S12 threshold summaries).\n"
        "- **Primary axis**: `max_partner_identity = max(protein_identity, rna_identity)`.\n"
        "- **Secondary axes**: protein-only and RNA-only (reported for completeness).\n"
        "- **Bins**: [0,0.3), [0.3,0.4), [0.4,0.7), [0.7,1.0]; plus pooled mid [0.3,0.7) when atomic mid bins are sparse.\n"
        "- **Estimand**: paired-intersection ΔF1 (pSMP − base) under the primary five-sample round.\n"
        "- **Uncertainty**: bootstrap 95% CI on mean ΔF1 (2000 resamples); one-sided Wilcoxon (greater).\n"
    )
    print("=== 10% max_partner (atomic + pooled mid) ===")
    for lab in ("[0,0.3)", "[0.3,0.4)", "[0.4,0.7)", "[0.3,0.7)", "[0.7,1.0]"):
        xs = by.get(("10%", "max_partner_identity", lab), [])
        m, lo, hi = bootstrap_ci(xs)
        p = wilcoxon_greater(xs)
        print(f"  {lab}: n={len(xs)} mean={m} CI=[{lo},{hi}] p={p}" if xs else f"  {lab}: n=0")
    print("wrote", out)


if __name__ == "__main__":
    main()
