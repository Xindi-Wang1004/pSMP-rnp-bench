#!/usr/bin/env python3
"""Classify missing_prediction / failure modes from interface_*_ext.json (review §4.3-11)."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "stage12_raw" / "eval_work"
OUT = ROOT / "tables"
NOTE = ROOT / "supplementary"


def classify(r: dict) -> tuple[str, str]:
    if r.get("contact_precision_5A") is not None and r.get("contact_f1_5A") is not None:
        return "evaluable", ""
    err = str(r.get("error") or "")
    if "No such file" in err or "not found" in err.lower() or "pred_cif_not_found" in err:
        # distinguish empty pred dir vs wrong nested path
        if "sample_0.cif" in err:
            return "missing_prediction_no_cif", "expected_sample0_cif_absent"
        if "pred_cif_not_found" in err:
            return "missing_prediction_finder_empty", "find_best_pred_cif_returned_none"
        return "missing_prediction_path", "other_path_error"
    n_nat = r.get("n_native_contacts_5A")
    n_pred = r.get("n_predicted_contacts_5A")
    if n_nat == 0:
        return "no_interface", ""
    if "no_interface" in err:
        return "no_interface", ""
    if "mismatch" in err:
        return "evaluator_interface_atom_mismatch", err[:120]
    if n_pred == 0:
        return "no_predicted_contact", ""
    if err:
        return "pipeline_error_other", err[:160]
    if n_nat is None:
        return "missing_prediction_unscored", "native_none"
    return "other", err[:120]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    detail = []
    global_modes = Counter()
    for pct in (10, 25, 50, 100):
        for tag in ("base", "psmp"):
            d = json.loads((RAW / f"interface_pct{pct}_{tag}_ext.json").read_text())
            ctr = Counter()
            for r in d["results"]:
                mode, detail_s = classify(r)
                ctr[mode] += 1
                global_modes[mode] += 1
                if mode != "evaluable":
                    detail.append(
                        {
                            "Train_fraction": f"{pct}%",
                            "method": tag,
                            "case": r.get("name"),
                            "failure_mode": mode,
                            "detail": detail_s,
                            "error_head": str(r.get("error") or "")[:200],
                        }
                    )
            row = {"Train_fraction": f"{pct}%", "method": tag, **{k: ctr.get(k, 0) for k in sorted(set(ctr) | {"evaluable"})}}
            # ensure common columns
            for k in (
                "evaluable",
                "missing_prediction_no_cif",
                "missing_prediction_finder_empty",
                "missing_prediction_path",
                "missing_prediction_unscored",
                "no_interface",
                "no_predicted_contact",
                "evaluator_interface_atom_mismatch",
                "pipeline_error_other",
                "other",
            ):
                row.setdefault(k, ctr.get(k, 0))
            summary.append(row)

    # unify columns
    cols = ["Train_fraction", "method"] + [
        "evaluable",
        "missing_prediction_no_cif",
        "missing_prediction_finder_empty",
        "missing_prediction_path",
        "missing_prediction_unscored",
        "no_interface",
        "no_predicted_contact",
        "evaluator_interface_atom_mismatch",
        "pipeline_error_other",
        "other",
    ]
    sum_path = OUT / "TableS6d_missing_prediction_rootcause.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in summary:
            w.writerow({k: row.get(k, 0) for k in cols})

    det_path = OUT / "TableS6e_missing_prediction_by_case.csv"
    with det_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0].keys()))
        w.writeheader()
        w.writerows(detail)

    # dominant story
    note = NOTE / "Note18_missing_prediction_rootcause.md"
    note.write_text(
        "\n".join(
            [
                "# Supplementary Note 18 | missing_prediction root-cause inventory",
                "",
                "## Method",
                "Re-parse deposited single-sample `interface_pct*_ext.json` error strings",
                "(no new inference). Modes defined in `scripts/analyze_missing_prediction_rootcause.py`.",
                "",
                "## Dominant finding",
                "Across 10–50% fine-tunes, the large majority of non-evaluable cases are",
                "**`missing_prediction_no_cif`**: the evaluator expected",
                "`.../predictions/<case>_sample_0.cif` but the file was absent.",
                "A smaller recurring set is **`missing_prediction_finder_empty`**",
                "(`find_best_pred_cif` returned none / `pred_cif_not_found`).",
                "",
                "This points primarily to **inference write/skip failures or path layout drift**,",
                "not to evaluator math on existing CIFs. At 100% the CIF-missing rate collapses,",
                "consistent with more successful writes under the full-data fine-tune.",
                "",
                "## What this does *not* yet prove",
                "- Whether absences are OOM, NaN coords, process kill, or intentional skip.",
                "- Whether five-sample reinference removes the same failures (in progress on cluster).",
                "",
                "## Next engineering step (review §4.3-11)",
                "1. For a stratified sample of missing cases, inspect pred directories and inference logs.",
                "2. Fix path contract or retry policy; then rescore — do not only raise `-e` to hide gaps.",
                "3. Keep failure-aware F1 (Note 17) as the full-cohort estimand meanwhile.",
                "",
                "## Outputs",
                f"- `{sum_path.relative_to(ROOT)}`",
                f"- `{det_path.relative_to(ROOT)}`",
                "",
                f"Mode totals (all fractions × methods): {dict(global_modes)}",
                "",
                "_Generated: 2026-08-03_",
                "",
            ]
        )
    )
    print(f"wrote {sum_path}")
    print(f"wrote {det_path}")
    print(f"wrote {note}")
    print("mode totals", dict(global_modes))


if __name__ == "__main__":
    main()
