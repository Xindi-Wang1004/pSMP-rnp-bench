#!/usr/bin/env python3
"""Build Table S6 evaluability summary from low-data extended metrics."""
import csv
import json
from pathlib import Path

BASE = Path("/home/wangxindi/RNA_Protein/fusai/train/runs/rnp_real_lowdata/eval_work")
OUT = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper/tables")


def is_ok(r: dict) -> bool:
    return "contact_precision_5A" in r and r.get("contact_precision_5A") is not None


def exclusion_reason(r: dict) -> str:
    if is_ok(r):
        return "evaluable"
    if "error" in r:
        return "pipeline_error"
    n_nat = r.get("n_native_contacts_5A")
    n_pred = r.get("n_predicted_contacts_5A")
    if n_nat is None:
        return "missing_native_or_pred"
    if (n_nat or 0) == 0:
        return "no_native_contact"
    if n_pred is None or (n_pred or 0) == 0:
        return "no_predicted_contact"
    return "other"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_rows = []
    for pct in [10, 25, 50, 100]:
        b = json.loads((BASE / f"interface_pct{pct}_base_ext.json").read_text())
        p = json.loads((BASE / f"interface_pct{pct}_psmp_ext.json").read_text())
        bmap = {r["name"]: r for r in b["results"] if "name" in r}
        pmap = {r["name"]: r for r in p["results"] if "name" in r}
        names = sorted(set(bmap) | set(pmap))
        bok = {n for n, r in bmap.items() if is_ok(r)}
        pok = {n for n, r in pmap.items() if is_ok(r)}
        pair = bok & pok

        def counts(mmap):
            n_ok = n_excl = n_native = n_pred = n_err = n_no_nat = n_no_pred = 0
            for r in mmap.values():
                reason = exclusion_reason(r)
                if reason == "evaluable":
                    n_ok += 1
                else:
                    n_excl += 1
                if reason == "pipeline_error":
                    n_err += 1
                if reason == "no_native_contact":
                    n_no_nat += 1
                if reason == "no_predicted_contact":
                    n_no_pred += 1
                if (r.get("n_native_contacts_5A") or 0) > 0:
                    n_native += 1
                if (r.get("n_predicted_contacts_5A") or 0) > 0:
                    n_pred += 1
            return n_ok, n_excl, n_native, n_pred, n_err, n_no_nat, n_no_pred

        bo, be, bn, bp, berr, bn0, bp0 = counts(bmap)
        po, pe, pn, pp, perr, pn0, pp0 = counts(pmap)
        row = {
            "Train_fraction": f"{pct}%",
            "n_cases": b.get("n_cases", len(names)),
            "n_ok_base": bo,
            "n_excluded_base": be,
            "excl_no_pred_contact_base": bp0,
            "excl_no_native_or_error_base": be - bp0,
            "n_ok_psmp": po,
            "n_excluded_psmp": pe,
            "excl_no_pred_contact_psmp": pp0,
            "excl_no_native_or_error_psmp": pe - pp0,
            "n_pair": len(pair),
        }
        summary_rows.append(row)
        print(row)
        for name in names:
            br = bmap.get(name, {})
            pr = pmap.get(name, {})
            detail_rows.append(
                {
                    "Train_fraction": f"{pct}%",
                    "case": name,
                    "base_reason": exclusion_reason(br) if name in bmap else "missing",
                    "psmp_reason": exclusion_reason(pr) if name in pmap else "missing",
                    "base_n_native": br.get("n_native_contacts_5A", ""),
                    "base_n_pred": br.get("n_predicted_contacts_5A", ""),
                    "psmp_n_native": pr.get("n_native_contacts_5A", ""),
                    "psmp_n_pred": pr.get("n_predicted_contacts_5A", ""),
                    "in_paired_intersection": int(name in pair),
                }
            )

    sum_path = OUT / "TableS6_evaluability_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    det_path = OUT / "TableS6_evaluability_by_case.csv"
    with det_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)
    print("wrote", sum_path, det_path)


if __name__ == "__main__":
    main()
