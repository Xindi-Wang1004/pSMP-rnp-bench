#!/usr/bin/env python3
"""Rebuild failure-aware summary for existing Chai-1 / Boltz-2 val50 interface JSONs (CPU)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
OUT = ROOT / "nar_paper" / "tables"

SOURCES = {
    "chai1": ROOT / "train/runs/chai1_baseline_eval/chai1_val50/interface_chai1_val50_ext.json",
    "boltz2": ROOT / "train/runs/boltz2_baseline_eval/boltz2_val50/interface_boltz2_val50_ext.json",
}
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"


def status(r: dict) -> str:
    if r.get("contact_f1_5A") is not None:
        return "evaluable"
    err = str(r.get("error") or "")
    if "missing" in err.lower() or "No such file" in err or not r.get("pred_cif"):
        return "missing_prediction"
    if r.get("n_native_contacts_5A") == 0:
        return "no_interface"
    return "other"


def fa_f1(r: dict):
    st = status(r)
    if st == "no_interface":
        return None
    if st == "evaluable":
        return float(r["contact_f1_5A"])
    return 0.0


def summarize(tag: str, path: Path, all_names: list[str]) -> dict:
    d = json.loads(path.read_text()) if path.exists() else {"results": []}
    mmap = {r["name"]: r for r in d.get("results", []) if "name" in r}
    n_ok = sum(1 for n in all_names if status(mmap.get(n, {"error": "absent"})) == "evaluable")
    vals = []
    for n in all_names:
        v = fa_f1(mmap.get(n, {"error": "absent"}))
        if v is not None:
            vals.append(v)
    return {
        "method": tag,
        "n_cases": len(all_names),
        "n_ok": n_ok,
        "complete_metric_rate": n_ok / len(all_names) if all_names else 0,
        "F1_failure_aware_mean": sum(vals) / len(vals) if vals else None,
        "n_failure_aware_defined": len(vals),
        "source_json": str(path),
        "exists": path.exists(),
    }


def main() -> None:
    names = [r["name"] for r in csv.DictReader(CASES.open(), delimiter="\t")]
    rows = [summarize(k, p, names) for k, p in SOURCES.items()]
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "TableS2c_external_failure_aware.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
