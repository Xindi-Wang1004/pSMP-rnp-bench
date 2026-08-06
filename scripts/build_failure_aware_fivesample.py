#!/usr/bin/env python3
"""Failure-aware summary for five-sample interface_5seed_*_val50_ext.json."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
EVAL = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"
OUT = ROOT / "nar_paper" / "tables"


def status(r: dict) -> str:
    if r.get("contact_f1_5A") is not None:
        return "evaluable"
    err = str(r.get("error") or "")
    if "missing" in err or "No such file" in err:
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


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for pct in (10, 25, 50, 100):
        bp = EVAL / f"interface_5seed_pct{pct}_base_val50_ext.json"
        pp = EVAL / f"interface_5seed_pct{pct}_psmp_val50_ext.json"
        if not bp.exists() or not pp.exists():
            print(f"MISSING {bp.name} or {pp.name}")
            continue
        bmap = {r["name"]: r for r in json.loads(bp.read_text())["results"]}
        pmap = {r["name"]: r for r in json.loads(pp.read_text())["results"]}
        names = sorted(set(bmap) | set(pmap))
        bf, pf, dd = [], [], []
        bok = pok = 0
        for n in names:
            fb = fa_f1(bmap.get(n, {"error": "missing"}))
            fp = fa_f1(pmap.get(n, {"error": "missing"}))
            if status(bmap.get(n, {})) == "evaluable":
                bok += 1
            if status(pmap.get(n, {})) == "evaluable":
                pok += 1
            if fb is None or fp is None:
                continue
            bf.append(fb)
            pf.append(fp)
            dd.append(fp - fb)
        rows.append(
            {
                "Train_fraction": f"{pct}%",
                "protocol": "five-sample_best_of_iptm",
                "n_cases": len(names),
                "n_ok_base": bok,
                "n_ok_psmp": pok,
                "F1_failure_aware_mean_base": mean(bf),
                "F1_failure_aware_mean_psmp": mean(pf),
                "Delta_F1_failure_aware_mean": mean(dd),
                "n_failure_aware_defined": len(dd),
            }
        )
        print(rows[-1])
    out = OUT / "TableS6b_failure_aware_fivesample.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["Train_fraction"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
