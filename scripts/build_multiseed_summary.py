#!/usr/bin/env python3
"""Summarize multiseed val50 eval JSONs → TableS16_multiseed_val50.csv."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
EVAL = ROOT / "train/runs/rnp_real_lowdata_multiseed/eval_work_val50"
TAB = ROOT / "nar_paper" / "tables"
REF = ROOT / "nar_paper" / "data" / "stage12_raw" / "eval_work"


def fa_mean(path: Path) -> tuple[int, float | None]:
    d = json.loads(path.read_text())
    f1s = []
    for r in d.get("results", []):
        if r.get("contact_f1_5A") is not None:
            f1s.append(float(r["contact_f1_5A"]))
            continue
        if r.get("n_native_contacts_5A") == 0:
            continue
        f1s.append(0.0)
    return len(f1s), (sum(f1s) / len(f1s) if f1s else None)


def main() -> None:
    rows = []
    if EVAL.is_dir():
        for p in sorted(EVAL.glob("interface_*_val50_ext.json")):
            tag = p.name.replace("interface_", "").replace("_val50_ext.json", "")
            n, fa = fa_mean(p)
            d = json.loads(p.read_text())
            rows.append(
                {
                    "tag": tag,
                    "n_fa": n,
                    "F1_failure_aware_mean": fa,
                    "n_ok": d.get("n_ok"),
                    "mean_contact_f1_5A_conditional": d.get("mean_contact_f1_5A"),
                    "source": str(p),
                }
            )
    # attach seed42 reference pct10 if available
    for init in ("base", "psmp"):
        ref = REF / f"interface_pct10_{init}_ext.json"
        if ref.is_file():
            n, fa = fa_mean(ref)
            rows.append(
                {
                    "tag": f"pct10_{init}_seed42_ref",
                    "n_fa": n,
                    "F1_failure_aware_mean": fa,
                    "n_ok": json.loads(ref.read_text()).get("n_ok"),
                    "mean_contact_f1_5A_conditional": json.loads(ref.read_text()).get(
                        "mean_contact_f1_5A"
                    ),
                    "source": str(ref),
                }
            )

    TAB.mkdir(parents=True, exist_ok=True)
    out = TAB / "TableS16_multiseed_val50.csv"
    if not rows:
        print("no multiseed eval JSONs yet")
        return
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} n={len(rows)}")


if __name__ == "__main__":
    main()
