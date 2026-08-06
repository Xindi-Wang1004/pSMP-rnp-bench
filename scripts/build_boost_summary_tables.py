#!/usr/bin/env python3
"""Aggregate boost experiment tables: Temporal20/union20, Boltz, PEFT, homology-hard."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
TAB = PAPER / "tables"
DATA = PAPER / "data"


def load_ext(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    d = json.loads(path.read_text())
    return {r["name"]: r for r in d.get("results", []) if "name" in r}


def fa_f1(r: dict | None) -> float:
    if r is None:
        return 0.0
    if r.get("contact_f1_5A") is not None:
        return float(r["contact_f1_5A"])
    return 0.0


def mean_fa(mmap: dict[str, dict], names: list[str]) -> tuple[float, int]:
    vals = [fa_f1(mmap.get(n)) for n in names]
    n_ok = sum(1 for n in names if mmap.get(n, {}).get("contact_f1_5A") is not None)
    return (sum(vals) / len(vals) if vals else float("nan"), n_ok)


def write_temporal_tables() -> None:
    t20_cases = []
    cases_tsv = ROOT / "train/runs/temporal20/cases_temporal20.tsv"
    if cases_tsv.is_file():
        with cases_tsv.open() as f:
            t20_cases = [r["name"] for r in csv.DictReader(f, delimiter="\t")]

    t5_cases = []
    t5_tsv = ROOT / "train/runs/temporal5/inputs/cases_temporal5.tsv"
    if t5_tsv.is_file():
        with t5_tsv.open() as f:
            t5_cases = [r["name"] for r in csv.DictReader(f, delimiter="\t")]

    base20 = load_ext(ROOT / "train/runs/temporal20/interface_base_pct10_ext.json")
    psmp20 = load_ext(ROOT / "train/runs/temporal20/interface_psmp_pct10_ext.json")
    boltz20 = load_ext(
        ROOT / "train/runs/boltz2_baseline_eval/boltz2_temporal20/interface_boltz2_temporal20_ext.json"
    )
    base5 = load_ext(ROOT / "train/runs/temporal5/interface_base_pct10_ext.json")
    psmp5 = load_ext(ROOT / "train/runs/temporal5/interface_psmp_pct10_ext.json")

    rows = []
    for label, names, bmap, pmap in [
        ("Temporal20_new", t20_cases, base20, psmp20),
        ("Temporal5_prior", t5_cases, base5, psmp5),
        ("Temporal_union20", list(dict.fromkeys(t20_cases + t5_cases)), {**base5, **base20}, {**psmp5, **psmp20}),
    ]:
        if not names:
            continue
        b_fa, b_ok = mean_fa(bmap, names)
        p_fa, p_ok = mean_fa(pmap, names)
        z_fa, z_ok = mean_fa(boltz20, names) if label.startswith("Temporal20") else (float("nan"), 0)
        rows.append(
            {
                "cohort": label,
                "n_cases": len(names),
                "FA_F1_base": b_fa,
                "FA_F1_psmp": p_fa,
                "delta_psmp_minus_base": p_fa - b_fa,
                "n_ok_base": b_ok,
                "n_ok_psmp": p_ok,
                "FA_F1_boltz2": z_fa if label.startswith("Temporal20") else "",
                "n_ok_boltz2": z_ok if label.startswith("Temporal20") else "",
                "protocol": "single-sample@10% (Protenix); Boltz zero-shot single-sample on Temporal20_new only",
            }
        )

    out = TAB / "TableS9b_temporal_confirmatory_summary.csv"
    TAB.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # per-case union20
    union = list(dict.fromkeys(t20_cases + t5_cases))
    bmap = {**base5, **base20}
    pmap = {**psmp5, **psmp20}
    by_case = []
    for n in union:
        by_case.append(
            {
                "case": n,
                "F1_base": fa_f1(bmap.get(n)),
                "F1_psmp": fa_f1(pmap.get(n)),
                "delta": fa_f1(pmap.get(n)) - fa_f1(bmap.get(n)),
                "F1_boltz2": fa_f1(boltz20.get(n)) if n in t20_cases else "",
            }
        )
    out2 = TAB / "TableS9c_temporal_union20_by_case.csv"
    with out2.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_case[0].keys()))
        w.writeheader()
        w.writerows(by_case)

    summary = {"summary_rows": rows, "union_n": len(union)}
    (DATA / "temporal_confirmatory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out}\nwrote {out2}")
    for r in rows:
        print(r)


def write_peft_table() -> None:
    peft_path = ROOT / "train/runs/rnp_real_lowdata_peft/eval_work_5seed_val50/interface_5seed_pct10_pairformer_only_val50_ext.json"
    base_path = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50/interface_5seed_pct10_base_val50_ext.json"
    psmp_path = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50/interface_5seed_pct10_psmp_val50_ext.json"
    if not peft_path.is_file():
        print("skip PEFT table — missing", peft_path)
        return
    peft = json.loads(peft_path.read_text())
    base = json.loads(base_path.read_text()) if base_path.is_file() else {}
    psmp = json.loads(psmp_path.read_text()) if psmp_path.is_file() else {}
    row = {
        "method": "pct10_pairformer_only",
        "protocol": peft.get("protocol"),
        "ckpt": peft.get("ckpt"),
        "n_cases": peft.get("n_cases"),
        "n_ok": peft.get("n_ok"),
        "FA_mean_contact_F1": peft.get("FA_mean_contact_f1_5A"),
        "mean_contact_F1_ok_only": peft.get("mean_contact_f1_5A"),
        "ref_base_FA_F1": base.get("mean_contact_f1_5A"),
        "ref_psmp_FA_F1": psmp.get("mean_contact_f1_5A"),
    }
    names = [r["name"] for r in peft.get("results", []) if "name" in r]
    bmap = load_ext(base_path)
    pmap = load_ext(psmp_path)
    pmap2 = load_ext(peft_path)
    b_fa, _ = mean_fa(bmap, names)
    p_fa, _ = mean_fa(pmap, names)
    z_fa, z_ok = mean_fa(pmap2, names)
    row.update(
        {
            "FA_F1_base_locked": b_fa,
            "FA_F1_psmp_locked": p_fa,
            "FA_F1_pairformer_only": z_fa,
            "n_ok_pairformer": z_ok,
        }
    )
    out = TAB / "TableS14c_peft_pairformer_val50_pct10.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"wrote {out}")
    print(json.dumps(row, indent=2))


def write_boltz_row() -> None:
    ext = ROOT / "train/runs/boltz2_baseline_eval/boltz2_temporal20/interface_boltz2_temporal20_ext.json"
    if not ext.is_file():
        print("skip boltz row")
        return
    d = json.loads(ext.read_text())
    row = {
        "method": "boltz2_temporal20",
        "n_cases": d.get("n_cases"),
        "n_ok": d.get("n_ok"),
        "FA_mean_contact_F1": d.get("FA_mean_contact_f1_5A"),
        "mean_contact_F1_ok_only": d.get("mean_contact_f1_5A"),
        "source": str(ext),
    }
    out = TAB / "TableS2d_boltz_temporal20_confirmatory.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"wrote {out}")


def main() -> None:
    write_temporal_tables()
    write_boltz_row()
    write_peft_table()
    date = __import__("datetime").datetime.now().isoformat()
    (DATA / "pipeline_markers" / "boost_tables.ok").write_text(date + "\n")
    print("boost tables done")


if __name__ == "__main__":
    main()
