#!/usr/bin/env python3
"""Build failure-aware / success-rate tables from single-sample interface_*_ext.json.

Decision rule (locked for this analysis):
  - missing_prediction / pipeline error / no CIF  -> F1 = 0, recall = 0, precision = 0
  - no_interface (n_native_contacts_5A == 0)      -> excluded from F1 denominators but counted in success rates
  - evaluable                                      -> use reported contact_*_5A

Primary estimand reported here: mean F1 over all 50 val cases with failures as zero
(failure-aware), alongside conditional means on evaluable / paired-intersection subsets.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "stage12_raw" / "eval_work"
OUT = ROOT / "tables"
DATA = ROOT / "data"


def status(r: dict) -> str:
    if r.get("contact_precision_5A") is not None and r.get("contact_f1_5A") is not None:
        return "evaluable"
    err = (r.get("error") or "")
    if isinstance(err, str) and ("No such file" in err or "not found" in err.lower()):
        return "missing_prediction"
    n_nat = r.get("n_native_contacts_5A")
    n_pred = r.get("n_predicted_contacts_5A")
    if n_nat == 0:
        return "no_interface"
    if err:
        return "pipeline_error"
    if n_pred == 0 or n_pred is None:
        return "no_predicted_contact"
    if n_nat is None:
        return "missing_prediction"
    return "other"


def f1_failure_aware(r: dict) -> float | None:
    st = status(r)
    if st == "no_interface":
        return None  # undefined native interface
    if st == "evaluable":
        return float(r["contact_f1_5A"])
    return 0.0


def recall_failure_aware(r: dict) -> float | None:
    st = status(r)
    if st == "no_interface":
        return None
    if st == "evaluable":
        return float(r.get("contact_recall_5A") or 0.0)
    return 0.0


def recovered_at_least(r: dict, k: int) -> int | None:
    st = status(r)
    if st == "no_interface":
        return None
    if st != "evaluable":
        return 0
    return 1 if (r.get("n_recovered_contacts_5A") or 0) >= k else 0


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def load_maps(pct: int) -> tuple[dict, dict]:
    b = json.loads((RAW / f"interface_pct{pct}_base_ext.json").read_text())
    p = json.loads((RAW / f"interface_pct{pct}_psmp_ext.json").read_text())
    bmap = {r["name"]: r for r in b["results"] if "name" in r}
    pmap = {r["name"]: r for r in p["results"] if "name" in r}
    return bmap, pmap


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    summary = []
    by_case = []
    for pct in (10, 25, 50, 100):
        bmap, pmap = load_maps(pct)
        names = sorted(set(bmap) | set(pmap))
        assert len(names) == 50, f"expected 50 cases at {pct}%, got {len(names)}"

        def collect(mmap: dict, tag: str):
            stats = {
                "evaluable": 0,
                "missing_prediction": 0,
                "no_predicted_contact": 0,
                "no_interface": 0,
                "pipeline_error": 0,
                "other": 0,
            }
            fa_f1, fa_rec = [], []
            cond_f1 = []
            ge1, ge5 = [], []
            for n in names:
                r = mmap.get(n, {"name": n, "error": "absent_from_json"})
                st = status(r)
                stats[st] = stats.get(st, 0) + 1
                fv = f1_failure_aware(r)
                rv = recall_failure_aware(r)
                if fv is not None:
                    fa_f1.append(fv)
                if rv is not None:
                    fa_rec.append(rv)
                if st == "evaluable":
                    cond_f1.append(float(r["contact_f1_5A"]))
                g1 = recovered_at_least(r, 1)
                g5 = recovered_at_least(r, 5)
                if g1 is not None:
                    ge1.append(g1)
                if g5 is not None:
                    ge5.append(g5)
                by_case.append(
                    {
                        "Train_fraction": f"{pct}%",
                        "case": n,
                        "method": tag,
                        "status": st,
                        "F1_raw": r.get("contact_f1_5A"),
                        "F1_failure_aware": fv,
                        "recall_failure_aware": rv,
                        "n_recovered": r.get("n_recovered_contacts_5A"),
                        "n_native": r.get("n_native_contacts_5A"),
                        "n_predicted": r.get("n_predicted_contacts_5A"),
                    }
                )
            return stats, fa_f1, fa_rec, cond_f1, ge1, ge5

        bs, bf, br, bc, bg1, bg5 = collect(bmap, "base")
        ps, pf, pr, pc, pg1, pg5 = collect(pmap, "psmp")

        bok = {n for n, r in bmap.items() if status(r) == "evaluable"}
        pok = {n for n, r in pmap.items() if status(r) == "evaluable"}
        pair = sorted(bok & pok)
        pair_db = [float(bmap[n]["contact_f1_5A"]) for n in pair]
        pair_dp = [float(pmap[n]["contact_f1_5A"]) for n in pair]
        pair_dd = [pmap[n]["contact_f1_5A"] - bmap[n]["contact_f1_5A"] for n in pair]

        # failure-aware delta on shared defined cases (exclude no_interface either side)
        fa_delta = []
        for n in names:
            fb = f1_failure_aware(bmap.get(n, {"error": "absent"}))
            fp = f1_failure_aware(pmap.get(n, {"error": "absent"}))
            if fb is None or fp is None:
                continue
            fa_delta.append(fp - fb)

        summary.append(
            {
                "Train_fraction": f"{pct}%",
                "n_cases": 50,
                "n_ok_base": bs["evaluable"],
                "n_ok_psmp": ps["evaluable"],
                "n_missing_pred_base": bs["missing_prediction"] + bs.get("pipeline_error", 0),
                "n_missing_pred_psmp": ps["missing_prediction"] + ps.get("pipeline_error", 0),
                "n_no_pred_contact_base": bs["no_predicted_contact"],
                "n_no_pred_contact_psmp": ps["no_predicted_contact"],
                "n_no_interface_base": bs["no_interface"],
                "n_no_interface_psmp": ps["no_interface"],
                "n_pair": len(pair),
                "complete_metric_rate_base": bs["evaluable"] / 50,
                "complete_metric_rate_psmp": ps["evaluable"] / 50,
                "rate_recover_ge1_base": mean(bg1),
                "rate_recover_ge1_psmp": mean(pg1),
                "rate_recover_ge5_base": mean(bg5),
                "rate_recover_ge5_psmp": mean(pg5),
                "F1_failure_aware_mean_base": mean(bf),
                "F1_failure_aware_mean_psmp": mean(pf),
                "Delta_F1_failure_aware_mean": mean(fa_delta),
                "F1_conditional_mean_base": mean(bc),
                "F1_conditional_mean_psmp": mean(pc),
                "F1_paired_mean_base": mean(pair_db),
                "F1_paired_mean_psmp": mean(pair_dp),
                "Delta_F1_paired_mean": mean(pair_dd),
                "n_failure_aware_defined": len(fa_delta),
            }
        )

    sum_path = OUT / "TableS6b_failure_aware_summary.csv"
    with sum_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    case_path = OUT / "TableS6c_failure_aware_by_case.csv"
    with case_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_case[0].keys()))
        w.writeheader()
        w.writerows(by_case)

    (DATA / "failure_aware_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {sum_path}")
    print(f"wrote {case_path}")
    for row in summary:
        print(
            f"{row['Train_fraction']}: FA ΔF1={row['Delta_F1_failure_aware_mean']:.4f} "
            f"(n={row['n_failure_aware_defined']}) | "
            f"paired ΔF1={row['Delta_F1_paired_mean']:.4f} (n_pair={row['n_pair']}) | "
            f"n_ok base/psmp={row['n_ok_base']}/{row['n_ok_psmp']}"
        )


if __name__ == "__main__":
    main()
