#!/usr/bin/env python3
"""Paired-intersection Table 2/S5 from val50 five-sample extended JSONs.

Expects (after reinfer + scoring):
  eval_work_5seed_val50/interface_5seed_pct{10,25,50,100}_{base,psmp}_val50_ext.json
or paper_tables paths configured below.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
EVAL = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"
PAPER = ROOT / "nar_paper"
TBL = PAPER / "tables"
DATA = PAPER / "data"


def load_ok(path: Path) -> dict[str, dict]:
    d = json.loads(path.read_text())
    return {r["name"]: r for r in d["results"] if "contact_f1_5A" in r}


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows_out = []
    csv_rows = []
    for pct in [10, 25, 50, 100]:
        bp = EVAL / f"interface_5seed_pct{pct}_base_val50_ext.json"
        pp = EVAL / f"interface_5seed_pct{pct}_psmp_val50_ext.json"
        # alternate naming from shell tag pct10_base
        if not bp.exists():
            bp = EVAL / f"interface_5seed_pct{pct}_base_ext.json"
        if not pp.exists():
            pp = EVAL / f"interface_5seed_pct{pct}_psmp_ext.json"
        if not bp.exists() or not pp.exists():
            print(f"MISSING {pct}% -> {bp} / {pp}")
            continue
        bm, pm = load_ok(bp), load_ok(pp)
        inter = sorted(set(bm) & set(pm))
        bf = np.array([bm[n]["contact_f1_5A"] for n in inter])
        pf = np.array([pm[n]["contact_f1_5A"] for n in inter])
        br = np.array([bm[n]["contact_recall_5A"] for n in inter])
        pr = np.array([pm[n]["contact_recall_5A"] for n in inter])
        w_f1 = stats.wilcoxon(pf, bf, alternative="greater", zero_method="wilcox")
        w_f2 = stats.wilcoxon(pf, bf, alternative="two-sided", zero_method="wilcox")
        row = {
            "pct": pct,
            "n_base": len(bm),
            "n_psmp": len(pm),
            "n_pair": len(inter),
            "mean_base_f1": float(bf.mean()) if len(inter) else None,
            "mean_psmp_f1": float(pf.mean()) if len(inter) else None,
            "delta_f1": float((pf - bf).mean()) if len(inter) else None,
            "wilcoxon_p_f1_onesided": float(w_f1.pvalue) if len(inter) >= 2 else None,
            "wilcoxon_p_f1_twosided": float(w_f2.pvalue) if len(inter) >= 2 else None,
            "mean_delta_recall": float((pr - br).mean()) if len(inter) else None,
        }
        rows_out.append(row)
        csv_rows.append([
            f"{pct}%", row["n_base"], row["n_psmp"], row["n_pair"],
            f"{row['mean_base_f1']:.4f}", f"{row['mean_psmp_f1']:.4f}", f"{row['delta_f1']:+.4f}",
            f"{row['wilcoxon_p_f1_onesided']:.6g}", f"{row['wilcoxon_p_f1_twosided']:.6g}",
        ])
        print(row)

    out = TBL / "Table2_fivesample_val50_paired.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Train_fraction", "n_base", "n_psmp", "n_pair",
            "Base_F1_pair", "pSMP_F1_pair", "Delta_F1",
            "Wilcoxon_p_F1_onesided", "Wilcoxon_p_F1_twosided",
        ])
        w.writerows(csv_rows)
    (DATA / "paired_intersection_fivesample_val50.json").write_text(
        json.dumps(rows_out, indent=2) + "\n"
    )
    print("written", out)


if __name__ == "__main__":
    main()
