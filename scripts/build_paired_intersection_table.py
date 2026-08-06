#!/usr/bin/env python3
"""Build Supplementary Table S5: paired-intersection low-data metrics."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from scipy import stats
import numpy as np

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
LOW = ROOT / "train/runs/rnp_real_lowdata/eval_work"
PAPER = ROOT / "nar_paper"
TBL = PAPER / "tables"
DATA = PAPER / "data"


def main() -> None:
    TBL.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows_out = []
    csv_rows = []

    for pct in [10, 25, 50, 100]:
        b = json.loads((LOW / f"interface_pct{pct}_base_ext.json").read_text())
        p = json.loads((LOW / f"interface_pct{pct}_psmp_ext.json").read_text())
        bm = {r["name"]: r for r in b["results"] if "contact_f1_5A" in r}
        pm = {r["name"]: r for r in p["results"] if "contact_f1_5A" in r}
        inter = sorted(set(bm) & set(pm))
        if len(inter) < 2:
            continue
        br = np.array([bm[n]["contact_recall_5A"] for n in inter])
        pr = np.array([pm[n]["contact_recall_5A"] for n in inter])
        bp = np.array([bm[n]["contact_precision_5A"] for n in inter])
        pp = np.array([pm[n]["contact_precision_5A"] for n in inter])
        bf = np.array([bm[n]["contact_f1_5A"] for n in inter])
        pf = np.array([pm[n]["contact_f1_5A"] for n in inter])
        w_r1 = stats.wilcoxon(pr, br, alternative="greater", zero_method="wilcox")
        w_f1 = stats.wilcoxon(pf, bf, alternative="greater", zero_method="wilcox")
        w_r2 = stats.wilcoxon(pr, br, alternative="two-sided", zero_method="wilcox")
        w_f2 = stats.wilcoxon(pf, bf, alternative="two-sided", zero_method="wilcox")
        row = {
            "pct": pct,
            "n_base": b["n_ok"],
            "n_psmp": p["n_ok"],
            "n_pair": len(inter),
            "mean_base_f1": float(bf.mean()),
            "mean_psmp_f1": float(pf.mean()),
            "delta_recall": float((pr - br).mean()),
            "delta_precision": float((pp - bp).mean()),
            "delta_f1": float((pf - bf).mean()),
            "wilcoxon_p_recall_onesided": float(w_r1.pvalue),
            "wilcoxon_p_f1_onesided": float(w_f1.pvalue),
            "wilcoxon_p_recall_twosided": float(w_r2.pvalue),
            "wilcoxon_p_f1_twosided": float(w_f2.pvalue),
        }
        rows_out.append(row)
        csv_rows.append([
            f"{pct}%", row["n_base"], row["n_psmp"], row["n_pair"],
            f"{row['mean_base_f1']:.4f}", f"{row['mean_psmp_f1']:.4f}",
            f"{row['delta_recall']:+.4f}", f"{row['delta_precision']:+.4f}", f"{row['delta_f1']:+.4f}",
            f"{row['wilcoxon_p_f1_onesided']:.6g}", f"{row['wilcoxon_p_f1_twosided']:.6g}",
            f"{row['wilcoxon_p_recall_onesided']:.6g}", f"{row['wilcoxon_p_recall_twosided']:.6g}",
        ])

    path = TBL / "TableS5_paired_intersection.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Train_fraction", "n_base", "n_psmp", "n_pair",
            "Base_F1_pair", "pSMP_F1_pair",
            "Delta_recall", "Delta_precision", "Delta_F1",
            "Wilcoxon_p_F1_onesided", "Wilcoxon_p_F1_twosided",
            "Wilcoxon_p_recall_onesided", "Wilcoxon_p_recall_twosided",
        ])
        w.writerows(csv_rows)

    (DATA / "paired_intersection_lowdata.json").write_text(
        json.dumps(rows_out, indent=2) + "\n", encoding="utf-8"
    )
    print(f"written {path}")
    for r in rows_out:
        print(r)


if __name__ == "__main__":
    main()
