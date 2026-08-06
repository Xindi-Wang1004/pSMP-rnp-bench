#!/usr/bin/env python3
"""BH / Bonferroni correction for Table S5 one-sided paired F1 p-values."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TBL = ROOT / "tables"
SRC = TBL / "TableS5_paired_intersection.csv"
OUT = TBL / "TableS5b_multiple_testing.csv"


def bh_adjust(pvals: list[float]) -> list[float]:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * n / rank
        prev = min(prev, val)
        adj[order[i]] = min(prev, 1.0)
    return adj.tolist()


def main() -> None:
    rows = list(csv.DictReader(SRC.open()))
    p_f1 = [float(r["Wilcoxon_p_F1_onesided"]) for r in rows]
    p_rec = [float(r["Wilcoxon_p_recall_onesided"]) for r in rows]
    bh_f1 = bh_adjust(p_f1)
    bh_rec = bh_adjust(p_rec)
    n = len(p_f1)
    bonf_f1 = [min(1.0, p * n) for p in p_f1]
    bonf_rec = [min(1.0, p * n) for p in p_rec]

    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "Train_fraction",
            "n_pair",
            "Wilcoxon_p_F1_onesided",
            "BH_p_F1_onesided",
            "Bonferroni_p_F1_onesided",
            "Wilcoxon_p_recall_onesided",
            "BH_p_recall_onesided",
            "Bonferroni_p_recall_onesided",
            "Significant_nominal_F1_0.05",
            "Significant_BH_F1_0.05",
            "Significant_Bonferroni_F1_0.05",
        ])
        for r, a, b, ar, br in zip(rows, bh_f1, bonf_f1, bh_rec, bonf_rec):
            p = float(r["Wilcoxon_p_F1_onesided"])
            w.writerow([
                r["Train_fraction"],
                r["n_pair"],
                f"{p:.6g}",
                f"{a:.6g}",
                f"{b:.6g}",
                f"{float(r['Wilcoxon_p_recall_onesided']):.6g}",
                f"{ar:.6g}",
                f"{br:.6g}",
                "yes" if p < 0.05 else "no",
                "yes" if a < 0.05 else "no",
                "yes" if b < 0.05 else "no",
            ])
    print(f"written {OUT}")
    for r, a, b in zip(rows, bh_f1, bonf_f1):
        print(r["Train_fraction"], "raw", r["Wilcoxon_p_F1_onesided"], "BH", f"{a:.4f}", "Bonf", f"{b:.4f}")


if __name__ == "__main__":
    main()
