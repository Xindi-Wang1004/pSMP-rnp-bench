#!/usr/bin/env python3
"""Paired statistics and figures for pSMP paper (test10 / dev40)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
FIG = ROOT / "nar_paper/figures"
DATA = ROOT / "nar_paper/data"
FIG.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300})


def load_pairs(base_json: Path, psmp_json: Path) -> list[dict]:
    base = {r["name"]: r for r in json.loads(base_json.read_text())["results"]}
    psmp = {r["name"]: r for r in json.loads(psmp_json.read_text())["results"]}
    pairs = []
    for name in sorted(set(base) & set(psmp)):
        b, p = base[name], psmp[name]
        if "contact_recall_5A" in b and "contact_recall_5A" in p:
            pairs.append(
                {
                    "name": name,
                    "base_recall": b["contact_recall_5A"],
                    "psmp_recall": p["contact_recall_5A"],
                    "base_precision": b.get("contact_precision_5A"),
                    "psmp_precision": p.get("contact_precision_5A"),
                    "base_f1": b.get("contact_f1_5A"),
                    "psmp_f1": p.get("contact_f1_5A"),
                }
            )
    return pairs


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = [float(np.mean(values[rng.integers(0, n, n)])) for _ in range(n_boot)]
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def analyze_split(split: str, base_json: Path, psmp_json: Path) -> dict:
    pairs = load_pairs(base_json, psmp_json)
    base_r = np.array([p["base_recall"] for p in pairs])
    psmp_r = np.array([p["psmp_recall"] for p in pairs])
    delta = psmp_r - base_r

    wilcoxon = stats.wilcoxon(psmp_r, base_r, alternative="greater", zero_method="wilcox")
    ci_lo, ci_hi = bootstrap_ci(delta)

    base_f1 = np.array([p["base_f1"] for p in pairs if p["base_f1"] is not None])
    psmp_f1 = np.array([p["psmp_f1"] for p in pairs if p["psmp_f1"] is not None])
    f1_wilcoxon = stats.wilcoxon(psmp_f1, base_f1, alternative="greater", zero_method="wilcox") if len(base_f1) else None

    return {
        "split": split,
        "n_pairs": len(pairs),
        "pairs": pairs,
        "mean_delta_recall": float(np.mean(delta)),
        "bootstrap_95ci_delta_recall": [ci_lo, ci_hi],
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_pvalue": float(wilcoxon.pvalue),
        "mean_base_recall": float(np.mean(base_r)),
        "mean_psmp_recall": float(np.mean(psmp_r)),
        "mean_base_f1": float(np.mean(base_f1)) if len(base_f1) else None,
        "mean_psmp_f1": float(np.mean(psmp_f1)) if len(psmp_f1) else None,
        "wilcoxon_f1_pvalue": float(f1_wilcoxon.pvalue) if f1_wilcoxon else None,
    }


def plot_paired_scatter(pairs: list[dict], title: str, out_stem: str) -> None:
    x = np.array([p["base_recall"] for p in pairs])
    y = np.array([p["psmp_recall"] for p in pairs])
    lim = max(0.05, float(max(x.max(), y.max()) * 1.15))

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.scatter(x, y, s=60, c="#DD8452", edgecolors="#333", zorder=3)
    ax.plot([0, lim], [0, lim], "--", color="#999", lw=1)
    for p in pairs:
        ax.annotate(p["name"].split("_")[0], (p["base_recall"], p["psmp_recall"]), fontsize=6, alpha=0.8)
    ax.set_xlim(-0.002, lim)
    ax.set_ylim(-0.002, lim)
    ax.set_xlabel("Base contact recall @5Å")
    ax.set_ylabel("pSMP contact recall @5Å")
    ax.set_title(title, fontweight="bold")
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"{out_stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pr_curve(base_json: Path, psmp_json: Path, out_stem: str) -> None:
    def pts(d):
        ok = [r for r in json.loads(d.read_text())["results"] if "contact_precision_5A" in r]
        return np.array([r["contact_recall_5A"] for r in ok]), np.array([r["contact_precision_5A"] for r in ok])

    br, bp = pts(base_json)
    pr, pp = pts(psmp_json)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(bp, br, s=50, c="#4C72B0", label="Base", alpha=0.85, edgecolors="#333", linewidths=0.4)
    ax.scatter(pp, pr, s=50, c="#DD8452", label="pSMP", alpha=0.85, edgecolors="#333", linewidths=0.4)
    for label, px, py, col in [
        ("Base mean", bp.mean(), br.mean(), "#4C72B0"),
        ("pSMP mean", pp.mean(), pr.mean(), "#DD8452"),
    ]:
        ax.scatter(px, py, s=120, marker="*", c=col, edgecolors="#333", zorder=4, label=label)
    ax.set_xlabel("Contact precision @5Å")
    ax.set_ylabel("Contact recall @5Å")
    ax.set_title("Precision–recall (per complex)", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / f"{out_stem}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{out_stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    runs = Path("/home/wangxindi/RNA_Protein/fusai/train/runs")
    val = Path("/home/wangxindi/RNA_Protein/fusai/train/data/rnp_real/val_protenix_inputs")
    pt = runs / "paper_tables"

    configs = [
        ("test10", pt / "interface_5seed_base_pct10_test10_ext.json", pt / "interface_5seed_allmix_pct10_test10_ext.json"),
        ("dev40", pt / "interface_5seed_base_pct10_dev40_ext.json", pt / "interface_5seed_allmix_pct10_dev40_ext.json"),
    ]

    summary = {}
    for split, bj, pj in configs:
        if not bj.exists() or not pj.exists():
            print(f"skip {split}: missing {bj} or {pj}")
            continue
        res = analyze_split(split, bj, pj)
        summary[split] = {k: v for k, v in res.items() if k != "pairs"}
        plot_paired_scatter(res["pairs"], f"Paired recall: {split}", f"FigS_paired_recall_{split}")

    if configs[0][1].exists() and configs[0][2].exists():
        plot_pr_curve(configs[0][1], configs[0][2], "FigS_pr_curve_test10")

    out = DATA / "paper_statistics.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"written {out}")


if __name__ == "__main__":
    main()
