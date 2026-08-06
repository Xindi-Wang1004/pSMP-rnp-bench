#!/usr/bin/env python3
"""Re-score homology-hard (cluster-disjoint) candidates from five-sample FA JSONs.

Prefer locked primary round under eval_work_5seed_val50; failure-aware over
all candidates (missing / incomplete → 0). Does not tune recipes.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
TAB = PAPER / "tables"
DATA = PAPER / "data"
EVAL5 = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"

# Strict five-sample evaluability (aligned with Table 2 lock)
TOKENS_LARGE = 4000


def usable(r: dict) -> bool:
    """Case contributes a numeric FA score (not missing).

    Primary five-sample JSONs often omit n_samples; treat a present
    contact_f1_5A as scored. Missing CIF / explicit error → not usable (FA=0).
    """
    if r.get("contact_f1_5A") is not None:
        return True
    err = (r.get("error") or "")
    if err:
        return False
    ns = int(r.get("n_samples") or 0)
    tok = int(r.get("num_tokens") or r.get("tokens") or 0)
    if ns >= 5:
        return True
    if ns >= 1 and tok and tok <= TOKENS_LARGE:
        return True
    return False


def fa_f1(r: dict | None) -> float:
    if r is None:
        return 0.0
    if r.get("contact_f1_5A") is not None:
        return float(r["contact_f1_5A"])
    # missing / failed → 0 under failure-aware
    return 0.0


def load_map(tag: str) -> tuple[dict[str, dict], str]:
    # tag like pct10_base
    path = EVAL5 / f"interface_5seed_{tag}_val50_ext.json"
    if not path.is_file():
        # fallback legacy single-sample
        for p in (
            ROOT / f"train/runs/rnp_real_lowdata/eval_work/interface_{tag}_ext.json",
            PAPER / f"data/stage12_raw/eval_work/interface_{tag}_ext.json",
        ):
            if p.is_file():
                d = json.loads(p.read_text())
                return {r["name"]: r for r in d.get("results", []) if "name" in r}, str(p)
        return {}, ""
    d = json.loads(path.read_text())
    return {r["name"]: r for r in d.get("results", []) if "name" in r}, str(path)


def main() -> None:
    cand_csv = TAB / "TableS15_cluster_disjoint_candidates.csv"
    ids = []
    with cand_csv.open() as f:
        for row in csv.DictReader(f):
            if str(row.get("cluster_disjoint_candidate", "0")) in {"1", "True", "true"}:
                ids.append(row["val_case"])
    if not ids:
        raise SystemExit("empty hard list")

    tags = [
        "pct10_base",
        "pct10_psmp",
        "pct25_base",
        "pct25_psmp",
        "pct50_base",
        "pct50_psmp",
        "pct100_base",
        "pct100_psmp",
    ]
    rows = []
    by_case = []
    for tag in tags:
        mmap, src = load_map(tag)
        f1s = []
        n_ok = 0
        for name in ids:
            r = mmap.get(name)
            v = fa_f1(r)
            f1s.append(v)
            ok = r is not None and usable(r) and r.get("contact_f1_5A") is not None
            if ok:
                n_ok += 1
            by_case.append(
                {
                    "tag": tag,
                    "val_case": name,
                    "F1": v,
                    "usable": int(ok),
                    "n_samples": (r or {}).get("n_samples"),
                }
            )
        rows.append(
            {
                "tag": tag,
                "n_candidates": len(ids),
                "n_ok_strict5": n_ok,
                "F1_failure_aware_mean": sum(f1s) / len(f1s),
                "protocol": "five-sample FA (strict); missing→0",
                "source_json": src or "MISSING",
            }
        )

    paired = []
    for pct in (10, 25, 50, 100):
        b = next(r for r in rows if r["tag"] == f"pct{pct}_base")
        p = next(r for r in rows if r["tag"] == f"pct{pct}_psmp")
        paired.append(
            {
                "pct": pct,
                "F1_fa_base": b["F1_failure_aware_mean"],
                "F1_fa_psmp": p["F1_failure_aware_mean"],
                "delta_psmp_minus_base": p["F1_failure_aware_mean"] - b["F1_failure_aware_mean"],
                "n_candidates": len(ids),
                "protocol": "five-sample FA strict",
                "source_base": b["source_json"],
                "source_psmp": p["source_json"],
            }
        )

    TAB.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    out_b = TAB / "TableS15b_cluster_disjoint_failure_aware.csv"
    out_c = TAB / "TableS15c_cluster_disjoint_paired_delta.csv"
    out_case = TAB / "TableS15d_cluster_disjoint_by_case.csv"
    with out_b.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with out_c.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)
    with out_case.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_case[0].keys()))
        w.writeheader()
        w.writerows(by_case)
    summary = {
        "n_candidates": len(ids),
        "candidate_ids": ids,
        "paired": paired,
        "note": "Companion homology-hard readout from locked five-sample primary; not used for recipe selection.",
    }
    (DATA / "cluster_disjoint_score_summary_fivesample.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    mark = DATA / "pipeline_markers" / "boost_homology_fa5.ok"
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text("ok\n")
    print(json.dumps(paired, indent=2))
    print(f"wrote {out_b}\nwrote {out_c}\nwrote {out_case}")


if __name__ == "__main__":
    main()
