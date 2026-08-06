#!/usr/bin/env python3
"""Score frozen cluster-disjoint candidates from existing interface_*_ext JSONs (CPU).

Does not tune recipes — list freeze + readout only.
Sources (first existing wins per method family):
  - five-sample: eval_work_5seed_val50/interface_5seed_{tag}_val50_ext.json
  - single-sample deposited: data/stage12_raw/eval_work/interface_{tag}_ext.json
  - single-sample server: train/runs/rnp_real_lowdata/eval_work/interface_{tag}_ext.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
CAND = PAPER / "data" / "cluster_disjoint_candidates.json"
TAB = PAPER / "tables"
DATA = PAPER / "data"

TAGS = [
    "pct10_base",
    "pct10_psmp",
    "pct25_base",
    "pct25_psmp",
    "pct50_base",
    "pct50_psmp",
    "pct100_base",
    "pct100_psmp",
]


def fa_f1(r: dict) -> float | None:
    if r.get("contact_f1_5A") is not None:
        return float(r["contact_f1_5A"])
    n_nat = r.get("n_native_contacts_5A")
    if n_nat == 0:
        return None
    return 0.0


def load_map(tag: str) -> tuple[dict[str, dict], str]:
    candidates = [
        ROOT / f"train/runs/rnp_real_lowdata/eval_work_5seed_val50/interface_5seed_{tag}_val50_ext.json",
        PAPER / f"data/stage12_raw/eval_work/interface_{tag}_ext.json",
        ROOT / f"train/runs/rnp_real_lowdata/eval_work/interface_{tag}_ext.json",
        ROOT / f"train/runs/rnp_real_lowdata/eval_work/interface_{tag}_ext.json",
    ]
    for p in candidates:
        if p.is_file():
            d = json.loads(p.read_text())
            return {r["name"]: r for r in d.get("results", []) if "name" in r}, str(p)
    return {}, ""


def main() -> None:
    if not CAND.is_file():
        # build from TableS15 if JSON missing
        csv_path = TAB / "TableS15_cluster_disjoint_candidates.csv"
        if not csv_path.is_file():
            raise SystemExit(f"missing {CAND} and {csv_path}")
        ids = []
        with csv_path.open() as f:
            for row in csv.DictReader(f):
                if str(row.get("cluster_disjoint_candidate", "0")) in {"1", "True", "true"}:
                    ids.append(row["val_case"])
        manifest = {"candidate_ids": ids}
    else:
        manifest = json.loads(CAND.read_text())
    ids = list(manifest.get("candidate_ids") or [])
    if not ids:
        raise SystemExit("empty candidate list")

    rows = []
    for tag in TAGS:
        mmap, src = load_map(tag)
        f1s = []
        n_ok = 0
        n_miss = 0
        for name in ids:
            r = mmap.get(name)
            if r is None:
                n_miss += 1
                f1s.append(0.0)
                continue
            v = fa_f1(r)
            if v is None:
                continue  # no_interface
            f1s.append(v)
            if r.get("contact_f1_5A") is not None:
                n_ok += 1
        rows.append(
            {
                "tag": tag,
                "n_candidates": len(ids),
                "n_scored_in_source": n_ok,
                "n_absent_in_source": n_miss,
                "F1_failure_aware_mean": (sum(f1s) / len(f1s)) if f1s else float("nan"),
                "source_json": src or "MISSING",
            }
        )

    # paired base vs psmp deltas on shared pct
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
                "source_base": b["source_json"],
                "source_psmp": p["source_json"],
            }
        )

    TAB.mkdir(parents=True, exist_ok=True)
    out_csv = TAB / "TableS15b_cluster_disjoint_failure_aware.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    out_pair = TAB / "TableS15c_cluster_disjoint_paired_delta.csv"
    with out_pair.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)
    (DATA / "cluster_disjoint_score_summary.json").write_text(
        json.dumps({"by_tag": rows, "paired": paired, "candidate_ids": ids}, indent=2) + "\n"
    )
    print(f"wrote {out_csv}")
    print(f"wrote {out_pair}")
    for r in paired:
        print(
            f"pct{r['pct']}: FA ΔF1={r['delta_psmp_minus_base']:+.4f} "
            f"(base={r['F1_fa_base']:.4f} psmp={r['F1_fa_psmp']:.4f})"
        )


if __name__ == "__main__":
    main()
