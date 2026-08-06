#!/usr/bin/env python3
"""List val50 five-sample cases with fewer than N sample CIFs (default 5).

Writes:
  data/fivesample_incomplete.json
  tables/TableS6f_fivesample_incomplete.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
EVAL = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sample", type=int, default=5)
    ap.add_argument("--max-tokens", type=float, default=4000.0)
    ap.add_argument("--tags", nargs="*", default=TAGS)
    args = ap.parse_args()

    cases = list(csv.DictReader(CASES.open(), delimiter="\t"))
    rows = []
    by_tag: dict[str, list[str]] = {}
    for tag in args.tags:
        pred = EVAL / f"pred_{tag}"
        miss = []
        for c in cases:
            name = c["name"]
            tok = float(c.get("num_tokens") or 0)
            if tok > args.max_tokens:
                continue
            n = len(list(pred.glob(f"**/{name}_sample_*.cif"))) if pred.is_dir() else 0
            if n >= args.n_sample:
                continue
            miss.append(name)
            rows.append(
                {
                    "tag": tag,
                    "name": name,
                    "num_tokens": tok,
                    "n_cif": n,
                    "need": args.n_sample,
                }
            )
        by_tag[tag] = miss

    out_json = {
        "n_sample": args.n_sample,
        "max_tokens": args.max_tokens,
        "by_tag": by_tag,
        "n_missing_total": len(rows),
        "n_missing_by_tag": {t: len(v) for t, v in by_tag.items()},
    }
    PAPER.joinpath("data").mkdir(parents=True, exist_ok=True)
    PAPER.joinpath("tables").mkdir(parents=True, exist_ok=True)
    (PAPER / "data/fivesample_incomplete.json").write_text(json.dumps(out_json, indent=2) + "\n")
    csv_path = PAPER / "tables/TableS6f_fivesample_incomplete.csv"
    with csv_path.open("w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        else:
            f.write("tag,name,num_tokens,n_cif,need\n")
    print(json.dumps(out_json["n_missing_by_tag"], indent=2))
    print(f"total_missing={out_json['n_missing_total']} -> {csv_path}")


if __name__ == "__main__":
    main()
