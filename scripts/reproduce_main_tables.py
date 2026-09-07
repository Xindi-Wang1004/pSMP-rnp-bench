#!/usr/bin/env python3
"""Smoke-check deposited companions for main RNP-ContactBench tables."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path

REQUIRED = [
    "TableS6l_geometry_coverage_failure_decomp.csv",
    "TableS6m_token_stratum_FA_pct10.csv",
    "TableS9d_temporal34_locked_summary.csv",
    "TableS9e_temporal34_lowhomology_tracks.csv",
    "TableS19d_temporal34_exposure_compact.csv",
    "temporal34_proteinlt40_track_v0.1.json",
]

def ok(msg):
    print("OK ", msg)

def bad(msg):
    print("FAIL", msg)
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables-dir", type=Path, default=Path("tables"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    td = args.tables_dir
    rc = 0
    for name in REQUIRED:
        p = td / name
        if not p.exists():
            rc = bad(f"missing {p}")
        else:
            ok(f"found {name}")

    def read_csv(name):
        with (td / name).open() as f:
            return list(csv.DictReader(f))

    try:
        s9d = read_csv("TableS9d_temporal34_locked_summary.csv")
        row = next(r for r in s9d if "Temporal34" in r.get("cohort", ""))
        n = int(float(row["n_cases"]))
        if n != 34:
            rc = bad(f"Temporal34 n_cases={n} expected 34")
        else:
            ok("Temporal34 n=34")
        s9e = read_csv("TableS9e_temporal34_lowhomology_tracks.csv")
        lh = next(r for r in s9e if "lt40" in r.get("cohort", ""))
        if int(float(lh["n"])) < 10:
            rc = bad("low-homology track too small")
        else:
            ok(f"low-homology track n={lh['n']}")
        s19 = read_csv("TableS19d_temporal34_exposure_compact.csv")
        ok(f"exposure compact rows={len(s19)}")
        track = json.loads((td / "temporal34_proteinlt40_track_v0.1.json").read_text())
        ok(f"track freeze n={track.get('n')}")
    except Exception as e:
        rc = bad(f"parse error: {e}")
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "REPRO_SMOKE.txt").write_text("pass\n" if rc == 0 else "fail\n")
    sys.exit(rc)

if __name__ == "__main__":
    main()
