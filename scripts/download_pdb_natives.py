#!/usr/bin/env python3
"""Download PDB mmCIF files for RNP-ContactBench portable case lists.

Does not redistribute PDB files; fetches from RCSB for local use under PDB terms.

Example:
  python3 scripts/download_pdb_natives.py \\
    --cases splits/val50/cases_portable.tsv \\
    --out_dir data/natives/val50
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path


RCSB_CIF = "https://files.rcsb.org/download/{pdb_id}.cif.gz"


def read_pdb_ids(cases: Path) -> list[str]:
    ids: list[str] = []
    if cases.suffix.lower() == ".txt":
        for line in cases.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#"):
                ids.append(s.lower())
        return sorted(set(ids))

    with cases.open() as f:
        # try tab then comma
        sample = f.read(2048)
        f.seek(0)
        delim = "\t" if "\t" in sample.splitlines()[0] else ","
        reader = csv.DictReader(f, delimiter=delim)
        if not reader.fieldnames:
            raise SystemExit(f"empty table: {cases}")
        key = None
        for cand in ("pdb_id", "pdb", "PDB_ID"):
            if cand in reader.fieldnames:
                key = cand
                break
        if key is None:
            raise SystemExit(f"no pdb_id column in {cases}; columns={reader.fieldnames}")
        for row in reader:
            v = (row.get(key) or "").strip()
            if v:
                ids.append(v.lower())
    return sorted(set(ids))


def download(pdb_id: str, out_dir: Path, timeout: float = 60.0) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{pdb_id}.cif.gz"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    url = RCSB_CIF.format(pdb_id=pdb_id.upper())
    req = urllib.request.Request(url, headers={"User-Agent": "pSMP-rnp-bench/download_pdb_natives"})
    with urllib.request.urlopen(req, timeout=timeout) as r, dest.open("wb") as f:
        f.write(r.read())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, required=True, help="cases_portable.tsv or pdb_list.txt")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="optional cap for smoke tests")
    args = ap.parse_args()

    ids = read_pdb_ids(args.cases)
    if args.limit > 0:
        ids = ids[: args.limit]
    print(f"downloading {len(ids)} mmCIF.gz -> {args.out_dir}")
    ok = 0
    for i, pdb_id in enumerate(ids, 1):
        try:
            path = download(pdb_id, args.out_dir)
            ok += 1
            print(f"[{i}/{len(ids)}] OK {pdb_id} -> {path.name}")
        except Exception as e:
            print(f"[{i}/{len(ids)}] FAIL {pdb_id}: {e}", file=sys.stderr)
    print(f"done: {ok}/{len(ids)}")
    return 0 if ok == len(ids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
