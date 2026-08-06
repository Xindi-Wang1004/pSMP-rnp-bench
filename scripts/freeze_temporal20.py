#!/usr/bin/env python3
"""Freeze Temporal20 confirmatory shortlist from Table S7 (no recipe look-ahead).

Diversity caps: at most one structure per near-duplicate series (RNase P variants,
Retron-Eco8 variants, Cas9 DNA-bound, etc.).
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

PAPER = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper")
S7 = PAPER / "tables" / "TableS7_temporal_candidates.tsv"
OUT = PAPER / "data" / "temporal20_freeze.json"
OUT_TSV = PAPER / "tables" / "TableS7b_temporal20_shortlist.tsv"

TEMPORAL5 = {"10yz", "9zy0", "9o7t", "13fn", "9ype"}
RIBO = re.compile(r"ribosom|mitoribosom|70S|80S|polysome", re.I)
SKIP_TITLE = re.compile(
    r"AI-generated|ssDNA|B-form DNA|underwound DNA|virion",
    re.I,
)


def series_key(title: str, pdb_id: str) -> str:
    t = title.lower()
    if "rnase p" in t:
        return "rnase_p"
    if "retron-eco8" in t or "retron eco8" in t:
        return "retron_eco8"
    if "type iii-bv crispr" in t:
        return "crispr_iii_bv"
    if "uap56" in t:
        return "uap56"
    if "iscb" in t:
        return "iscb"
    if "nsun2" in t:
        return "nsun2"
    if "vif" in t and "apobec" in t:
        return "vif_apobec"
    if "telomerase" in t:
        return "telomerase"
    if "psecascade" in t or "tnsc" in t:
        return "psecascade"
    # fallback: first 3 title tokens
    toks = re.findall(r"[a-z0-9]+", t)
    return " ".join(toks[:3]) if toks else pdb_id


def main() -> None:
    rows = list(csv.DictReader(S7.open(), delimiter="\t"))
    cands = []
    for r in rows:
        pid = (r.get("pdb_id") or "").lower()
        if not pid or pid in TEMPORAL5:
            continue
        title = r.get("title") or ""
        if RIBO.search(title) or SKIP_TITLE.search(title):
            continue
        try:
            tok = int(float(r["est_tokens"]))
            res = float(r["resolution"] or 99)
            nprot = int(r["n_protein_entities"] or 99)
            nrna = int(r["n_rna_entities"] or 0)
            maxp = int(float(r["max_protein_len"] or 9999))
            maxr = int(float(r["max_rna_len"] or 9999))
        except Exception:
            continue
        if nrna < 1 or tok > 2000 or res > 3.5 or nprot > 6:
            continue
        if maxp > 1200 or maxr > 600:
            continue
        # prefer compact 1p1r slightly
        bonus = 0 if (nprot == 1 and nrna == 1) else 10
        score = (bonus, tok, nprot + nrna, res, pid)
        cands.append({**r, "pdb_id": pid, "series": series_key(title, pid), "_score": score})

    cands.sort(key=lambda x: x["_score"])
    picked = []
    seen_series: set[str] = set()
    for c in cands:
        if c["series"] in seen_series:
            continue
        seen_series.add(c["series"])
        picked.append(c)
        if len(picked) >= 20:
            break
    # if still short, allow a second pick from generic series keys only
    if len(picked) < 20:
        have = {p["pdb_id"] for p in picked}
        for c in cands:
            if c["pdb_id"] in have:
                continue
            if c["series"] in {
                "rnase_p",
                "retron_eco8",
                "crispr_iii_bv",
                "uap56",
                "iscb",
                "nsun2",
                "telomerase",
                "psecascade",
                "vif_apobec",
                "sars cov 2",
            }:
                continue  # keep hard caps on named families
            # also skip if this series already represented
            if c["series"] in {p["series"] for p in picked}:
                continue
            picked.append(c)
            have.add(c["pdb_id"])
            if len(picked) >= 20:
                break

    freeze = {
        "name": "Temporal20",
        "rule": (
            "Table S7; non-ribosome; est_tokens≤2000; n_protein≤6; exclude Temporal5; "
            "skip DNA-bound/AI/virion titles; ≤1 per named series (RNase P, Retron, …)"
        ),
        "n": len(picked),
        "pdb_ids": [p["pdb_id"] for p in picked],
        "series": [p["series"] for p in picked],
        "includes_temporal5_union_n": len(picked) + len(TEMPORAL5),
        "note": "Frozen confirmatory shortlist BEFORE any recipe change. Not used for model selection.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(freeze, indent=2) + "\n")

    fields = [
        "pdb_id",
        "release_date",
        "resolution",
        "est_tokens",
        "n_protein_entities",
        "n_rna_entities",
        "max_protein_len",
        "max_rna_len",
        "protein_chains",
        "rna_chains",
        "title",
        "series",
    ]
    with OUT_TSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        for p in picked:
            w.writerow(p)
    print(json.dumps(freeze, indent=2))
    print(f"wrote {OUT}\nwrote {OUT_TSV}")


if __name__ == "__main__":
    main()
