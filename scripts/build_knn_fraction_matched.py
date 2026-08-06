#!/usr/bin/env python3
"""Fraction-matched k-NN contact-transfer baseline (Table S14a).

For each frozen low-data pool (pct10/25/50/100), retrieve the best protein
neighbor **within that pool** (among PDBs with exported contact JSONs),
transfer contacts at identity ≥ thr, and score val50 under FA (missing→0).

Neighbor definition aligns with S11 spirit (protein partner identity), but the
search corpus is the fraction-matched train subset — not full train200.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from Bio.Align import PairwiseAligner

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
CONTACTS = PAPER / "data/contacts"
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
LOWDATA = ROOT / "train/data/rnp_real/lowdata"
OUT = PAPER / "tables"
THR = 0.40
PCTS = ("pct10", "pct25", "pct50", "pct100")


def aligner():
    aln = PairwiseAligner()
    aln.mode = "global"
    aln.match_score = 2
    aln.mismatch_score = -1
    aln.open_gap_score = -2
    aln.extend_gap_score = -0.5
    return aln


def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    aln = aligner()
    try:
        hit = next(iter(aln.align(a, b)))
    except Exception:
        return 0.0
    # identity over alignment length excluding pure gap columns
    s1, s2 = hit
    match = tot = 0
    for x, y in zip(s1, s2):
        if x == "-" and y == "-":
            continue
        tot += 1
        if x == y and x != "-":
            match += 1
    return match / tot if tot else 0.0


def index_map(src: str, dst: str) -> dict[int, int]:
    aln = aligner()
    try:
        hit = next(iter(aln.align(src, dst)))
    except Exception:
        return {}
    s1, s2 = hit
    m = {}
    i = j = 0
    for x, y in zip(s1, s2):
        if x != "-" and y != "-":
            m[i] = j
        if x != "-":
            i += 1
        if y != "-":
            j += 1
    return m


def load_contact(pdb: str, prot: str | None = None, rna: str | None = None):
    if prot and rna:
        for name in (f"{pdb}_{prot}_{rna}.json", f"{pdb.lower()}_{prot}_{rna}.json"):
            p = CONTACTS / name
            if p.exists():
                return json.loads(p.read_text()), p
    cands = sorted(CONTACTS.glob(f"{pdb.lower()}_*.json")) + sorted(CONTACTS.glob(f"{pdb}_*.json"))
    if not cands:
        return None, None
    best = None
    for p in cands:
        d = json.loads(p.read_text())
        if best is None or int(d.get("n_contacts") or 0) > int(best[0].get("n_contacts") or 0):
            best = (d, p)
    return best if best else (None, None)


def f1_sets(native: set, pred: set):
    if not native and not pred:
        return 0.0, 0.0, 0.0, 0
    tp = len(native & pred)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(native) if native else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, rec, prec, tp


def pool_pdbs(pct: str) -> list[str]:
    p = LOWDATA / pct / "pdb_list.txt"
    return [x.strip().lower() for x in p.read_text().split() if x.strip()]


def build_pool_index(pdbs: list[str]) -> dict[str, dict]:
    out = {}
    for pdb in pdbs:
        d, path = load_contact(pdb)
        if not d:
            continue
        ps = (d.get("protein_seq") or "").upper()
        if len(ps) < 20:
            continue
        out[pdb] = {"doc": d, "path": str(path), "protein_seq": ps, "rna_seq": (d.get("rna_seq") or "").upper()}
    return out


def score_case(val_case: dict, pool: dict[str, dict], thr: float = THR) -> dict:
    name = val_case["name"]
    pdb = val_case["pdb_id"].lower()
    val, _ = load_contact(pdb, val_case.get("protein_chain_id"), val_case.get("rna_chain_id"))
    if not val:
        return {
            "val_case": name,
            "status": "missing_val_contacts",
            "knn_f1": 0.0,
            "protein_identity": 0.0,
            "nearest_train_protein": "",
        }
    v_ps = (val.get("protein_seq") or "").upper()
    v_rs = (val.get("rna_seq") or "").upper()
    if not v_ps:
        return {
            "val_case": name,
            "status": "missing_sequences",
            "knn_f1": 0.0,
            "protein_identity": 0.0,
            "nearest_train_protein": "",
        }

    best_pdb, best_id, best_doc = "", 0.0, None
    for tpdb, meta in pool.items():
        ident = seq_identity(meta["protein_seq"], v_ps)
        if ident > best_id:
            best_id, best_pdb, best_doc = ident, tpdb, meta["doc"]

    if not best_pdb or best_id < thr:
        return {
            "val_case": name,
            "status": "no_protein_neighbor_in_pool" if not best_pdb or best_id <= 0 else "skipped_low_identity",
            "knn_f1": 0.0,
            "protein_identity": best_id,
            "nearest_train_protein": best_pdb,
            "thr": thr,
        }

    pmap = index_map(best_doc.get("protein_seq") or "", v_ps)
    rmap = index_map(best_doc.get("rna_seq") or "", v_rs) if (best_doc.get("rna_seq") and v_rs) else {}
    if len(pmap) < 0.5 * min(len(best_doc.get("protein_seq") or ""), len(v_ps)):
        return {
            "val_case": name,
            "status": "weak_protein_alignment",
            "knn_f1": 0.0,
            "protein_identity": best_id,
            "nearest_train_protein": best_pdb,
            "thr": thr,
        }

    native = {tuple(x) for x in (val.get("pairs") or [])}
    if not native:
        return {
            "val_case": name,
            "status": "no_interface",
            "knn_f1": 0.0,  # FA: no_interface → 0, stays in denom
            "protein_identity": best_id,
            "nearest_train_protein": best_pdb,
            "thr": thr,
        }
    pred = set()
    for i, j in best_doc.get("pairs") or []:
        ii, jj = pmap.get(i), rmap.get(j) if rmap else None
        if ii is None or jj is None:
            continue
        pred.add((ii, jj))
    f1, rec, prec, tp = f1_sets(native, pred)
    return {
        "val_case": name,
        "status": "scored",
        "knn_f1": f1,
        "knn_recall": rec,
        "knn_precision": prec,
        "knn_tp": tp,
        "protein_identity": best_id,
        "nearest_train_protein": best_pdb,
        "thr": thr,
        "n_map_p": len(pmap),
        "n_map_r": len(rmap),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cases = list(csv.DictReader(CASES.open(), delimiter="\t"))
    summary_rows = []
    all_case_rows = []

    for pct in PCTS:
        pdbs = pool_pdbs(pct)
        pool = build_pool_index(pdbs)
        print(pct, "pool_pdbs", len(pdbs), "with_usable_contacts", len(pool))
        rows = []
        for c in cases:
            r = score_case(c, pool, THR)
            r["train_fraction"] = pct
            r["n_pool_pdbs"] = len(pdbs)
            r["n_pool_with_contacts"] = len(pool)
            r["neighbor_definition"] = (
                f"best protein-seq identity within frozen {pct} pdb_list "
                f"(among contact-exportable pool members); thr={THR}; S11-style partner identity"
            )
            rows.append(r)
            all_case_rows.append(r)

        fa = [float(r["knn_f1"]) for r in rows]  # all 50; non-scored already 0
        scored = [r for r in rows if r["status"] == "scored"]
        summary_rows.append(
            {
                "train_fraction": pct,
                "n_pool_pdbs": len(pdbs),
                "n_pool_with_contacts": len(pool),
                "thr": THR,
                "n_val50": 50,
                "n_scored": len(scored),
                "n_no_neighbor_or_low_id": sum(
                    1
                    for r in rows
                    if r["status"] in ("no_protein_neighbor_in_pool", "skipped_low_identity")
                ),
                "n_no_interface": sum(1 for r in rows if r["status"] == "no_interface"),
                "F1_failure_aware_mean_full_cohort": sum(fa) / len(fa),
                "F1_conditional_mean_scored_only": (
                    sum(float(r["knn_f1"]) for r in scored) / len(scored) if scored else None
                ),
                "FA_denominator_policy": "all_50_including_no_interface_as_0",
                "data_budget_note": (
                    f"Retrieval corpus = frozen {pct} subset only (not full train200). "
                    "Comparable in data budget to Table 2 rows at the same fraction."
                ),
            }
        )
        print(" ", summary_rows[-1])

    # unify keys
    keys = sorted({k for r in all_case_rows for k in r})
    out_case = OUT / "TableS14a_knn_fraction_matched_by_case.csv"
    with out_case.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_case_rows)

    out_sum = OUT / "TableS14a_knn_fraction_matched_summary.csv"
    with out_sum.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    (OUT / "TableS14a_knn_failure_aware_summary.json").write_text(
        json.dumps(
            {
                "primary": "fraction_matched_retrieval_curve",
                "thr": THR,
                "coverage_rule": "protein sequence identity via global PairwiseAligner; transfer if identity>=thr",
                "rows": summary_rows,
                "note": (
                    "Full-cohort FA is comparable to Table 2 at the matching train fraction. "
                    "Conditional scored-only means are descriptive."
                ),
            },
            indent=2,
        )
    )
    print("wrote", out_case)
    print("wrote", out_sum)


if __name__ == "__main__":
    main()
