#!/usr/bin/env python3
"""k-NN contact-transfer baseline with sequence alignment (not exact length match).

For each val50 case with protein identity ≥ thr to a train neighbor:
  1) load val + train contact JSONs (pairs + sequences when present)
  2) align protein (and RNA) sequences with Biopython PairwiseAligner
  3) map train contact indices → val indices; score F1 vs val native contacts

Failures are recorded per-case; script always writes Table S14 + summary JSON.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from Bio.Align import PairwiseAligner

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "stage12_raw"
_CONTACT_CANDIDATES = [
    Path("/home/wangxindi/RNA_Protein/fusai/nar_paper/data/contacts"),
    RAW / "contacts",
]
AUDIT = RAW / "audit" / "TableS11_partner_sequence_audit.tsv"
if not AUDIT.exists():
    AUDIT = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper_audit/out/TableS11_partner_sequence_audit.tsv")
CASES = RAW / "splits" / "cases.tsv"
if not CASES.exists():
    CASES = Path("/home/wangxindi/RNA_Protein/fusai/train/data/rnp_real/val_protenix_inputs/cases.tsv")
CONTACTS = next((p for p in _CONTACT_CANDIDATES if p.exists()), RAW / "contacts")
OUT = ROOT / "tables"
THR_DEFAULT = 0.40


def load_contact_json(pdb: str, prot: str | None = None, rna: str | None = None) -> dict | None:
    cands: list[Path] = []
    if prot and rna:
        for name in (
            f"{pdb}_{prot}_{rna}.json",
            f"{pdb.lower()}_{prot}_{rna}.json",
        ):
            p = CONTACTS / name
            if p.exists():
                cands.append(p)
    cands.extend(sorted(CONTACTS.glob(f"{pdb.lower()}_*.json")))
    cands.extend(sorted(CONTACTS.glob(f"{pdb}_*.json")))
    if not cands:
        return None
    payloads = []
    for p in cands:
        try:
            d = json.loads(p.read_text())
            payloads.append(d)
        except Exception:
            continue
    if not payloads:
        return None
    # prefer files with contacts + sequences
    payloads.sort(
        key=lambda d: (
            int(d.get("n_contacts") or 0) > 0,
            bool(d.get("protein_seq")),
            int(d.get("n_contacts") or 0),
        ),
        reverse=True,
    )
    return payloads[0]


def build_aligner(mode: str = "protein") -> PairwiseAligner:
    aln = PairwiseAligner()
    aln.mode = "global"
    if mode == "protein":
        aln.match_score = 2
        aln.mismatch_score = -1
        aln.open_gap_score = -2
        aln.extend_gap_score = -0.5
    else:
        aln.match_score = 1
        aln.mismatch_score = -1
        aln.open_gap_score = -2
        aln.extend_gap_score = -0.5
    return aln


def index_map(seq_a: str, seq_b: str, mode: str = "protein") -> dict[int, int]:
    """Map indices in seq_a (train) -> seq_b (val) via best global alignment."""
    if not seq_a or not seq_b:
        return {}
    if seq_a == seq_b:
        return {i: i for i in range(len(seq_a))}
    aln = build_aligner(mode)
    alignments = aln.align(seq_a, seq_b)
    best = next(iter(alignments))
    # Bio.Align Alignment: best[0] / best[1] are aligned strings in recent biopython
    try:
        a_aln = str(best[0])
        b_aln = str(best[1])
    except Exception:
        # fallback older API
        a_aln, b_aln = best.aligned and (None, None)
        raise
    mp: dict[int, int] = {}
    ia = ib = 0
    for ca, cb in zip(a_aln, b_aln):
        if ca != "-" and cb != "-":
            mp[ia] = ib
            ia += 1
            ib += 1
        elif ca == "-":
            ib += 1
        elif cb == "-":
            ia += 1
    return mp


def f1_from_sets(native: set[tuple[int, int]], pred: set[tuple[int, int]]):
    if not native:
        return float("nan"), float("nan"), float("nan"), 0
    tp = len(native & pred)
    recall = tp / len(native)
    precision = tp / len(pred) if pred else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return f1, recall, precision, tp


def _aa_seq(residues) -> str:
    from Bio.Data.IUPACData import protein_letters_3to1

    out = []
    for r in residues:
        key = r.resname.strip().capitalize()
        out.append(protein_letters_3to1.get(key, "X"))
    return "".join(out)


def _rna_seq(residues) -> str:
    out = []
    for r in residues:
        name = r.resname.strip().upper().replace("R", "")
        if name in ("A", "U", "G", "C", "I", "T"):
            out.append("U" if name == "T" else name)
        else:
            out.append(name[-1] if name else "N")
    return "".join(out)


def sequences_from_cif(cif_path: str, pdb: str, pc: str, rc: str) -> tuple[str, str]:
    import sys

    fusai = Path("/home/wangxindi/RNA_Protein/fusai/train")
    if str(fusai) not in sys.path:
        sys.path.insert(0, str(fusai))
    from psmp.eval_interface import _parse_cif, find_chain, polymer_residues

    nat = _parse_cif(cif_path, pdb.lower())
    model = list(nat)[0]
    p_chain = find_chain(model, pc)
    r_chain = find_chain(model, rc)
    if p_chain is None or r_chain is None:
        return "", ""
    return _aa_seq(polymer_residues(p_chain, "protein")), _rna_seq(polymer_residues(r_chain, "rna"))


def ensure_sequences(payload: dict, case_row: dict | None = None) -> tuple[str, str]:
    ps = payload.get("protein_seq") or ""
    rs = payload.get("rna_seq") or ""
    if ps and rs:
        return ps, rs
    # fill from CIF when possible (val natives / train pool)
    cif = payload.get("source_cif")
    pdb = payload.get("pdb")
    pc = payload.get("protein_chain")
    rc = payload.get("rna_chain")
    if (not ps or not rs) and case_row and case_row.get("native_cif_gz"):
        cif = cif or case_row["native_cif_gz"]
        pdb = pdb or case_row.get("pdb_id")
        pc = pc or case_row.get("protein_chain_id")
        rc = rc or case_row.get("rna_chain_id")
    if cif and pdb and pc and rc and Path(cif).exists():
        try:
            ps2, rs2 = sequences_from_cif(str(cif), str(pdb), str(pc), str(rc))
            ps = ps or ps2
            rs = rs or rs2
        except Exception:
            pass
    return ps, rs


def main(thr: float = THR_DEFAULT) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = {r["name"]: r for r in csv.DictReader(CASES.open(), delimiter="\t")}
    audit = list(csv.DictReader(AUDIT.open(), delimiter="\t"))
    rows = []

    for a in audit:
        name = a["val_case"]
        c = cases[name]
        pid = float(a["protein_identity"] or 0)
        nn_pdb = (a.get("nearest_train_protein") or "").strip()
        status = "skipped_low_identity"
        reason = ""
        f1 = rec = prec = tp = None
        n_map_p = n_map_r = None

        if not nn_pdb:
            status = "no_protein_neighbor"
        elif pid < thr:
            status = "skipped_low_identity"
        else:
            tr = load_contact_json(nn_pdb)
            val = load_contact_json(c["pdb_id"], c["protein_chain_id"], c["rna_chain_id"])
            if tr is None:
                status = "missing_train_contacts"
                reason = f"no_json_for_{nn_pdb}"
            elif val is None:
                status = "missing_val_contacts"
                reason = "val_contact_json_absent"
            else:
                t_ps, t_rs = ensure_sequences(tr)
                v_ps, v_rs = ensure_sequences(val, c)
                # if sequences are dummy poly-A of different lengths, still try alignment
                if not t_ps or not v_ps:
                    status = "missing_sequences"
                    reason = "protein_seq_absent"
                else:
                    pmap = index_map(t_ps, v_ps, "protein")
                    rmap = index_map(t_rs, v_rs, "rna") if (t_rs and v_rs) else {}
                    n_map_p, n_map_r = len(pmap), len(rmap)
                    if len(pmap) < 0.5 * min(len(t_ps), len(v_ps)):
                        status = "weak_protein_alignment"
                        reason = f"mapped={len(pmap)} min_len={min(len(t_ps), len(v_ps))}"
                    else:
                        native = {tuple(x) for x in val["pairs"]}
                        pred = set()
                        for i, j in tr["pairs"]:
                            ii = pmap.get(i)
                            jj = rmap.get(j) if rmap else (j if j < int(val.get("n_rna") or 0) else None)
                            if ii is None or jj is None:
                                continue
                            pred.add((ii, jj))
                        f1, rec, prec, tp = f1_from_sets(native, pred)
                        status = "scored"
                        reason = f"p_map={n_map_p},r_map={n_map_r},pred_n={len(pred)}"

        rows.append(
            {
                "val_case": name,
                "val_pdb": c["pdb_id"],
                "nearest_train_protein": nn_pdb,
                "protein_identity": pid,
                "thr": thr,
                "status": status,
                "reason": reason,
                "knn_f1": f1,
                "knn_recall": rec,
                "knn_precision": prec,
                "knn_tp": tp,
                "both_ge40": a.get("both_ge40"),
            }
        )

    out_case = OUT / "TableS14_knn_contact_transfer_by_case.csv"
    with out_case.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scored = [r for r in rows if r["status"] == "scored" and r["knn_f1"] == r["knn_f1"]]
    status_counts = Counter(r["status"] for r in rows)
    summary = {
        "thr": thr,
        "n_val": len(rows),
        "n_scored": len(scored),
        "mean_knn_f1_scored": (sum(float(r["knn_f1"]) for r in scored) / len(scored)) if scored else None,
        "median_knn_f1_scored": (
            sorted(float(r["knn_f1"]) for r in scored)[len(scored) // 2] if scored else None
        ),
        "status_counts": dict(status_counts),
        "contacts_dir": str(CONTACTS),
        "n_contact_files": len(list(CONTACTS.glob("*.json"))) if CONTACTS.exists() else 0,
    }
    (ROOT / "data" / "knn_baseline_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_case}")


if __name__ == "__main__":
    main()
