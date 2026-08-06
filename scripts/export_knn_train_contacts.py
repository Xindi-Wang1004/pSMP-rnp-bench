#!/usr/bin/env python3
"""Export/refresh contact JSONs for k-NN: val50 + nearest-train, max-contact chain pairs."""
from __future__ import annotations

import csv
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
sys.path.insert(0, str(ROOT / "train"))

from Bio.Data.IUPACData import protein_letters_3to1  # noqa: E402
from psmp.eval_interface import (  # noqa: E402
    _interface_atoms,
    _parse_cif,
    classify_chain,
    find_chain,
    polymer_residues,
)

OUT = ROOT / "nar_paper/data/contacts"
CIF_POOL = Path("/home/wangxindi/RNA_Protein/rnp_data/pdb_rna_protein_contact_cif")
AUDIT = ROOT / "nar_paper_audit/out/TableS11_partner_sequence_audit.tsv"
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
TRAIN_IDX = ROOT / "train/data/rnp_real/train/rnp_indices.csv"
MAX_PAIR_PRODUCT = 350_000
MAX_CAND_PAIRS = 40


def aa_seq(residues) -> str:
    out = []
    for r in residues:
        key = r.resname.strip().capitalize()
        out.append(protein_letters_3to1.get(key, "X"))
    return "".join(out)


def rna_seq(residues) -> str:
    out = []
    for r in residues:
        name = r.resname.strip().upper().replace("R", "")
        if name in ("A", "U", "G", "C", "I", "T"):
            out.append("U" if name == "T" else name)
        elif name:
            out.append(name[-1] if name[-1] in "AUGC" else "N")
        else:
            out.append("N")
    return "".join(out)


def pick_best_pair(model, prefer_pc=None, prefer_rc=None):
    """Return (p_chain, r_chain, n_prot, n_rna, pairs, n_nat) maximizing n_nat under product cap."""
    # try preferred first
    if prefer_pc and prefer_rc:
        pc = find_chain(model, prefer_pc)
        rc = find_chain(model, prefer_rc)
        if pc is not None and rc is not None and classify_chain(pc) == "protein" and classify_chain(rc) == "rna":
            pr = polymer_residues(pc, "protein")
            rr = polymer_residues(rc, "rna")
            prod = len(pr) * len(rr)
            if 0 < prod <= MAX_PAIR_PRODUCT:
                _, pairs, n_nat = _interface_atoms(pr, rr, cutoff=5.0)
                if n_nat > 0:
                    return pc, rc, pr, rr, pairs, n_nat

    prot_chains = []
    rna_chains = []
    for ch in model:
        ct = classify_chain(ch)
        if ct == "protein":
            pr = polymer_residues(ch, "protein")
            if len(pr) >= 15:
                prot_chains.append((ch, pr))
        elif ct == "rna":
            rr = polymer_residues(ch, "rna")
            if len(rr) >= 5:
                rna_chains.append((ch, rr))

    # prefer smaller RNAs first
    rna_chains.sort(key=lambda x: len(x[1]))
    prot_chains.sort(key=lambda x: len(x[1]))

    best = None
    tried = 0
    for pc, pr in prot_chains:
        for rc, rr in rna_chains:
            prod = len(pr) * len(rr)
            if prod == 0 or prod > MAX_PAIR_PRODUCT:
                continue
            tried += 1
            if tried > MAX_CAND_PAIRS:
                break
            _, pairs, n_nat = _interface_atoms(pr, rr, cutoff=5.0)
            if best is None or n_nat > best[-1]:
                best = (pc, rc, pr, rr, pairs, n_nat)
            if n_nat >= 20:
                return best
        if tried > MAX_CAND_PAIRS:
            break
    return best


def write_payload(pdb, split, pc, rc, pr, rr, pairs, n_nat, cif, extra=None):
    payload = {
        "pdb": pdb.lower(),
        "protein_chain": str(pc.id),
        "rna_chain": str(rc.id),
        "split": split,
        "n_prot": len(pr),
        "n_rna": len(rr),
        "n_contacts": int(n_nat),
        "pairs": [[int(i), int(j)] for i, j in pairs],
        "protein_seq": aa_seq(pr),
        "rna_seq": rna_seq(rr),
        "source_cif": str(cif),
    }
    if extra:
        payload.update(extra)
    out = OUT / f"{pdb.lower()}_{pc.id}_{rc.id}.json"
    out.write_text(json.dumps(payload))
    # also write under requested names if val case style
    return out, payload


def export_cif(pdb, cif, split, prefer_pc=None, prefer_rc=None, force=False):
    # skip if a non-empty contact file already exists for this pdb unless force
    existing = list(OUT.glob(f"{pdb.lower()}_*.json"))
    if existing and not force:
        for p in existing:
            d = json.loads(p.read_text())
            if int(d.get("n_contacts") or 0) > 0 and d.get("protein_seq"):
                return f"keep:{p.name}:n={d['n_contacts']}"
    nat = _parse_cif(str(cif), pdb.lower())
    model = list(nat)[0]
    best = pick_best_pair(model, prefer_pc, prefer_rc)
    if best is None:
        return "no_viable_pair"
    pc, rc, pr, rr, pairs, n_nat = best
    out, payload = write_payload(pdb, split, pc, rc, pr, rr, pairs, n_nat, cif)
    return f"wrote:{out.name}:n={n_nat}:prod={len(pr)*len(rr)}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    by_pdb = {}
    with TRAIN_IDX.open() as f:
        for r in csv.DictReader(f):
            by_pdb.setdefault(r["pdb_id"].lower(), []).append(r)

    # 1) refresh val50 (force rewrite with sequences; skip huge by token)
    print("=== VAL ===", flush=True)
    with CASES.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            tokens = float(r.get("num_tokens") or 0)
            prod = float(r.get("protein_len") or 0) * float(r.get("rna_len") or 0)
            if tokens > 8000 or prod > MAX_PAIR_PRODUCT:
                print(f"SKIP val {r['name']} tokens={tokens} prod={prod}", flush=True)
                continue
            try:
                msg = export_cif(
                    r["pdb_id"],
                    r["native_cif_gz"],
                    "val",
                    r["protein_chain_id"],
                    r["rna_chain_id"],
                    force=True,
                )
            except Exception as e:
                msg = f"error:{e}"
                traceback.print_exc()
            print(f"{r['name']} -> {msg}", flush=True)

    # 2) nearest train >=0.4
    print("=== TRAIN NN ===", flush=True)
    nn = sorted(
        {
            a["nearest_train_protein"].lower()
            for a in csv.DictReader(AUDIT.open(), delimiter="\t")
            if a.get("nearest_train_protein") and float(a.get("protein_identity") or 0) >= 0.4
        }
    )
    for pdb in nn:
        cif = CIF_POOL / f"{pdb}.cif.gz"
        if not cif.exists():
            print(f"{pdb} missing_cif", flush=True)
            continue
        rows = by_pdb.get(pdb, [])
        prefer_pc = prefer_rc = None
        if rows:
            r0 = rows[0]
            c1, t1, c2, t2 = r0["chain_1_id"], r0["mol_1_type"], r0["chain_2_id"], r0["mol_2_type"]
            if t1 == "prot" and t2 == "nuc":
                prefer_pc, prefer_rc = c1, c2
            elif t2 == "prot" and t1 == "nuc":
                prefer_pc, prefer_rc = c2, c1
        try:
            msg = export_cif(pdb, cif, "train_nn", prefer_pc, prefer_rc, force=True)
        except Exception as e:
            msg = f"error:{e}"
            traceback.print_exc()
        print(f"{pdb} -> {msg}", flush=True)
    print("DONE_EXPORT", flush=True)


if __name__ == "__main__":
    main()
