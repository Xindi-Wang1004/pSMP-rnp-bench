#!/usr/bin/env python3
"""Export 5Å contact JSONs for frozen lowdata/{pct}/ PDBs into nar_paper/data/contacts."""
from __future__ import annotations

import json
import sys
import time
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

PCT = "pct10"
OUT = ROOT / "nar_paper/data/contacts"
LOW = ROOT / "train/data/rnp_real/lowdata" / PCT
CIF_POOL = Path("/home/wangxindi/RNA_Protein/rnp_data/pdb_rna_protein_contact_cif")
MMCIF = LOW / "mmcif"
MAX_PAIR_PRODUCT = 350_000


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


def pick_best_pair(model):
    prot_chains, rna_chains = [], []
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
            if tried > 40:
                break
            _, pairs, n_nat = _interface_atoms(pr, rr, cutoff=5.0)
            if best is None or n_nat > best[-1]:
                best = (pc, rc, pr, rr, pairs, n_nat)
            if n_nat >= 20:
                return best
        if tried > 40:
            break
    return best


def find_cif(pdb: str) -> Path | None:
    for base in (MMCIF, CIF_POOL):
        if not base.exists():
            continue
        for name in (f"{pdb}.cif.gz", f"{pdb}.cif", f"{pdb.upper()}.cif.gz", f"{pdb.lower()}.cif.gz"):
            p = base / name
            if p.exists():
                return p
        hits = list(base.glob(f"{pdb.lower()}*")) + list(base.glob(f"{pdb.upper()}*"))
        for h in hits:
            if h.suffix in {".cif", ".gz"} or h.name.endswith(".cif.gz"):
                return h
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pdbs = [x.strip().lower() for x in (LOW / "pdb_list.txt").read_text().split() if x.strip()]
    print("pct", PCT, "n_pdb", len(pdbs))
    ok = skip = 0
    for pdb in pdbs:
        # skip if any contact json exists
        if list(OUT.glob(f"{pdb}_*.json")):
            print("have", pdb)
            ok += 1
            continue
        cif = find_cif(pdb)
        if cif is None:
            print("NOCIF", pdb)
            skip += 1
            continue
        t0 = time.time()
        try:
            nat = _parse_cif(str(cif), pdb)
            model = list(nat)[0]
            best = pick_best_pair(model)
            if best is None:
                print("NOPAIR", pdb)
                skip += 1
                continue
            pc, rc, pr, rr, pairs, n_nat = best
            payload = {
                "pdb": pdb,
                "protein_chain": pc.id,
                "rna_chain": rc.id,
                "split": PCT,
                "n_prot": len(pr),
                "n_rna": len(rr),
                "n_contacts": int(n_nat),
                "pairs": [[int(a), int(b)] for a, b in pairs],
                "protein_seq": aa_seq(pr),
                "rna_seq": rna_seq(rr),
                "source_cif": str(cif),
                "contact_time_s": time.time() - t0,
            }
            out = OUT / f"{pdb}_{pc.id}_{rc.id}.json"
            out.write_text(json.dumps(payload))
            print("OK", out.name, "n", n_nat, f"{time.time()-t0:.1f}s")
            ok += 1
        except Exception as e:
            print("FAIL", pdb, e)
            skip += 1
    print("done ok", ok, "skip", skip)


if __name__ == "__main__":
    main()
