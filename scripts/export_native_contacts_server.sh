#!/usr/bin/env bash
# Export contacts for val50 + MMseqs nearest-train PDBs from pdb_rna_protein_contact_cif/.
set -euo pipefail
ROOT="${ROOT:-/home/wangxindi/RNA_Protein/fusai}"
PY="${PYTHON:-$HOME/miniconda3/envs/protenix/bin/python3}"
OUT="${OUT:-$ROOT/nar_paper/data/contacts}"
CIF_POOL="${CIF_POOL:-/home/wangxindi/RNA_Protein/rnp_data/pdb_rna_protein_contact_cif}"
mkdir -p "$OUT"

export PYTHONPATH="$ROOT/train:${PYTHONPATH:-}"
"$PY" -u - <<'PY'
import csv, json, sys, time, traceback
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
sys.path.insert(0, str(ROOT / "train"))
from psmp.eval_interface import _parse_cif, _best_chain_pair, _interface_atoms
from psmp.structure_io import polymer_residues

OUT = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper/data/contacts")
CIF_POOL = Path("/home/wangxindi/RNA_Protein/rnp_data/pdb_rna_protein_contact_cif")
SKIP_LOG = Path("/home/wangxindi/RNA_Protein/fusai/nar_paper/data/contacts_skipped.tsv")
OUT.mkdir(parents=True, exist_ok=True)
MAX_TOKENS = 4000
MAX_PAIR_PRODUCT = 200_000

jobs = []
# val50 natives
with (ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv").open() as f:
    for r in csv.DictReader(f, delimiter="\t"):
        jobs.append(
            dict(
                pdb=r["pdb_id"],
                pc=r["protein_chain_id"],
                rc=r["rna_chain_id"],
                path=r["native_cif_gz"],
                split="val",
                num_tokens=int(float(r.get("num_tokens") or 0)),
                protein_len=int(float(r.get("protein_len") or 0)),
                rna_len=int(float(r.get("rna_len") or 0)),
            )
        )

# nearest train PDBs from MMseqs audit + train indices for chains
audit = ROOT / "nar_paper_audit/out/TableS11_partner_sequence_audit.tsv"
nn_pdbs = set()
if audit.exists():
    with audit.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("nearest_train_protein"):
                nn_pdbs.add(r["nearest_train_protein"].lower())
            if r.get("nearest_train_rna"):
                nn_pdbs.add(r["nearest_train_rna"].lower())

by_pdb = {}
with (ROOT / "train/data/rnp_real/train/rnp_indices.csv").open() as f:
    for r in csv.DictReader(f):
        by_pdb.setdefault(r["pdb_id"].lower(), []).append(r)

for pdb in sorted(nn_pdbs):
    cif = CIF_POOL / f"{pdb}.cif.gz"
    if not cif.exists():
        continue
    rows = by_pdb.get(pdb, [])
    if not rows:
        # unknown chains: let _best_chain_pair auto-pick
        jobs.append(dict(pdb=pdb, pc="A", rc="B", path=str(cif), split="train_nn", num_tokens=0, protein_len=0, rna_len=0))
        continue
    r0 = rows[0]
    c1, t1 = r0["chain_1_id"], r0["mol_1_type"]
    c2, t2 = r0["chain_2_id"], r0["mol_2_type"]
    if t1 == "prot" and t2 == "nuc":
        pc, rc = c1, c2
    elif t2 == "prot" and t1 == "nuc":
        pc, rc = c2, c1
    else:
        pc, rc = c1, c2
    ntok = int(float(r0.get("num_tokens") or 0))
    jobs.append(dict(pdb=pdb, pc=pc, rc=rc, path=str(cif), split="train_nn", num_tokens=ntok, protein_len=0, rna_len=0))

print(f"jobs={len(jobs)} nn_pdbs={len(nn_pdbs)}", flush=True)
skipped = []
ok = fail = 0
t0 = time.time()
for j in jobs:
    pdb, pc, rc, path = j["pdb"], j["pc"], j["rc"], j["path"]
    out = OUT / f"{pdb.lower()}_{pc}_{rc}.json"
    if out.exists():
        ok += 1
        continue
    prod = (j["protein_len"] or 0) * (j["rna_len"] or 0)
    if j["num_tokens"] > MAX_TOKENS or prod > MAX_PAIR_PRODUCT:
        reason = f"too_large tokens={j['num_tokens']} prod={prod}"
        print(f"SKIP {pdb} {pc}/{rc}: {reason}", flush=True)
        skipped.append({**j, "reason": reason})
        continue
    try:
        nat = _parse_cif(path, pdb.lower())
        model = list(nat)[0]
        p_chain, r_chain, mode = _best_chain_pair(model, pc, rc)
        if p_chain is None:
            skipped.append({**j, "reason": "no_chain_pair"})
            continue
        n_prot = polymer_residues(p_chain, "protein")
        n_rna = polymer_residues(r_chain, "rna")
        if len(n_prot) * len(n_rna) > MAX_PAIR_PRODUCT:
            reason = f"too_large_after_parse prod={len(n_prot)*len(n_rna)}"
            print(f"SKIP {pdb} {pc}/{rc}: {reason}", flush=True)
            skipped.append({**j, "reason": reason})
            continue
        _, pairs, n_nat = _interface_atoms(n_prot, n_rna, cutoff=5.0)
        payload = {
            "pdb": pdb.lower(),
            "protein_chain": str(getattr(p_chain, "id", pc)),
            "rna_chain": str(getattr(r_chain, "id", rc)),
            "split": j["split"],
            "n_prot": len(n_prot),
            "n_rna": len(n_rna),
            "n_contacts": int(n_nat),
            "pairs": [[int(i), int(j_)] for i, j_ in pairs],
            "source_cif": path,
            "chain_mode": mode,
        }
        # write under both requested and resolved chain ids
        out.write_text(json.dumps(payload))
        alt = OUT / f"{payload['pdb']}_{payload['protein_chain']}_{payload['rna_chain']}.json"
        if alt != out:
            alt.write_text(json.dumps(payload))
        ok += 1
        if ok % 10 == 0:
            print(f"wrote {ok} elapsed={time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        fail += 1
        print(f"FAIL {pdb} {pc}/{rc}: {e}", flush=True)
        skipped.append({**j, "reason": f"error:{e}"})

with SKIP_LOG.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["pdb", "pc", "rc", "split", "num_tokens", "protein_len", "rna_len", "path", "reason"])
    w.writeheader()
    for s in skipped:
        w.writerow({k: s.get(k) for k in w.fieldnames})
print(f"done ok={ok} fail={fail} skipped={len(skipped)} out={OUT}", flush=True)
PY
