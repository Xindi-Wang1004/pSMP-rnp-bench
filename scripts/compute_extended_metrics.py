#!/usr/bin/env python3
"""Re-evaluate existing interface JSONs to add contact precision and F1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FUSAI_ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
sys.path.insert(0, str(FUSAI_ROOT / "train"))

from psmp.eval_interface import eval_case, find_best_pred_cif  # noqa: E402


def resolve_pred_cif(row: dict, data: dict) -> str | None:
    pred_cif = row.get("pred_cif")
    if pred_cif and Path(pred_cif).is_file():
        return pred_cif
    pred_dir = data.get("pred_dir")
    if pred_dir:
        found = find_best_pred_cif(Path(pred_dir), row["name"])
        if found is not None:
            return str(found)
    return pred_cif


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def eval_case_extended(
    native_path: str,
    pred_path: str,
    pdb_id: str,
    protein_chain_id: str,
    rna_chain_id: str,
) -> dict:
    base = eval_case(native_path, pred_path, pdb_id, protein_chain_id, rna_chain_id)
    if "error" in base:
        return base

    # Re-parse for predicted contact counts (eval_case internals not exported)
    from psmp.eval_interface import (  # noqa: E402
        _best_chain_pair,
        _interface_atoms,
        _pair_distance,
        _parse_cif,
        find_chain,
        polymer_residues,
    )

    nat = _parse_cif(native_path, pdb_id)
    prd = _parse_cif(pred_path, pdb_id)
    model_n, model_p = nat[0], prd[0]
    np_chain, nr_chain, _ = _best_chain_pair(model_n, protein_chain_id, rna_chain_id)
    pp_chain = find_chain(model_p, "A")
    rp_chain = find_chain(model_p, "B")
    if not all([np_chain, nr_chain, pp_chain, rp_chain]):
        return {**base, "error": "chain_not_found_extended"}

    n_prot = polymer_residues(np_chain, "protein")
    n_rna = polymer_residues(nr_chain, "rna")
    p_prot = polymer_residues(pp_chain, "protein")
    p_rna = polymer_residues(rp_chain, "rna")
    n = min(len(n_prot), len(p_prot))
    m = min(len(n_rna), len(p_rna))
    n_prot, p_prot = n_prot[:n], p_prot[:n]
    n_rna, p_rna = n_rna[:m], p_rna[:m]

    _, native_pairs, _ = _interface_atoms(n_prot, n_rna)
    tp = sum(1 for i, j in native_pairs if _pair_distance(p_prot, p_rna, i, j) <= 5.0)
    n_pred = sum(
        1
        for i in range(len(p_prot))
        for j in range(len(p_rna))
        if _pair_distance(p_prot, p_rna, i, j) <= 5.0
    )
    fp = max(n_pred - tp, 0)
    recall = tp / max(len(native_pairs), 1)
    precision = tp / max(n_pred, 1)
    f1 = _f1(precision, recall)

    return {
        **base,
        "n_recovered_contacts_5A": tp,
        "n_predicted_contacts_5A": n_pred,
        "n_false_positive_contacts_5A": fp,
        "contact_precision_5A": float(precision),
        "contact_f1_5A": float(f1),
    }


def summarize(rows: list[dict]) -> dict:
    ok = [r for r in rows if "contact_precision_5A" in r]
    if not ok:
        return {"n_ok": 0}
    return {
        "n_ok": len(ok),
        "mean_interface_lddt": float(np.mean([r["interface_lddt"] for r in ok])),
        "mean_contact_recall_5A": float(np.mean([r["contact_recall_5A"] for r in ok])),
        "mean_contact_precision_5A": float(np.mean([r["contact_precision_5A"] for r in ok])),
        "mean_contact_f1_5A": float(np.mean([r["contact_f1_5A"] for r in ok])),
    }


def process(interface_json: Path, cases_tsv: Path, out_json: Path) -> dict:
    data = json.loads(interface_json.read_text())
    cases = pd.read_csv(cases_tsv, sep="\t").set_index("name")
    rows: list[dict] = []
    for row in data["results"]:
        name = row["name"]
        pred_cif = resolve_pred_cif(row, data)
        if not pred_cif or name not in cases.index:
            rows.append({**row, **({"error": "pred_cif_not_found"} if "interface_lddt" not in row else {})})
            continue
        c = cases.loc[name]
        try:
            ext = eval_case_extended(
                str(c.native_cif_gz),
                str(pred_cif),
                str(c.pdb_id).lower(),
                str(c.protein_chain_id),
                str(c.rna_chain_id),
            )
            rows.append({"name": name, "pred_cif": pred_cif, **ext})
        except Exception as e:
            rows.append({"name": name, "error": str(e)})

    out = {**{k: v for k, v in data.items() if k != "results"}, **summarize(rows), "results": rows}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface-json", type=Path, required=True)
    ap.add_argument("--cases-tsv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()
    out = process(args.interface_json, args.cases_tsv, args.out_json)
    print(
        f"written {args.out_json} n_ok={out['n_ok']} "
        f"recall={out.get('mean_contact_recall_5A', float('nan')):.4f} "
        f"precision={out.get('mean_contact_precision_5A', float('nan')):.4f} "
        f"f1={out.get('mean_contact_f1_5A', float('nan')):.4f}"
    )


if __name__ == "__main__":
    main()
