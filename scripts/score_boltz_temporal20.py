#!/usr/bin/env python3
"""Build interface JSON + extended metrics for Boltz-2 Temporal20 preds."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
RUN = ROOT / "train/runs/boltz2_baseline_eval/boltz2_temporal20"
PRED = RUN / "pred_boltz"
CASES = ROOT / "train/runs/temporal20/cases_temporal20.tsv"
OUT_IFACE = RUN / "interface_boltz2_temporal20.json"
OUT_EXT = RUN / "interface_boltz2_temporal20_ext.json"

sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(PAPER / "scripts"))

from psmp.eval_interface import eval_case  # noqa: E402
from compute_extended_metrics import eval_case_extended, summarize  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> None:
    if not CASES.is_file():
        raise SystemExit(f"missing {CASES}")
    cases = pd.read_csv(CASES, sep="\t")
    rows = []
    for _, c in cases.iterrows():
        name = c["name"]
        pred = PRED / name / f"{name}_sample_0.cif"
        if not pred.is_file():
            rows.append({"name": name, "error": "pred_missing"})
            continue
        try:
            m = eval_case(
                str(c["native_cif_gz"]),
                str(pred),
                str(c["pdb_id"]).lower(),
                str(c["protein_chain_id"]),
                str(c["rna_chain_id"]),
            )
            rows.append({"name": name, "pred_cif": str(pred), **m})
        except Exception as e:
            rows.append({"name": name, "error": str(e), "pred_cif": str(pred)})

    ok = [r for r in rows if "contact_recall_5A" in r]
    payload = {
        "tag": "boltz2_temporal20",
        "protocol": "boltz2_zero_shot_single_sample",
        "pred_dir": str(PRED),
        "cases_tsv": str(CASES),
        "n_cases": len(rows),
        "n_ok": len(ok),
        "mean_interface_lddt": (sum(r["interface_lddt"] for r in ok) / len(ok)) if ok else None,
        "mean_contact_recall_5A": (sum(r["contact_recall_5A"] for r in ok) / len(ok)) if ok else None,
        "results": rows,
    }
    OUT_IFACE.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT_IFACE} n_ok={len(ok)}/{len(rows)}")

    ext_rows = []
    for row in rows:
        name = row["name"]
        if row.get("error") and "contact_recall_5A" not in row:
            ext_rows.append({**row, "contact_f1_5A": None})
            continue
        c = cases[cases["name"] == name].iloc[0]
        try:
            ext = eval_case_extended(
                str(c["native_cif_gz"]),
                str(row["pred_cif"]),
                str(c["pdb_id"]).lower(),
                str(c["protein_chain_id"]),
                str(c["rna_chain_id"]),
            )
            ext_rows.append({"name": name, "pred_cif": row["pred_cif"], **ext})
        except Exception as e:
            ext_rows.append({"name": name, "error": str(e)})

    fa_f1 = []
    for r in ext_rows:
        if r.get("contact_f1_5A") is not None:
            fa_f1.append(float(r["contact_f1_5A"]))
        else:
            fa_f1.append(0.0)
    out_ext = {
        **payload,
        **summarize([r for r in ext_rows if r.get("contact_f1_5A") is not None]),
        "FA_mean_contact_f1_5A": sum(fa_f1) / len(fa_f1) if fa_f1 else None,
        "results": ext_rows,
    }
    OUT_EXT.write_text(json.dumps(out_ext, indent=2) + "\n")
    print(
        f"wrote {OUT_EXT} FA_F1={out_ext.get('FA_mean_contact_f1_5A'):.4f} "
        f"n_ok={out_ext.get('n_ok')}"
    )


if __name__ == "__main__":
    main()
