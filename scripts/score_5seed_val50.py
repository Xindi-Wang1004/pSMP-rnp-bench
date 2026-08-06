#!/usr/bin/env python3
"""Score val50 five-sample preds: best-of-5 by predicted iptm → extended interface JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
sys.path.insert(0, str(ROOT / "train"))

from psmp.eval_interface import eval_case  # noqa: E402

# reuse extended metric logic
sys.path.insert(0, str(ROOT / "nar_paper" / "scripts"))
from compute_extended_metrics import eval_case_extended  # noqa: E402

EVAL = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
TAGS = [
    "pct10_base",
    "pct10_psmp",
    "pct25_base",
    "pct25_psmp",
    "pct50_base",
    "pct50_psmp",
    "pct100_base",
    "pct100_psmp",
]


def load_cases() -> list[dict]:
    import csv

    with CASES.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def find_samples(pred_root: Path, name: str) -> list[tuple[float, Path]]:
    """Return list of (iptm, cif_path)."""
    out: list[tuple[float, Path]] = []
    # common layouts
    patterns = [
        pred_root / name / name / "seed_101" / "predictions",
        pred_root / name / "seed_101" / "predictions",
        pred_root / name,
    ]
    pred_dir = next((p for p in patterns if p.is_dir()), None)
    if pred_dir is None:
        # recursive search for sample_0
        hits = list(pred_root.glob(f"**/{name}_sample_0.cif"))
        if not hits:
            return out
        pred_dir = hits[0].parent
    for i in range(5):
        cif = pred_dir / f"{name}_sample_{i}.cif"
        conf = pred_dir / f"{name}_summary_confidence_sample_{i}.json"
        if not cif.is_file():
            continue
        iptm = float("-inf")
        if conf.is_file():
            try:
                d = json.loads(conf.read_text())
                iptm = float(d.get("iptm", d.get("ranking_score", 0.0)) or 0.0)
            except Exception:
                iptm = 0.0
        out.append((iptm, cif))
    return out


def score_tag(tag: str, cases: list[dict]) -> Path:
    pred_root = EVAL / f"pred_{tag}"
    out_json = EVAL / f"interface_5seed_{tag}_val50_ext.json"
    results = []
    n_ok = 0
    for c in cases:
        name = c["name"]
        samples = find_samples(pred_root, name)
        if not samples:
            results.append({"name": name, "error": "missing_prediction_no_cif"})
            continue
        samples.sort(key=lambda x: x[0], reverse=True)
        best_iptm, best_cif = samples[0]
        try:
            m = eval_case_extended(
                c["native_cif_gz"],
                str(best_cif),
                c["pdb_id"],
                c["protein_chain_id"],
                c["rna_chain_id"],
            )
        except Exception as e:
            results.append({"name": name, "error": str(e), "pred_cif": str(best_cif)})
            continue
        m = {
            **m,
            "name": name,
            "pred_cif": str(best_cif),
            "best_iptm": best_iptm,
            "n_samples_found": len(samples),
        }
        if m.get("contact_f1_5A") is not None:
            n_ok += 1
        results.append(m)
    f1s = [r["contact_f1_5A"] for r in results if r.get("contact_f1_5A") is not None]
    payload = {
        "tag": tag,
        "protocol": "five-sample_best_of_iptm_seed101",
        "pred_dir": str(pred_root),
        "cases_tsv": str(CASES),
        "n_cases": len(cases),
        "n_ok": n_ok,
        "mean_contact_f1_5A": (sum(f1s) / len(f1s)) if f1s else None,
        "results": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json} n_ok={n_ok}/{len(cases)}")
    return out_json


def main() -> None:
    cases = load_cases()
    for tag in TAGS:
        pred = EVAL / f"pred_{tag}"
        if not pred.exists():
            print(f"SKIP missing pred root {pred}")
            continue
        score_tag(tag, cases)


if __name__ == "__main__":
    main()
