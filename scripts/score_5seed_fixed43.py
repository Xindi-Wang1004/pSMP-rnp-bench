#!/usr/bin/env python3
"""Score Fixed43 under five-sample best-of-ipTM (same artifacts/protocol as Table 2 @100%)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "nar_paper" / "scripts"))

from compute_extended_metrics import eval_case_extended  # noqa: E402

EVAL = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"
CASES43 = ROOT / "nar_paper/data/cases_fixed43.tsv"
OUT_DIR = ROOT / "nar_paper/tables"
TAGS = ["pct100_base", "pct100_psmp"]


def load_cases() -> list[dict]:
    with CASES43.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def find_samples(pred_root: Path, name: str) -> list[tuple[float, Path]]:
    out: list[tuple[float, Path]] = []
    patterns = [
        pred_root / name / name / "seed_101" / "predictions",
        pred_root / name / "seed_101" / "predictions",
        pred_root / name,
    ]
    pred_dir = next((p for p in patterns if p.is_dir()), None)
    if pred_dir is None:
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


def score_tag(tag: str, cases: list[dict]) -> dict:
    pred_root = EVAL / f"pred_{tag}"
    results = []
    n_ok = 0
    for c in cases:
        name = c["name"]
        samples = find_samples(pred_root, name)
        if not samples:
            results.append({"name": name, "error": "missing_prediction_no_cif", "status": "missing_prediction"})
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
            results.append(
                {
                    "name": name,
                    "error": str(e),
                    "pred_cif": str(best_cif),
                    "status": "error",
                    "n_samples_found": len(samples),
                }
            )
            continue
        status = "evaluable" if m.get("contact_f1_5A") is not None else "no_interface"
        if m.get("contact_f1_5A") is not None:
            n_ok += 1
        results.append(
            {
                **m,
                "name": name,
                "pred_cif": str(best_cif),
                "best_iptm": best_iptm,
                "n_samples_found": len(samples),
                "status": status,
            }
        )
    f1s = [r["contact_f1_5A"] for r in results if r.get("contact_f1_5A") is not None]
    recalls = [r["contact_recall_5A"] for r in results if r.get("contact_recall_5A") is not None]
    precs = [r["contact_precision_5A"] for r in results if r.get("contact_precision_5A") is not None]
    ilddts = [r["interface_lddt"] for r in results if r.get("interface_lddt") is not None]
    payload = {
        "tag": tag,
        "subset": "Fixed43",
        "protocol": "five-sample_best_of_iptm_seed101",
        "pred_dir": str(pred_root),
        "cases_tsv": str(CASES43),
        "n_cases": len(cases),
        "n_ok": n_ok,
        "mean_contact_f1_5A": (sum(f1s) / len(f1s)) if f1s else None,
        "mean_contact_recall_5A": (sum(recalls) / len(recalls)) if recalls else None,
        "mean_contact_precision_5A": (sum(precs) / len(precs)) if precs else None,
        "mean_interface_lddt": (sum(ilddts) / len(ilddts)) if ilddts else None,
        "results": results,
    }
    out_json = EVAL / f"interface_5seed_{tag}_fixed43_ext.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_json} n_ok={n_ok}/{len(cases)}")
    return payload


def fa_and_paired(base: dict, psmp: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bmap = {r["name"]: r for r in base["results"]}
    pmap = {r["name"]: r for r in psmp["results"]}
    names = sorted(set(bmap) | set(pmap))

    # failure-aware: missing/non-evaluable → 0
    rows_fa = []
    for n in names:
        bf = bmap.get(n, {}).get("contact_f1_5A")
        pf = pmap.get(n, {}).get("contact_f1_5A")
        br = bmap.get(n, {}).get("contact_recall_5A")
        pr = pmap.get(n, {}).get("contact_recall_5A")
        bi = bmap.get(n, {}).get("interface_lddt")
        pi = pmap.get(n, {}).get("interface_lddt")
        rows_fa.append(
            {
                "name": n,
                "status_base": bmap.get(n, {}).get("status") or bmap.get(n, {}).get("error") or "missing",
                "status_psmp": pmap.get(n, {}).get("status") or pmap.get(n, {}).get("error") or "missing",
                "F1_base_fa": 0.0 if bf is None else float(bf),
                "F1_psmp_fa": 0.0 if pf is None else float(pf),
                "recall_base_fa": 0.0 if br is None else float(br),
                "recall_psmp_fa": 0.0 if pr is None else float(pr),
                "ilddt_base_fa": 0.0 if bi is None else float(bi),
                "ilddt_psmp_fa": 0.0 if pi is None else float(pi),
                "n_samples_base": bmap.get(n, {}).get("n_samples_found", 0),
                "n_samples_psmp": pmap.get(n, {}).get("n_samples_found", 0),
            }
        )

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    summary = {
        "subset": "Fixed43",
        "protocol": "five-sample_best_of_iptm_seed101_failure_aware",
        "n_cases": len(names),
        "n_ok_base": base["n_ok"],
        "n_ok_psmp": psmp["n_ok"],
        "F1_fa_mean_base": mean([r["F1_base_fa"] for r in rows_fa]),
        "F1_fa_mean_psmp": mean([r["F1_psmp_fa"] for r in rows_fa]),
        "Delta_F1_fa": None,
        "recall_fa_mean_base": mean([r["recall_base_fa"] for r in rows_fa]),
        "recall_fa_mean_psmp": mean([r["recall_psmp_fa"] for r in rows_fa]),
        "ilddt_fa_mean_base": mean([r["ilddt_base_fa"] for r in rows_fa]),
        "ilddt_fa_mean_psmp": mean([r["ilddt_psmp_fa"] for r in rows_fa]),
        # variable-n_ok means (legacy-style complementary)
        "F1_n_ok_mean_base": base["mean_contact_f1_5A"],
        "F1_n_ok_mean_psmp": psmp["mean_contact_f1_5A"],
        "recall_n_ok_mean_base": base["mean_contact_recall_5A"],
        "recall_n_ok_mean_psmp": psmp["mean_contact_recall_5A"],
        "ilddt_n_ok_mean_base": base["mean_interface_lddt"],
        "ilddt_n_ok_mean_psmp": psmp["mean_interface_lddt"],
    }
    summary["Delta_F1_fa"] = summary["F1_fa_mean_psmp"] - summary["F1_fa_mean_base"]

    # paired intersection
    pair_rows = []
    for n in names:
        bf = bmap.get(n, {}).get("contact_f1_5A")
        pf = pmap.get(n, {}).get("contact_f1_5A")
        if bf is None or pf is None:
            continue
        pair_rows.append(
            {
                "name": n,
                "F1_base": float(bf),
                "F1_psmp": float(pf),
                "delta_F1": float(pf) - float(bf),
                "recall_base": float(bmap[n].get("contact_recall_5A") or 0.0),
                "recall_psmp": float(pmap[n].get("contact_recall_5A") or 0.0),
                "ilddt_base": float(bmap[n].get("interface_lddt") or 0.0),
                "ilddt_psmp": float(pmap[n].get("interface_lddt") or 0.0),
            }
        )

    # Wilcoxon one-sided (psmp > base) if scipy available
    p_one = p_two = None
    if pair_rows:
        try:
            from scipy.stats import wilcoxon

            diffs = [r["delta_F1"] for r in pair_rows]
            # wilcoxon needs non-zero diffs ideally
            if any(abs(d) > 0 for d in diffs):
                p_two = float(wilcoxon(diffs, alternative="two-sided").pvalue)
                p_one = float(wilcoxon(diffs, alternative="greater").pvalue)
        except Exception as e:
            print("wilcoxon skipped:", e)

    signs = {
        "improve": sum(1 for r in pair_rows if r["delta_F1"] > 0),
        "worse": sum(1 for r in pair_rows if r["delta_F1"] < 0),
        "tie": sum(1 for r in pair_rows if r["delta_F1"] == 0),
    }
    paired_summary = {
        "subset": "Fixed43",
        "protocol": "five-sample_best_of_iptm_seed101_paired_intersection",
        "n_pair": len(pair_rows),
        "mean_F1_base": mean([r["F1_base"] for r in pair_rows]),
        "mean_F1_psmp": mean([r["F1_psmp"] for r in pair_rows]),
        "delta_F1": mean([r["delta_F1"] for r in pair_rows]),
        "mean_recall_base": mean([r["recall_base"] for r in pair_rows]),
        "mean_recall_psmp": mean([r["recall_psmp"] for r in pair_rows]),
        "mean_ilddt_base": mean([r["ilddt_base"] for r in pair_rows]),
        "mean_ilddt_psmp": mean([r["ilddt_psmp"] for r in pair_rows]),
        "wilcoxon_p_f1_onesided": p_one,
        "wilcoxon_p_f1_twosided": p_two,
        "signs": signs,
    }

    # write CSVs
    fa_csv = OUT_DIR / "Table4b_fixed43_fivesample_failure_aware.csv"
    with fa_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_fa[0].keys()) if rows_fa else ["name"])
        w.writeheader()
        w.writerows(rows_fa)
    pair_csv = OUT_DIR / "Table4c_fixed43_fivesample_paired.csv"
    with pair_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()) if pair_rows else ["name"])
        w.writeheader()
        w.writerows(pair_rows)

    summary_path = OUT_DIR / "Table4_fixed43_fivesample_summary.json"
    payload = {"failure_aware": summary, "paired": paired_summary, "legacy_single_seed_note": (
        "Legacy Table 4 single-seed Fixed43 deposit remains historical; "
        "these rows are the Table-2-aligned five-sample read on the same 43 names."
    )}
    summary_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("wrote", fa_csv)
    print("wrote", pair_csv)
    print("wrote", summary_path)


def main():
    cases = load_cases()
    assert len(cases) == 43, len(cases)
    payloads = {tag: score_tag(tag, cases) for tag in TAGS}
    fa_and_paired(payloads["pct100_base"], payloads["pct100_psmp"])


if __name__ == "__main__":
    main()
