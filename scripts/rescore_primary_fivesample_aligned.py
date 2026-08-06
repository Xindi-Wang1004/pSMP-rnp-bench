#!/usr/bin/env python3
"""Rescore primary five-sample tables with one locked protocol.

Protocol (locked for Table 2 / S6 / Fixed43):
- best-of-ipTM among usable CIFs from seed-101 five-sample trees
- Evaluable only if n_samples_found >= 1 AND (n_samples_found >= 5 OR num_tokens <= MAX_TOKENS)
  → excludes incomplete NSAMPLE=1 large-backfill orphans from n_ok
- Failure-aware denominator = all cases in the cohort (val50: 50; Fixed43: 43)
- missing_prediction / no_interface / non-evaluable → F1 = 0 (included in FA mean)
"""
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
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
F43 = ROOT / "nar_paper/data/cases_fixed43.tsv"
OUT = ROOT / "nar_paper/tables"
MAX_TOKENS = 4000.0
TAGS_PCT = [10, 25, 50, 100]


def load_cases(path: Path) -> list[dict]:
    with path.open() as f:
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


def protocol_allows(n_samples: int, num_tokens: float) -> bool:
    if n_samples < 1:
        return False
    if n_samples >= 5:
        return True
    return num_tokens <= MAX_TOKENS


def score_tag(tag: str, cases: list[dict]) -> dict:
    pred_root = EVAL / f"pred_{tag}"
    results = []
    n_ok = 0
    for c in cases:
        name = c["name"]
        tokens = float(c.get("num_tokens") or 0.0)
        samples = find_samples(pred_root, name)
        if not protocol_allows(len(samples), tokens):
            results.append(
                {
                    "name": name,
                    "error": "missing_prediction_or_incomplete_large",
                    "status": "missing_prediction",
                    "n_samples_found": len(samples),
                    "num_tokens": tokens,
                    "excluded_reason": (
                        "incomplete_large_backfill"
                        if len(samples) > 0 and tokens > MAX_TOKENS
                        else "no_usable_cif_under_fivesample_protocol"
                    ),
                }
            )
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
                    "status": "error",
                    "pred_cif": str(best_cif),
                    "n_samples_found": len(samples),
                    "num_tokens": tokens,
                }
            )
            continue
        if m.get("n_native_contacts_5A") == 0:
            status = "no_interface"
        elif m.get("contact_f1_5A") is not None:
            status = "evaluable"
            n_ok += 1
        else:
            status = "other"
        results.append(
            {
                **m,
                "name": name,
                "pred_cif": str(best_cif),
                "best_iptm": best_iptm,
                "n_samples_found": len(samples),
                "num_tokens": tokens,
                "status": status,
            }
        )
    f1s = [r["contact_f1_5A"] for r in results if r.get("contact_f1_5A") is not None]
    return {
        "tag": tag,
        "protocol": "five-sample_best_of_iptm_seed101_max_tokens_4000_strict",
        "max_tokens": MAX_TOKENS,
        "pred_dir": str(pred_root),
        "n_cases": len(cases),
        "n_ok": n_ok,
        "mean_contact_f1_5A": (sum(f1s) / len(f1s)) if f1s else None,
        "results": results,
    }


def fa_value(r: dict) -> float:
    """All cohort members enter FA denominator; non-evaluable → 0."""
    if r.get("status") == "evaluable" and r.get("contact_f1_5A") is not None:
        return float(r["contact_f1_5A"])
    return 0.0


def fa_metric(r: dict, key: str) -> float:
    if r.get("status") == "evaluable" and r.get(key) is not None:
        return float(r[key])
    return 0.0


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def wilcoxon_stats(deltas: list[float]):
    p1 = p2 = None
    try:
        from scipy.stats import wilcoxon

        if any(abs(d) > 0 for d in deltas):
            p2 = float(wilcoxon(deltas, alternative="two-sided").pvalue)
            p1 = float(wilcoxon(deltas, alternative="greater").pvalue)
    except Exception as e:
        print("wilcoxon:", e)
    return p1, p2


def build_fa_paired(base: dict, psmp: dict, cohort: str):
    bmap = {r["name"]: r for r in base["results"]}
    pmap = {r["name"]: r for r in psmp["results"]}
    names = sorted(set(bmap) | set(pmap))
    fa_rows = []
    for n in names:
        br, pr = bmap.get(n, {"status": "missing_prediction"}), pmap.get(n, {"status": "missing_prediction"})
        fa_rows.append(
            {
                "case": n,
                "base_status": br.get("status") or "missing_prediction",
                "psmp_status": pr.get("status") or "missing_prediction",
                "base_f1_fa": fa_value(br),
                "psmp_f1_fa": fa_value(pr),
                "base_recall_fa": fa_metric(br, "contact_recall_5A"),
                "psmp_recall_fa": fa_metric(pr, "contact_recall_5A"),
                "base_ilddt_fa": fa_metric(br, "interface_lddt"),
                "psmp_ilddt_fa": fa_metric(pr, "interface_lddt"),
                "base_n_samples": br.get("n_samples_found", 0),
                "psmp_n_samples": pr.get("n_samples_found", 0),
                "in_paired_intersection": int(
                    br.get("status") == "evaluable" and pr.get("status") == "evaluable"
                ),
            }
        )
    pair = [r for r in fa_rows if r["in_paired_intersection"]]
    deltas = [r["psmp_f1_fa"] - r["base_f1_fa"] for r in pair]
    p1, p2 = wilcoxon_stats(deltas)
    signs = {
        "improve": sum(1 for d in deltas if d > 0),
        "worse": sum(1 for d in deltas if d < 0),
        "tie": sum(1 for d in deltas if d == 0),
    }
    summary = {
        "cohort": cohort,
        "protocol": base["protocol"],
        "n_cases": len(names),
        "n_ok_base": sum(1 for r in fa_rows if r["base_status"] == "evaluable"),
        "n_ok_psmp": sum(1 for r in fa_rows if r["psmp_status"] == "evaluable"),
        "n_failure_aware_defined": len(names),
        "F1_fa_mean_base": mean([r["base_f1_fa"] for r in fa_rows]),
        "F1_fa_mean_psmp": mean([r["psmp_f1_fa"] for r in fa_rows]),
        "Delta_F1_fa": mean([r["psmp_f1_fa"] - r["base_f1_fa"] for r in fa_rows]),
        "recall_fa_mean_base": mean([r["base_recall_fa"] for r in fa_rows]),
        "recall_fa_mean_psmp": mean([r["psmp_recall_fa"] for r in fa_rows]),
        "ilddt_fa_mean_base": mean([r["base_ilddt_fa"] for r in fa_rows]),
        "ilddt_fa_mean_psmp": mean([r["psmp_ilddt_fa"] for r in fa_rows]),
        "n_pair": len(pair),
        "paired_F1_mean_base": mean([r["base_f1_fa"] for r in pair]),
        "paired_F1_mean_psmp": mean([r["psmp_f1_fa"] for r in pair]),
        "paired_Delta_F1": mean(deltas) if deltas else None,
        "wilcoxon_p_f1_onesided": p1,
        "wilcoxon_p_f1_twosided": p2,
        "signs": signs,
    }
    return fa_rows, summary


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_cases = load_cases(CASES)
    f43_names = {r["name"] for r in load_cases(F43)}
    f43_cases = [c for c in all_cases if c["name"] in f43_names]
    not_f43 = sorted(set(c["name"] for c in all_cases) - f43_names)
    (OUT / "Fixed43_membership.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Predefined before this revision: the 43 val50 cases that were "
                    "interface-evaluable under BOTH methods in the original single-seed "
                    "pct100 deposit (eval_work/interface_pct100_{base,psmp}.json, n_ok=43)."
                ),
                "n": 43,
                "names": sorted(f43_names),
                "excluded_from_val50_n7": not_f43,
                "exclusion_reason": (
                    "Not interface-evaluable under the original single-seed full-data "
                    "benchmark for at least one of base/pSMP (pre-specified Fixed43 rule; "
                    "not redefined by five-sample coverage)."
                ),
            },
            indent=2,
        )
    )
    print("Fixed43 excluded:", not_f43)

    s6b_rows = []
    paired_rows = []
    s6c_rows = []

    # Rescore only pct100 fully for alignment; optionally refresh all fractions' FA from existing JSON with new FA rule
    # For consistency of n_ok policy across fractions, rescore all tags (slow). Do pct100 first then others if needed.
    for pct in TAGS_PCT:
        for init in ("base", "psmp"):
            tag = f"pct{pct}_{init}"
            print("scoring", tag)
            payload = score_tag(tag, all_cases)
            outj = EVAL / f"interface_5seed_{tag}_val50_ext.json"
            # backup old
            if outj.exists():
                bak = outj.with_suffix(".json.before_strict_protocol")
                if not bak.exists():
                    bak.write_text(outj.read_text())
            outj.write_text(json.dumps(payload, indent=2))
            print(" wrote", outj, "n_ok", payload["n_ok"])

        base = json.loads((EVAL / f"interface_5seed_pct{pct}_base_val50_ext.json").read_text())
        psmp = json.loads((EVAL / f"interface_5seed_pct{pct}_psmp_val50_ext.json").read_text())
        fa_rows, summary = build_fa_paired(base, psmp, "val50")
        s6b_rows.append(
            {
                "Train_fraction": f"{pct}%",
                "protocol": summary["protocol"],
                "n_cases": summary["n_cases"],
                "n_ok_base": summary["n_ok_base"],
                "n_ok_psmp": summary["n_ok_psmp"],
                "n_failure_aware_defined": summary["n_failure_aware_defined"],
                "F1_failure_aware_mean_base": summary["F1_fa_mean_base"],
                "F1_failure_aware_mean_psmp": summary["F1_fa_mean_psmp"],
                "Delta_F1_failure_aware_mean": summary["Delta_F1_fa"],
                "FA_denominator_policy": "all_n_cases_missing_or_no_interface_as_0",
            }
        )
        paired_rows.append(
            {
                "Train_fraction": f"{pct}%",
                "n_base": summary["n_ok_base"],
                "n_psmp": summary["n_ok_psmp"],
                "n_pair": summary["n_pair"],
                "Base_F1_pair": summary["paired_F1_mean_base"],
                "pSMP_F1_pair": summary["paired_F1_mean_psmp"],
                "Delta_F1": summary["paired_Delta_F1"],
                "Wilcoxon_p_F1_onesided": summary["wilcoxon_p_f1_onesided"],
                "Wilcoxon_p_F1_twosided": summary["wilcoxon_p_f1_twosided"],
                "n_improve": summary["signs"]["improve"],
                "n_worse": summary["signs"]["worse"],
                "n_tie": summary["signs"]["tie"],
            }
        )
        for r in fa_rows:
            s6c_rows.append({"Train_fraction": f"{pct}%", **r})
        print("pct", pct, "FA", summary["F1_fa_mean_base"], summary["F1_fa_mean_psmp"], "n_ok", summary["n_ok_base"], summary["n_ok_psmp"], "n_pair", summary["n_pair"])

    write_csv(OUT / "TableS6b_failure_aware_fivesample.csv", s6b_rows)
    write_csv(OUT / "Table2_fivesample_val50_paired.csv", paired_rows)
    write_csv(OUT / "TableS6c_fivesample_status_by_case.csv", s6c_rows)

    # Fixed43 aligned
    for init in ("base", "psmp"):
        tag = f"pct100_{init}"
        # filter val50 scored results
        full = json.loads((EVAL / f"interface_5seed_{tag}_val50_ext.json").read_text())
        sub = {
            **full,
            "subset": "Fixed43",
            "n_cases": len(f43_cases),
            "results": [r for r in full["results"] if r["name"] in f43_names],
        }
        sub["n_ok"] = sum(1 for r in sub["results"] if r.get("status") == "evaluable")
        (EVAL / f"interface_5seed_{tag}_fixed43_ext.json").write_text(json.dumps(sub, indent=2))

    b43 = json.loads((EVAL / "interface_5seed_pct100_base_fixed43_ext.json").read_text())
    p43 = json.loads((EVAL / "interface_5seed_pct100_psmp_fixed43_ext.json").read_text())
    fa43, sum43 = build_fa_paired(b43, p43, "Fixed43")
    write_csv(OUT / "Table4b_fixed43_fivesample_failure_aware.csv", fa43)
    write_csv(
        OUT / "Table4c_fixed43_fivesample_paired.csv",
        [
            {
                "name": r["case"],
                "F1_base": r["base_f1_fa"],
                "F1_psmp": r["psmp_f1_fa"],
                "delta_F1": r["psmp_f1_fa"] - r["base_f1_fa"],
                "recall_base": r["base_recall_fa"],
                "recall_psmp": r["psmp_recall_fa"],
                "ilddt_base": r["base_ilddt_fa"],
                "ilddt_psmp": r["psmp_ilddt_fa"],
            }
            for r in fa43
            if r["in_paired_intersection"]
        ],
    )
    (OUT / "Table4_fixed43_fivesample_summary.json").write_text(
        json.dumps(
            {
                "failure_aware": sum43,
                "membership": json.loads((OUT / "Fixed43_membership.json").read_text()),
                "legacy_single_seed_note": (
                    "Legacy single-seed Fixed43 metrics (recall 0.033→0.046 etc.) are historical "
                    "only and are not Table 4 primary rows."
                ),
            },
            indent=2,
        )
    )
    print("Fixed43 summary:", json.dumps(sum43, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
