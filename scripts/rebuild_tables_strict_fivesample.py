#!/usr/bin/env python3
"""Rebuild Table2/S6/Fixed43 from existing five-sample JSONs under locked protocol.

Does NOT re-run Protenix. Applies:
1) exclude incomplete large-backfill orphans (0 < n_samples < 5 and tokens > 4000) from n_ok
2) FA denominator = all cohort members; missing/no_interface/excluded → 0
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
EVAL = ROOT / "train/runs/rnp_real_lowdata/eval_work_5seed_val50"
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
F43 = ROOT / "nar_paper/data/cases_fixed43.tsv"
OUT = ROOT / "nar_paper/tables"
MAX_TOKENS = 4000.0
PROTOCOL = "five-sample_best_of_iptm_seed101_max_tokens_4000_strict"


def load_cases(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def wilcoxon_stats(deltas):
    p1 = p2 = None
    try:
        from scipy.stats import wilcoxon

        if any(abs(d) > 0 for d in deltas):
            p2 = float(wilcoxon(deltas, alternative="two-sided").pvalue)
            p1 = float(wilcoxon(deltas, alternative="greater").pvalue)
    except Exception as e:
        print("wilcoxon", e)
    return p1, p2


def n_samples_of(r: dict) -> int:
    if r.get("n_samples_found") is not None:
        return int(r["n_samples_found"])
    # infer from pred path existence if needed
    return 5 if r.get("contact_f1_5A") is not None else 0


def classify(r: dict, tokens: float) -> str:
    """Return evaluable | no_interface | missing_prediction | excluded_incomplete_large | other"""
    ns = n_samples_of(r)
    # incomplete large backfill orphans
    if 0 < ns < 5 and tokens > MAX_TOKENS:
        return "excluded_incomplete_large"
    if r.get("contact_f1_5A") is not None:
        return "evaluable"
    if r.get("n_native_contacts_5A") == 0:
        return "no_interface"
    err = str(r.get("error") or r.get("status") or "")
    if "missing" in err or err in ("missing_prediction", "missing_prediction_no_cif"):
        return "missing_prediction"
    if ns < 1:
        return "missing_prediction"
    return "other"


def fa0(status: str, r: dict, key="contact_f1_5A") -> float:
    if status == "evaluable" and r.get(key) is not None:
        return float(r[key])
    return 0.0


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def process_cohort(names: list[str], tok: dict, pct: int, cohort: str):
    base = json.loads((EVAL / f"interface_5seed_pct{pct}_base_val50_ext.json").read_text())
    psmp = json.loads((EVAL / f"interface_5seed_pct{pct}_psmp_val50_ext.json").read_text())
    # Prefer fixed43 json metrics if present for membership filter only; always read val50 trees
    bmap = {r["name"]: r for r in base["results"]}
    pmap = {r["name"]: r for r in psmp["results"]}

    fa_rows = []
    for n in names:
        br = bmap.get(n, {"error": "missing_prediction_no_cif"})
        pr = pmap.get(n, {"error": "missing_prediction_no_cif"})
        # if fixed43 scored extras into separate file, merge n_samples from fixed43 json when present
        for tag, rr in (("base", br), ("psmp", pr)):
            pass
        sb = classify(br, tok.get(n, 0.0))
        sp = classify(pr, tok.get(n, 0.0))
        fa_rows.append(
            {
                "Train_fraction": f"{pct}%",
                "case": n,
                "subset": "Fixed43" if cohort == "Fixed43" else "",
                "base_status": sb,
                "psmp_status": sp,
                "base_f1": fa0(sb, br),
                "psmp_f1": fa0(sp, pr),
                "base_iflddt": fa0(sb, br, "interface_lddt"),
                "psmp_iflddt": fa0(sp, pr, "interface_lddt"),
                "base_n_samples": n_samples_of(br),
                "psmp_n_samples": n_samples_of(pr),
                "in_paired_intersection": int(sb == "evaluable" and sp == "evaluable"),
            }
        )

    # enrich subset labels for val50 from spotlight/dev if available
    return fa_rows


def main():
    cases = load_cases(CASES)
    tok = {c["name"]: float(c.get("num_tokens") or 0.0) for c in cases}
    all_names = [c["name"] for c in cases]
    f43 = {r["name"] for r in load_cases(F43)}
    not_f43 = sorted(set(all_names) - f43)
    (OUT / "Fixed43_membership.json").write_text(
        json.dumps(
            {
                "definition": (
                    "Pre-specified: 43 val50 cases interface-evaluable under BOTH base and pSMP "
                    "in the original single-seed pct100 deposit "
                    "(eval_work/interface_pct100_{base,psmp}.json)."
                ),
                "n": 43,
                "names": sorted(f43),
                "excluded_from_val50_n7": not_f43,
                "exclusion_reason": (
                    "Failed the original single-seed interface-evaluable criterion for at least "
                    "one method; membership is not redefined by five-sample coverage."
                ),
            },
            indent=2,
        )
    )

    # Merge n_samples from fixed43 json into val50 maps for pct100 where available
    for init in ("base", "psmp"):
        vpath = EVAL / f"interface_5seed_pct100_{init}_val50_ext.json"
        fpath = EVAL / f"interface_5seed_pct100_{init}_fixed43_ext.json"
        if not vpath.exists():
            continue
        v = json.loads(vpath.read_text())
        if fpath.exists():
            fmap = {r["name"]: r for r in json.loads(fpath.read_text())["results"]}
            for r in v["results"]:
                fr = fmap.get(r["name"])
                if fr and fr.get("n_samples_found") is not None:
                    r["n_samples_found"] = fr["n_samples_found"]
                # if val50 missing metrics but fixed43 had them from 1-cif, keep metrics but
                # rebuild will exclude via classify — still copy n_samples
                if fr and r.get("contact_f1_5A") is None and fr.get("contact_f1_5A") is not None:
                    # import metrics so exclusion is explicit rather than silent miss
                    for k, val in fr.items():
                        if k not in r or r.get(k) is None:
                            r[k] = val
            vpath.write_text(json.dumps(v, indent=2))

    s6b, paired, s6c = [], [], []
    for pct in (10, 25, 50, 100):
        # ensure n_samples populated from pred trees when missing
        for init in ("base", "psmp"):
            path = EVAL / f"interface_5seed_pct{pct}_{init}_val50_ext.json"
            d = json.loads(path.read_text())
            changed = False
            for r in d["results"]:
                if r.get("n_samples_found") is None:
                    name = r["name"]
                    od = EVAL / f"pred_pct{pct}_{init}" / name
                    ns = len(list(od.rglob(f"{name}_sample_*.cif"))) if od.exists() else 0
                    r["n_samples_found"] = ns
                    changed = True
            if changed:
                path.write_text(json.dumps(d, indent=2))

        rows = process_cohort(all_names, tok, pct, "val50")
        # fill subset
        spot = set()
        for p in (
            ROOT / "train/data/rnp_real/val_protenix_inputs/cases_test10.tsv",
        ):
            if p.exists():
                spot = {r["name"] for r in load_cases(p)}
        for r in rows:
            r["subset"] = "spotlight10" if r["case"] in spot else ("Fixed43" if r["case"] in f43 else "dev40")
            if r["case"] in f43 and r["case"] not in spot:
                r["subset"] = "dev40" if r["case"] not in spot else r["subset"]
            # canonical: spotlight vs dev40 partition of val50
            r["subset"] = "spotlight10" if r["case"] in spot else "dev40"

        n_ok_b = sum(1 for r in rows if r["base_status"] == "evaluable")
        n_ok_p = sum(1 for r in rows if r["psmp_status"] == "evaluable")
        fa_b = [r["base_f1"] for r in rows]
        fa_p = [r["psmp_f1"] for r in rows]
        s6b.append(
            {
                "Train_fraction": f"{pct}%",
                "protocol": PROTOCOL,
                "n_cases": 50,
                "n_ok_base": n_ok_b,
                "n_ok_psmp": n_ok_p,
                "n_failure_aware_defined": 50,
                "F1_failure_aware_mean_base": mean(fa_b),
                "F1_failure_aware_mean_psmp": mean(fa_p),
                "Delta_F1_failure_aware_mean": mean([a - b for a, b in zip(fa_p, fa_b)]),
                "FA_denominator_policy": "all_50_missing_no_interface_excluded_incomplete_large_as_0",
            }
        )
        pair = [r for r in rows if r["in_paired_intersection"]]
        deltas = [r["psmp_f1"] - r["base_f1"] for r in pair]
        p1, p2 = wilcoxon_stats(deltas)
        paired.append(
            {
                "Train_fraction": f"{pct}%",
                "n_base": n_ok_b,
                "n_psmp": n_ok_p,
                "n_pair": len(pair),
                "Base_F1_pair": mean([r["base_f1"] for r in pair]),
                "pSMP_F1_pair": mean([r["psmp_f1"] for r in pair]),
                "Delta_F1": mean(deltas) if deltas else None,
                "Wilcoxon_p_F1_onesided": p1,
                "Wilcoxon_p_F1_twosided": p2,
                "n_improve": sum(1 for d in deltas if d > 0),
                "n_worse": sum(1 for d in deltas if d < 0),
                "n_tie": sum(1 for d in deltas if d == 0),
            }
        )
        s6c.extend(rows)
        print(
            f"pct{pct}: n_ok {n_ok_b}/{n_ok_p} n_pair {len(pair)} "
            f"FA {mean(fa_b):.4f}->{mean(fa_p):.4f} excluded_large "
            f"{sum(1 for r in rows if r['base_status']=='excluded_incomplete_large')}/"
            f"{sum(1 for r in rows if r['psmp_status']=='excluded_incomplete_large')}"
        )

    write_csv(OUT / "TableS6b_failure_aware_fivesample.csv", s6b)
    write_csv(OUT / "Table2_fivesample_val50_paired.csv", paired)
    write_csv(OUT / "TableS6c_fivesample_status_by_case.csv", s6c)

    # Fixed43 @100%
    rows43 = [r for r in s6c if r["Train_fraction"] == "100%" and r["case"] in f43]
    # rewrite subset label
    for r in rows43:
        r["subset"] = "Fixed43"
    n_ok_b = sum(1 for r in rows43 if r["base_status"] == "evaluable")
    n_ok_p = sum(1 for r in rows43 if r["psmp_status"] == "evaluable")
    fa_b = [r["base_f1"] for r in rows43]
    fa_p = [r["psmp_f1"] for r in rows43]
    pair = [r for r in rows43 if r["in_paired_intersection"]]
    deltas = [r["psmp_f1"] - r["base_f1"] for r in pair]
    p1, p2 = wilcoxon_stats(deltas)
    signs = {
        "improve": sum(1 for d in deltas if d > 0),
        "worse": sum(1 for d in deltas if d < 0),
        "tie": sum(1 for d in deltas if d == 0),
    }
    summary = {
        "cohort": "Fixed43",
        "protocol": PROTOCOL,
        "n_cases": 43,
        "n_ok_base": n_ok_b,
        "n_ok_psmp": n_ok_p,
        "n_failure_aware_defined": 43,
        "F1_fa_mean_base": mean(fa_b),
        "F1_fa_mean_psmp": mean(fa_p),
        "Delta_F1_fa": mean([a - b for a, b in zip(fa_p, fa_b)]),
        "recall_note": "FA includes all 43; excluded_incomplete_large and missing → 0",
        "n_pair": len(pair),
        "paired_F1_mean_base": mean([r["base_f1"] for r in pair]),
        "paired_F1_mean_psmp": mean([r["psmp_f1"] for r in pair]),
        "paired_Delta_F1": mean(deltas) if deltas else None,
        "wilcoxon_p_f1_onesided": p1,
        "wilcoxon_p_f1_twosided": p2,
        "signs": signs,
        "ilddt_fa_mean_base": mean([r["base_iflddt"] for r in rows43]),
        "ilddt_fa_mean_psmp": mean([r["psmp_iflddt"] for r in rows43]),
    }
    write_csv(OUT / "Table4b_fixed43_fivesample_failure_aware.csv", rows43)
    write_csv(
        OUT / "Table4c_fixed43_fivesample_paired.csv",
        [
            {
                "name": r["case"],
                "F1_base": r["base_f1"],
                "F1_psmp": r["psmp_f1"],
                "delta_F1": r["psmp_f1"] - r["base_f1"],
                "ilddt_base": r["base_iflddt"],
                "ilddt_psmp": r["psmp_iflddt"],
            }
            for r in pair
        ],
    )
    (OUT / "Table4_fixed43_fivesample_summary.json").write_text(
        json.dumps(
            {
                "failure_aware_and_paired": summary,
                "membership": json.loads((OUT / "Fixed43_membership.json").read_text()),
                "legacy_single_seed": {
                    "status": "historical_only_not_table4_primary",
                    "recall": "0.033→0.046",
                    "interface_lddt": "0.232→0.243",
                },
            },
            indent=2,
        )
    )
    print("Fixed43", json.dumps(summary, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
