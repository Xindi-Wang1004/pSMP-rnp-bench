#!/usr/bin/env python3
"""Build extended paper tables with precision/F1 from *_ext.json files."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
RUNS = ROOT / "train/runs"
PAPER = ROOT / "nar_paper"
PT = RUNS / "paper_tables"
LOW = RUNS / "rnp_real_lowdata/eval_work"
OUT = PAPER / "data"
TBL = PAPER / "tables"


def load_summary(ext_path: Path, fallback_path: Path | None = None) -> dict:
    d = json.loads(ext_path.read_text())
    s = {
        "n_ok": d.get("n_ok", 0),
        "iflddt": d.get("mean_interface_lddt"),
        "recall": d.get("mean_contact_recall_5A"),
        "precision": d.get("mean_contact_precision_5A"),
        "f1": d.get("mean_contact_f1_5A"),
    }
    if (s["precision"] is None or s["n_ok"] == 0) and fallback_path and fallback_path.is_file():
        fb = json.loads(fallback_path.read_text())
        s["iflddt"] = s["iflddt"] if s["iflddt"] is not None else fb.get("mean_interface_lddt")
        s["recall"] = s["recall"] if s["recall"] is not None else fb.get("mean_contact_recall_5A")
        s["n_ok"] = fb.get("n_ok", s["n_ok"])
    for k in ("iflddt", "recall", "precision", "f1"):
        if s[k] is None:
            s[k] = float("nan")
    return s


def fmt(x: float) -> str:
    return "—" if x != x else f"{x:.4f}"


def delta(a: float, b: float) -> str:
    if a != a or b != b:
        return "—"
    return f"{a - b:+.4f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TBL.mkdir(parents=True, exist_ok=True)

    rows = [
        ("Protenix base", "10% fine-tune (5-seed best-of-5)", "Best ipTM",
         load_summary(PT / "interface_5seed_base_pct10_test10_ext.json")),
        ("pSMP all-mix", "10% fine-tune (5-seed best-of-5)", "Best ipTM",
         load_summary(PT / "interface_5seed_allmix_pct10_test10_ext.json")),
        ("Chai-1", "Zero-shot", "Best aggregate_score",
         load_summary(RUNS / "chai1_baseline_eval/chai1_test10/interface_chai1_test10_ext.json")),
    ]

    t1_path = TBL / "Table1_test10.csv"
    with t1_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Setting", "Selection_criterion", "n_ok", "Interface_lDDT",
                    "Contact_recall_5A", "Contact_precision_5A", "Contact_F1_5A"])
        for method, setting, sel, s in rows:
            w.writerow([method, setting, sel, s["n_ok"],
                        fmt(s["iflddt"]), fmt(s["recall"]), fmt(s["precision"]), fmt(s["f1"])])

    t1_tex = TBL / "Table1_test10.tex"
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Independent test10 results (primary held-out evaluation).}",
        r"\label{tab:test10}",
        r"\begin{tabular}{lllccccc}",
        r"\toprule",
        r"Method & Setting & Selection criterion & $n_{\mathrm{ok}}$ & Interface lDDT & Recall @5\,\AA & Precision @5\,\AA & F1 @5\,\AA \\",
        r"\midrule",
    ]
    for method, setting, sel, s in rows:
        recall = fmt(s["recall"])
        prec = fmt(s["precision"])
        f1v = fmt(s["f1"])
        if method == "pSMP all-mix":
            recall = f"\\textbf{{{recall}}}"
            prec = f"\\textbf{{{prec}}}"
            f1v = f"\\textbf{{{f1v}}}"
        iflddt = fmt(s["iflddt"])
        if method == "Chai-1":
            iflddt = f"\\textbf{{{iflddt}}}"
        setting_tex = setting.replace("%", "\\%")
        sel_tex = sel.replace("_", "\\_")
        lines.append(
            f"{method} & {setting_tex} & {sel_tex} & {s['n_ok']} & {iflddt} & {recall} & {prec} & {f1v} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    t1_tex.write_text("\n".join(lines), encoding="utf-8")

    t2_rows = []
    for pct in [10, 25, 50, 100]:
        b = load_summary(LOW / f"interface_pct{pct}_base_ext.json", LOW / f"interface_pct{pct}_base.json")
        p = load_summary(LOW / f"interface_pct{pct}_psmp_ext.json", LOW / f"interface_pct{pct}_psmp.json")
        t2_rows.append({"pct": pct, "base": b, "psmp": p})

    t2_path = TBL / "Table2_lowdata.csv"
    with t2_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Train_fraction", "Base_iflddt", "pSMP_iflddt", "Delta_iflddt",
                    "Base_recall", "pSMP_recall", "Delta_recall",
                    "Base_precision", "pSMP_precision", "Delta_precision",
                    "Base_F1", "pSMP_F1", "Delta_F1"])
        for r in t2_rows:
            b, p = r["base"], r["psmp"]
            w.writerow([
                f"{r['pct']}%", fmt(b["iflddt"]), fmt(p["iflddt"]), delta(p["iflddt"], b["iflddt"]),
                fmt(b["recall"]), fmt(p["recall"]), delta(p["recall"], b["recall"]),
                fmt(b["precision"]), fmt(p["precision"]), delta(p["precision"], b["precision"]),
                fmt(b["f1"]), fmt(p["f1"]), delta(p["f1"], b["f1"]),
            ])

    t2_tex = TBL / "Table2_lowdata.tex"
    t2_lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Low-data fine-tuning: base vs pSMP across training fractions ($\Delta$ values are pSMP $-$ base).}",
        r"\label{tab:lowdata}",
        r"\small",
        r"\begin{tabular}{ccccccccccccc}",
        r"\toprule",
        r"Train \% & Base iflddt & pSMP iflddt & $\Delta$ iflddt & Base recall & pSMP recall & $\Delta$ recall & Base prec. & pSMP prec. & $\Delta$ prec. & Base F1 & pSMP F1 & $\Delta$ F1 \\",
        r"\midrule",
    ]
    for r in t2_rows:
        b, p = r["base"], r["psmp"]
        f1_psmp = fmt(p["f1"])
        d_f1 = delta(p["f1"], b["f1"])
        if r["pct"] in (10, 25, 50):
            f1_psmp = f"\\textbf{{{f1_psmp}}}"
            d_f1 = f"\\textbf{{{d_f1}}}"
        t2_lines.append(
            f"{r['pct']}\\% & {fmt(b['iflddt'])} & {fmt(p['iflddt'])} & {delta(p['iflddt'], b['iflddt'])} & "
            f"{fmt(b['recall'])} & {fmt(p['recall'])} & {delta(p['recall'], b['recall'])} & "
            f"{fmt(b['precision'])} & {fmt(p['precision'])} & {delta(p['precision'], b['precision'])} & "
            f"{fmt(b['f1'])} & {f1_psmp} & {d_f1} \\\\"
        )
    t2_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    t2_tex.write_text("\n".join(t2_lines), encoding="utf-8")

    summary = {"table1": {rows[i][0]: rows[i][3] for i in range(len(rows))}, "table2": t2_rows}
    (OUT / "extended_metrics_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"written {t1_path}, {t2_path}, {t2_tex}, {OUT / 'extended_metrics_summary.json'}")


if __name__ == "__main__":
    main()
