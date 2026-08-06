#!/usr/bin/env python3
"""Third-party fine-tune baseline through the resource harness.

Trains a small residue-pair contact MLP on frozen low-data train contacts
(pct10 by default), then scores predicted contacts on val50 with the same
5 Å contact F1 / failure-aware contract used for Table 2.

This is intentionally NOT Protenix/pSMP: it demonstrates that a non-author
adaptation method can consume the frozen splits + contact metric harness.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/home/wangxindi/RNA_Protein/fusai")
PAPER = ROOT / "nar_paper"
CONTACTS = PAPER / "data/contacts"
CASES = ROOT / "train/data/rnp_real/val_protenix_inputs/cases.tsv"
LOWDATA = ROOT / "train/data/rnp_real/lowdata"
OUT_TBL = PAPER / "tables"
OUT_DATA = PAPER / "data"
THR = 5.0  # evaluation contact threshold (Å) — matches metric spec
SEED = 42
PCT = "pct10"
STRICT_PCT10_ONLY = True  # do not expand beyond frozen pct10 pdb_list
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_PAIRS_PER_COMPLEX = 20000
EPOCHS = 8
BATCH = 8192
LR = 1e-3

AA = "ACDEFGHIKLMNPQRSTVWY"
NT = "ACGU"
AA_I = {c: i + 1 for i, c in enumerate(AA)}  # 0=unk
NT_I = {c: i + 1 for i, c in enumerate(NT)}


def set_seed(s: int = SEED) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def load_contact(path: Path) -> dict | None:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    return d


def contact_pairs(d: dict) -> list[tuple[int, int]]:
    """Return 0-based (prot_idx, rna_idx) pairs."""
    raw = d.get("contacts") or d.get("pairs") or d.get("contact_pairs") or []
    out = []
    for x in raw:
        if isinstance(x, dict):
            i = x.get("i", x.get("prot", x.get("protein_idx", x.get("p"))))
            j = x.get("j", x.get("rna", x.get("rna_idx", x.get("r"))))
            if i is None or j is None:
                continue
            out.append((int(i), int(j)))
        elif isinstance(x, (list, tuple)) and len(x) >= 2:
            out.append((int(x[0]), int(x[1])))
    return out


def find_contact_file(pdb: str, prot: str | None = None, rna: str | None = None) -> Path | None:
    cands = []
    if prot and rna:
        for name in (f"{pdb}_{prot}_{rna}.json", f"{pdb.lower()}_{prot}_{rna}.json"):
            p = CONTACTS / name
            if p.exists():
                return p
    cands = sorted(CONTACTS.glob(f"{pdb.lower()}_*.json")) + sorted(CONTACTS.glob(f"{pdb}_*.json"))
    return cands[0] if cands else None


def resolve_train_cases(pct: str = PCT) -> list[dict]:
    """Prefer frozen lowdata/{pct}/pdb_list.txt; map each PDB to a contact JSON."""
    cands = [
        LOWDATA / pct / "cases.tsv",
        LOWDATA / f"{pct}_cases.tsv",
        LOWDATA / pct / "train_cases.tsv",
        PAPER / "data/stage12_raw/splits" / f"cases_{pct}.tsv",
    ]
    for p in cands:
        if p.exists():
            with p.open() as f:
                rows = list(csv.DictReader(f, delimiter="\t"))
            if rows and ("name" in rows[0] or "pdb_id" in rows[0]):
                return rows

    pdb_list_paths = [
        LOWDATA / pct / "pdb_list.txt",
        PAPER / "data/stage12_raw/train_val/train_pdb_list.txt",
        ROOT / "train/data/rnp_real/train_pdb_list.txt",
    ]
    pdbs: list[str] = []
    used_frozen_pct_list = False
    for pdb_list in pdb_list_paths:
        if not pdb_list.exists():
            continue
        pdbs = [x.strip().lower() for x in pdb_list.read_text().split() if x.strip()]
        used_frozen_pct_list = "lowdata" in str(pdb_list) and pct in str(pdb_list)
        break
    if not pdbs:
        return []
    if not used_frozen_pct_list:
        # subsample full train list only when frozen pct list missing
        frac = {"pct10": 0.1, "pct25": 0.25, "pct50": 0.5, "pct100": 1.0}.get(pct, 0.1)
        rng = random.Random(SEED)
        pdbs = sorted(pdbs)
        rng.shuffle(pdbs)
        n = max(1, int(round(len(pdbs) * frac)))
        pdbs = pdbs[:n]
    rows = []
    for pdb in pdbs:
        cf = find_contact_file(pdb)
        if cf is None:
            continue
        stem = cf.stem  # pdb_P_R
        parts = stem.split("_")
        prot = parts[1] if len(parts) >= 3 else ""
        rna = "_".join(parts[2:]) if len(parts) >= 3 else ""
        rows.append({"name": stem, "pdb_id": pdb, "protein_chain_id": prot, "rna_chain_id": rna})
    return rows


class PairDataset(Dataset):
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        self.x = torch.from_numpy(xs)
        self.y = torch.from_numpy(ys)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class ContactMLP(nn.Module):
    def __init__(self, n_aa=22, n_nt=6, d=32):
        super().__init__()
        self.aa = nn.Embedding(n_aa, d)
        self.nt = nn.Embedding(n_nt, d)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: [B,4] aa_idx, nt_idx, i_norm, j_norm
        aa = self.aa(x[:, 0].long())
        nt = self.nt(x[:, 1].long())
        cont = x[:, 2:]
        return self.mlp(torch.cat([aa, nt, cont], dim=-1)).squeeze(-1)


def encode_complex(d: dict, max_pairs: int = MAX_PAIRS_PER_COMPLEX) -> tuple[np.ndarray, np.ndarray] | None:
    pseq = (d.get("protein_seq") or d.get("prot_seq") or "").upper().replace("X", "A")
    rseq = (d.get("rna_seq") or "").upper().replace("T", "U")
    if not pseq or not rseq:
        return None
    pos = set(contact_pairs(d))
    if not pos and not d.get("n_contacts"):
        # still usable as all-negative if lengths known — skip empty
        pass
    Lp, Lr = len(pseq), len(rseq)
    # sample negatives + all positives
    pos_list = [(i, j) for i, j in pos if 0 <= i < Lp and 0 <= j < Lr]
    n_pos = len(pos_list)
    if n_pos == 0:
        return None
    n_neg = min(max(n_pos * 5, n_pos), max_pairs - n_pos)
    neg = set()
    rng = random.Random(hash(pseq[:20] + rseq[:20]) % (2**32))
    while len(neg) < n_neg:
        i = rng.randrange(Lp)
        j = rng.randrange(Lr)
        if (i, j) not in pos:
            neg.add((i, j))
    xs = []
    ys = []
    for i, j in pos_list + list(neg):
        aa = AA_I.get(pseq[i], 0)
        nt = NT_I.get(rseq[j], 0)
        xs.append([aa, nt, i / max(Lp - 1, 1), j / max(Lr - 1, 1)])
        ys.append(1.0 if (i, j) in pos else 0.0)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def build_train_arrays(train_rows: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    Xs, Ys = [], []
    n_used = 0
    for r in train_rows:
        pdb = (r.get("pdb_id") or r.get("name", "").split("_")[0]).lower()
        prot = r.get("protein_chain_id") or r.get("protein_chain") or ""
        rna = r.get("rna_chain_id") or r.get("rna_chain") or ""
        cf = find_contact_file(pdb, prot, rna)
        if cf is None:
            # try name stem
            name = r.get("name")
            if name:
                p = CONTACTS / f"{name}.json"
                cf = p if p.exists() else None
        if cf is None:
            continue
        d = load_contact(cf)
        if not d:
            continue
        enc = encode_complex(d)
        if enc is None:
            continue
        x, y = enc
        Xs.append(x)
        Ys.append(y)
        n_used += 1
    if not Xs:
        raise SystemExit("No train contacts encoded — check contacts/ and train list")
    return np.concatenate(Xs), np.concatenate(Ys), n_used


@torch.no_grad()
def predict_contacts(model: ContactMLP, d: dict, thr_logit: float = 0.0) -> set[tuple[int, int]]:
    pseq = (d.get("protein_seq") or d.get("prot_seq") or "").upper()
    rseq = (d.get("rna_seq") or "").upper().replace("T", "U")
    Lp, Lr = len(pseq), len(rseq)
    if Lp == 0 or Lr == 0:
        return set()
    # score all pairs in chunks
    pred = set()
    model.eval()
    chunk = 65536
    coords = [(i, j) for i in range(Lp) for j in range(Lr)]
    for k in range(0, len(coords), chunk):
        batch = coords[k : k + chunk]
        x = np.asarray(
            [
                [
                    AA_I.get(pseq[i], 0),
                    NT_I.get(rseq[j], 0),
                    i / max(Lp - 1, 1),
                    j / max(Lr - 1, 1),
                ]
                for i, j in batch
            ],
            dtype=np.float32,
        )
        logits = model(torch.from_numpy(x).to(DEVICE)).detach().cpu().numpy()
        for (i, j), logit in zip(batch, logits):
            if logit >= thr_logit:
                pred.add((i, j))
    return pred


def f1_sets(pred: set, truth: set) -> tuple[float, float, float]:
    if not pred and not truth:
        return 0.0, 0.0, 0.0
    tp = len(pred & truth)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return f1, rec, prec


def calibrate_threshold(model, train_rows, max_complexes=12) -> float:
    """Pick logit threshold maximizing mean F1 on a train subsample."""
    scores = []
    for r in train_rows[:max_complexes]:
        pdb = (r.get("pdb_id") or r.get("name", "").split("_")[0]).lower()
        cf = find_contact_file(pdb, r.get("protein_chain_id"), r.get("rna_chain_id"))
        if cf is None:
            continue
        d = load_contact(cf)
        if not d:
            continue
        truth = set(contact_pairs(d))
        if not truth:
            continue
        # get logits for truth+random neg to pick thr — use full predict is heavy; sample
        pseq = (d.get("protein_seq") or "").upper()
        rseq = (d.get("rna_seq") or "").upper().replace("T", "U")
        Lp, Lr = len(pseq), len(rseq)
        rng = random.Random(0)
        cand = list(truth) + [(rng.randrange(Lp), rng.randrange(Lr)) for _ in range(len(truth) * 5)]
        x = np.asarray(
            [
                [AA_I.get(pseq[i], 0), NT_I.get(rseq[j], 0), i / max(Lp - 1, 1), j / max(Lr - 1, 1)]
                for i, j in cand
                if 0 <= i < Lp and 0 <= j < Lr
            ],
            dtype=np.float32,
        )
        with torch.no_grad():
            logits = model(torch.from_numpy(x).to(DEVICE)).cpu().numpy()
        best_thr, best_f1 = 0.0, -1.0
        for thr in np.linspace(-3, 3, 25):
            pred = {cand[k] for k, lo in enumerate(logits) if lo >= thr}
            f1, _, _ = f1_sets(pred, truth)
            if f1 > best_f1:
                best_f1, best_thr = f1, float(thr)
        scores.append(best_thr)
    return float(np.median(scores)) if scores else 0.0


def main() -> None:
    set_seed()
    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    print("device", DEVICE, "contacts", CONTACTS, "n_files", len(list(CONTACTS.glob("*.json"))))

    train_rows = resolve_train_cases(PCT)
    print("train_rows", len(train_rows), "pct", PCT, "strict", STRICT_PCT10_ONLY)
    if (not STRICT_PCT10_ONLY) and len(train_rows) < 8:
        raise SystemExit("non-strict expand disabled for submission rebuild")
    if STRICT_PCT10_ONLY and len(train_rows) < 1:
        raise SystemExit("No pct10 contacts available after export")
    X, y, n_used = build_train_arrays(train_rows)
    print("encoded complexes", n_used, "pairs", len(y), "pos_rate", float(y.mean()))

    # class weight
    pos = float(y.sum())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=DEVICE)

    ds = PairDataset(X, y)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)
    model = ContactMLP().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    for ep in range(EPOCHS):
        total = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            logit = model(xb)
            loss = loss_fn(logit, yb)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
            n += len(yb)
        print(f"epoch {ep+1}/{EPOCHS} loss={total/max(n,1):.4f}")

    ckpt = OUT_DATA / f"seqpair_contact_mlp_{PCT}.pt"
    torch.save({"model": model.state_dict(), "pct": PCT, "seed": SEED}, ckpt)
    print("wrote", ckpt)

    # fixed threshold 0.0 after training with BCE; also try light calibration
    thr = 0.0
    try:
        thr = calibrate_threshold(model, train_rows)
    except Exception as e:
        print("calibrate skipped", e)
    print("logit_threshold", thr)

    # evaluate val50
    val_cases = list(csv.DictReader(CASES.open(), delimiter="\t"))
    rows = []
    f1s_fa = []
    for c in val_cases:
        name = c["name"]
        pdb = c["pdb_id"].lower()
        cf = find_contact_file(pdb, c.get("protein_chain_id"), c.get("rna_chain_id"))
        if cf is None:
            rows.append(
                {
                    "case": name,
                    "status": "missing_val_contacts",
                    "f1": 0.0,
                    "recall": 0.0,
                    "precision": 0.0,
                    "n_pred": 0,
                    "n_native": 0,
                }
            )
            f1s_fa.append(0.0)
            continue
        d = load_contact(cf)
        if not d:
            rows.append({"case": name, "status": "bad_contacts", "f1": 0.0, "recall": 0.0, "precision": 0.0, "n_pred": 0, "n_native": 0})
            f1s_fa.append(0.0)
            continue
        truth = set(contact_pairs(d))
        if not truth:
            rows.append({"case": name, "status": "no_interface", "f1": 0.0, "recall": 0.0, "precision": 0.0, "n_pred": 0, "n_native": 0})
            f1s_fa.append(0.0)
            continue
        # skip huge maps
        Lp = len(d.get("protein_seq") or "")
        Lr = len(d.get("rna_seq") or "")
        if Lp * Lr > 2_000_000:
            rows.append({"case": name, "status": "too_large", "f1": 0.0, "recall": 0.0, "precision": 0.0, "n_pred": 0, "n_native": len(truth)})
            f1s_fa.append(0.0)
            continue
        pred = predict_contacts(model, d, thr_logit=thr)
        f1, rec, prec = f1_sets(pred, truth)
        rows.append(
            {
                "case": name,
                "status": "evaluable",
                "f1": f1,
                "recall": rec,
                "precision": prec,
                "n_pred": len(pred),
                "n_native": len(truth),
            }
        )
        f1s_fa.append(f1)

    out_case = OUT_TBL / f"TableS14b_seqpair_mlp_{PCT}_by_case.csv"
    with out_case.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_ok = sum(1 for r in rows if r["status"] == "evaluable")
    summary = {
        "method": "seqpair_contact_mlp",
        "role": "third_party_finetune_baseline_through_harness",
        "train_fraction": PCT,
        "n_train_complexes_encoded": n_used,
        "n_val": len(rows),
        "n_ok": n_ok,
        "F1_failure_aware_mean": float(np.mean(f1s_fa)) if f1s_fa else None,
        "F1_evaluable_mean": float(np.mean([r["f1"] for r in rows if r["status"] == "evaluable"])) if n_ok else None,
        "logit_threshold": thr,
        "checkpoint": str(ckpt),
        "metric": "contact_F1@5A_vs_native_contact_json",
        "note": (
            "Non-Protenix pair MLP fine-tuned on frozen low-data train contacts; "
            "scored with the resource contact-F1 / failure-aware contract (missing→0)."
        ),
    }
    out_sum = OUT_TBL / f"TableS14b_seqpair_mlp_{PCT}_summary.json"
    out_sum.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("wrote", out_case)

    # side-by-side with k-NN FA if present
    knn = PAPER / "tables/TableS14_knn_contact_transfer_by_case.csv"
    if knn.exists():
        knn_f1 = []
        with knn.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") == "scored" and r.get("knn_f1") not in (None, ""):
                    knn_f1.append(float(r["knn_f1"]))
                else:
                    knn_f1.append(0.0)
        # align by case order in knn file vs our rows
        print("knn_FA_mean_approx", float(np.mean(knn_f1)) if knn_f1 else None)


if __name__ == "__main__":
    # fix typo guard
    main()
