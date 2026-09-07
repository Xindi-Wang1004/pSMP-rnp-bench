# Benchmark data guide (RNP-ContactBench / pSMP-rnp-bench)

This page is the **entry point for external users** who want to download the benchmark and plug it into their own training / inference / scoring code.

## What lives where

| You need | Where | Notes |
|---|---|---|
| **Frozen case lists / splits** | **This GitHub repo** (`splits/`) | Primary benchmark dataset for protocol reuse |
| **Scoring scripts + tables** | This GitHub repo (`scripts/`, `tables/`) | Four-axis report helpers |
| **Fine-tuned reference checkpoints** | **Zenodo** [10.5281/zenodo.21822700](https://doi.org/10.5281/zenodo.21822700) | Optional; only needed to reproduce bundled pSMP/base EMA weights |
| **Native structures** | **PDB / RCSB** (download yourself) | Not redistributed here (PDB terms). Use `pdb_id` + chain IDs from the case lists |

**Zenodo does not host the benchmark splits.** If you only open the Zenodo record, you will see checkpoint shards (`.pt.part.*`), not `val50`/`train200`.

## 60-second download

```bash
git clone https://github.com/Xindi-Wang1004/pSMP-rnp-bench.git
cd pSMP-rnp-bench

# Portable validation cases (no cluster-absolute paths)
cut -f1-6 splits/val50/cases_portable.tsv | column -t | head

# Optional: download mmCIF natives for val50 from RCSB
python3 scripts/download_pdb_natives.py \
  --cases splits/val50/cases_portable.tsv \
  --out_dir data/natives/val50
```

Optional checkpoints (large):

```bash
# See ZENODO.md / zenodo/README_ZENODO.md for reassembly of *.part.* shards
# DOI: https://doi.org/10.5281/zenodo.21822700
```

## Core cohorts

| Cohort | File | Role |
|---|---|---|
| **train200** | `splits/train200/pdb_list.txt` | Frozen fine-tuning gallery (PDB IDs) |
| **val50** | `splits/val50/cases_portable.tsv` | Public **development** / reporting cohort (50 cases) |
| low-data fractions | `splits/lowdata/pct{10,25,50,100}/` | Seed-42 subsets of train200 |
| **Temporal34** | `splits/temporal34/` | Locked independent protocol-check cohort (not method ranking) |
| protein low-homology track | `splits/temporal34/temporal34_proteinlt40_track_v0.1.json` | Secondary track (n=19) |

Legacy path columns in `splits/val50/cases.tsv` (`json_path`, `native_cif_gz`) point to the authors’ cluster layout. **External users should prefer `cases_portable.tsv`.**

### `cases_portable.tsv` columns

| Column | Meaning |
|---|---|
| `name` | Case ID, typically `{pdb}_{proteinChain}_{rnaChain}` |
| `pdb_id` | PDB ID |
| `protein_chain_id` | Protein auth asym ID used in the freeze |
| `rna_chain_id` | RNA auth asym ID used in the freeze |
| `protein_len` / `rna_len` | Lengths |
| `num_tokens` | Protenix token count (shared case property; completion can still differ by method) |

## Plug into your own model / code

### A. Train / fine-tune on the frozen low-data recipe

1. Read PDB IDs from `splits/lowdata/pct10/pdb_list.txt` (or pct25/50/100).
2. Build your own tensors / crop / MSA features however you like.
3. Keep the **same case membership**; do not silently drop hard cases without reporting `n_ok`.
4. Treat **val50 as development** (may inform recipe selection). Do not present val50 gains alone as confirmatory SOTA.

### B. Run your predictor on val50 / Temporal34

1. Iterate `name,pdb_id,protein_chain_id,rna_chain_id` from the portable case list.
2. Produce one or more predicted structures (CIF) or contact maps per case.
3. Suggested prediction layout for the shared scorer:

```text
pred_root/
  1m5p_G_C/           # folder name == case `name`
    model_0.cif
    model_1.cif
    ...
```

4. Score with the harness (when using deposited CIF-based metrics):

```bash
python3 scripts/compute_extended_metrics.py \
  --pred_root /path/to/pred_root \
  --cases splits/val50/cases.tsv \
  --out /tmp/interface_mymethod_ext.json
```

> Note: full CIF contact scoring currently expects the authors’ `psmp.eval_interface` dependency (see `TUTORIAL.md`).  
> If you only need the **case membership** for your own metrics, `cases_portable.tsv` is enough—emit your metrics keyed by `name`, then co-report `n_ok`, conditional F1, FA-all, and exposure.

### C. Minimal “bring your own scorer” contract

For each frozen case `i = 1..N`:

1. Attempt prediction under your declared protocol.
2. If not evaluable → contribute `F1=0` to FA-all; do **not** drop the case from the denominator.
3. Report all four axes: conditional contact F1 (and/or paired intersection), `n_ok`, FA-all, sequence-neighbor exposure (or cite deposited audits).

## Smoke checks (no GPU)

```bash
python3 scripts/reproduce_main_tables.py --tables-dir tables
```

## License / provenance

- Code & split lists: MIT (this repository).
- Structures: PDB distribution terms (download from RCSB yourself).
- Checkpoints: see Zenodo record + Protenix / third-party model licenses.
