# Tutorial — scoring & reference reuse

For **downloading splits and plugging them into your own model**, read **[`BENCHMARK_DATA.md`](BENCHMARK_DATA.md)** first.

This page focuses on scoring predicted CIFs and (optionally) the bundled Protenix / pSMP reference path.

## 1. Environment

```bash
# Scoring helpers: Python ≥3.9, pandas, numpy
pip install pandas numpy

# Full CIF contact metrics currently import authors' eval helpers
# (see scripts/compute_extended_metrics.py). Bring-your-own metrics keyed by
# case `name` are fine if you follow the four-axis reporting contract.
```

Protenix fine-tuning / inference (optional reference): conda env with Protenix + backbone weights `protenix_base_default_v1.0.0.pt` (from the Protenix release; not on Zenodo).

## 2. Benchmark splits (use portable files)

| File | Role |
|------|------|
| `splits/val50/cases_portable.tsv` | Public development cohort (preferred) |
| `splits/train200/pdb_list.txt` | Training PDB IDs |
| `splits/lowdata/pct{10,25,50,100}/` | Seed-42 low-data fractions |
| `splits/temporal34/` | Locked Temporal34 protocol-check cohort |

Download natives:

```bash
python3 scripts/download_pdb_natives.py \
  --cases splits/val50/cases_portable.tsv \
  --out_dir data/natives/val50
```

## 3. Score predicted CIFs

Expected layout:

```text
pred_root/
  <case_name>/
    *.cif
```

```bash
python3 scripts/compute_extended_metrics.py \
  --pred_root /path/to/pred_<tag> \
  --cases splits/val50/cases.tsv \
  --out /path/to/interface_<tag>_ext.json
```

## 4. Optional: reference checkpoints

Zenodo DOI: https://doi.org/10.5281/zenodo.21822700  
See `ZENODO.md` and `zenodo/README_ZENODO.md` (reassemble `*.part.*` shards).

## 5. Reporting contract (short)

Co-report: conditional contact F1, `n_ok`, protocol-specific FA-all, and sequence-neighbor exposure.  
`val50` is a **development** cohort; Temporal34 is a **protocol check**, not a powered method-ranking set.

Frozen package ID: `pSMP-rnp-bench`.
