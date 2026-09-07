# pSMP-rnp-bench

**RNP-ContactBench** (repository: `pSMP-rnp-bench`): a reusable, failure-aware scoring harness for low-data RNA–protein contact evaluation.

## Start here (external users)

| Goal | Go to |
|---|---|
| **Download & plug benchmark data into your code** | **[`BENCHMARK_DATA.md`](BENCHMARK_DATA.md)** ← main entry |
| Frozen split files | [`splits/README.md`](splits/README.md) |
| Score predicted CIFs | [`TUTORIAL.md`](TUTORIAL.md) |
| Reproduce deposited tables | [`REPRODUCE.md`](REPRODUCE.md) |
| Optional checkpoints | Zenodo [10.5281/zenodo.21822700](https://doi.org/10.5281/zenodo.21822700) (see [`ZENODO.md`](ZENODO.md)) |

```bash
git clone https://github.com/Xindi-Wang1004/pSMP-rnp-bench.git
cd pSMP-rnp-bench

# Portable val50 cases (no absolute cluster paths)
head -5 splits/val50/cases_portable.tsv

# Optional: fetch mmCIF natives from RCSB
python3 scripts/download_pdb_natives.py \
  --cases splits/val50/cases_portable.tsv \
  --out_dir data/natives/val50
```

Score your predictions (CIF layout: `pred_root/<case_name>/*.cif`):

```bash
python3 scripts/compute_extended_metrics.py \
  --pred_root /path/to/your_cifs \
  --cases splits/val50/cases.tsv \
  --out /tmp/interface_your_method_ext.json
```

Table-only smoke check (CPU, ~1 min):

```bash
python3 scripts/reproduce_main_tables.py --tables-dir tables
```

## What this resource is

- **Primary deliverable:** frozen splits + shared scorer contract + four-axis report (`n_ok`, conditional F1, FA-all, exposure audits).
- **Bundled reference (optional):** pSMP checkpoints on Zenodo — **not required** to use the harness.
- **Why four axes:** geometry-only and coverage-only summaries can disagree; the harness makes that visible.

Manuscript target: *Bioinformatics* — Original Paper / evaluation-resource framing.

## Layout

```
pSMP-rnp-bench/
├── BENCHMARK_DATA.md   # ← data download + integration guide
├── README.md
├── REPRODUCE.md
├── TUTORIAL.md
├── DATA_PATHS.md
├── ZENODO.md
├── zenodo/README_ZENODO.md
├── scripts/            # scoring / download_pdb_natives / table smoke checks
├── splits/             # frozen membership (portable TSVs)
├── tables/             # machine-readable companions
├── figures/
├── supplementary/
└── notes/
```

## Citation

Please cite the *Bioinformatics* article (DOI to be inserted) and this resource freeze `pSMP-rnp-bench`.

## License

MIT (code and split lists). Structural coordinates remain subject to PDB terms (download from RCSB).
