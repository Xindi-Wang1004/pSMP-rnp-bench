# Zenodo record — checkpoints only (not the benchmark splits)

**DOI:** https://doi.org/10.5281/zenodo.21822700  
**Title:** pSMP-rnp-bench: checkpoints for low-data RNA-protein interface contact recovery

## Looking for the benchmark dataset?

The **frozen case lists / splits / scoring scripts** are on GitHub, not in this Zenodo deposit:

- GitHub: https://github.com/Xindi-Wang1004/pSMP-rnp-bench
- Data guide: https://github.com/Xindi-Wang1004/pSMP-rnp-bench/blob/main/BENCHMARK_DATA.md
- Portable val50 cases: `splits/val50/cases_portable.tsv`
- Download PDB natives: `python3 scripts/download_pdb_natives.py --cases splits/val50/cases_portable.tsv --out_dir data/natives/val50`

Native mmCIF files are **not** redistributed here (PDB terms). Checkpoints below are optional reference weights.

## What this Zenodo deposit contains

- Fine-tuned EMA checkpoints: `pct{10,25,50,100}_{base,psmp}` (often uploaded as `*.pt.part.*` shards)
- `CHECKSUMS_checkpoints.sha256`
- `checkpoints_manifest.json`
- `REASSEMBLE_CHECKPOINTS.md` (how to concatenate parts)

## Reassemble a sharded checkpoint

```bash
# Example pattern (see REASSEMBLE_CHECKPOINTS.md for exact names)
cat pct10_psmp__*.pt.part.* > pct10_psmp_ema.pt
shasum -a 256 pct10_psmp_ema.pt   # compare to CHECKSUMS_checkpoints.sha256
```

## Day-1 reuse without Zenodo

Most users only need GitHub:

1. Clone the repo  
2. Read `splits/val50/cases_portable.tsv`  
3. Run your model  
4. Optionally score CIFs with `scripts/compute_extended_metrics.py`

Zenodo is required only to reproduce the bundled Protenix-base / pSMP reference weights.
