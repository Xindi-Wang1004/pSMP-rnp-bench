# Zenodo companion (large checkpoints)

Fine-tuned EMA checkpoints for **`pSMP-rnp-bench`**.

**DOI:** https://doi.org/10.5281/zenodo.21822700 (`10.5281/zenodo.21822700`)  
Draft deposit: https://zenodo.org/deposit/21822700  
(Large EMA uploads may still be finalizing; the record is published when the deposit status is public.)

## Contents (Zenodo)

- 8× ~4.2 GB EMA weights: `pct{10,25,50,100}_{base,psmp}`
- `CHECKSUMS_checkpoints.sha256`
- `checkpoints_manifest.json`
- `README_ZENODO.md` (and, if split for upload, `REASSEMBLE_CHECKPOINTS.md` + `*.part.*`)

Cluster staging path (maintainers):  
`/home/wangxindi/RNA_Protein/fusai/zenodo_pSMP-rnp-bench/`

## GitHub vs Zenodo

This GitHub freeze keeps code, frozen splits, tables, figures, audits, tutorial, and package checksums.  
Large binaries live on Zenodo; see also `DATA_PATHS.md`.
