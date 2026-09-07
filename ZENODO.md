# Zenodo companion (large checkpoints only)

**DOI:** https://doi.org/10.5281/zenodo.21822700  

> **Benchmark splits are on GitHub**, not Zenodo.  
> Start here for data use: https://github.com/Xindi-Wang1004/pSMP-rnp-bench/blob/main/BENCHMARK_DATA.md

## Contents (Zenodo)

- 8× EMA weights: `pct{10,25,50,100}_{base,psmp}` (may be sharded as `*.pt.part.*`)
- `CHECKSUMS_checkpoints.sha256`
- `checkpoints_manifest.json`
- Reassembly notes: `REASSEMBLE_CHECKPOINTS.md`
- Human-readable pointer: upload/replace with `zenodo/README_ZENODO.md` from this repo

## GitHub vs Zenodo

| GitHub | Zenodo |
|---|---|
| splits, scripts, tables, docs | fine-tuned EMA checkpoints |
| Day-1 benchmark reuse | optional reference-weight reuse |

## Citation

Wang, Xindi; Luo, Junyu; Li, Yixue; Hon, Chitin (2026). pSMP-rnp-bench: checkpoints for low-data RNA-protein interface contact recovery. Zenodo. https://doi.org/10.5281/zenodo.21822700

## Important: which Zenodo version to download

- **Checkpoint binaries (full):** version DOI **https://doi.org/10.5281/zenodo.21822700** (92 files / sharded `*.pt.part.*`).
- **Docs + cleaned manifest only:** later versions may exist under the same concept DOI; they do **not** replace the binaries. Always download weights from `10.5281/zenodo.21822700`.
- Clean public manifest (no local paths): also in this repo at `zenodo/checkpoints_manifest.json`.
