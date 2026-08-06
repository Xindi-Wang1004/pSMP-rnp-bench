# RESOURCE_MANIFEST.md — pSMP-rnp-bench

**Frozen package ID:** `pSMP-rnp-bench`  
**GitHub:** https://github.com/Xindi-Wang1004/pSMP-rnp-bench  
**Zenodo:** DOI pending (large checkpoints)

## GitHub contents

| Item | Path |
|------|------|
| Tutorial | `TUTORIAL.md` |
| This manifest | `RESOURCE_MANIFEST.md` |
| Path map | `DATA_PATHS.md` |
| Reviewer access | `REVIEWER_ACCESS.md` |
| License | `LICENSE` |
| Checksums (tree files) | `CHECKSUMS.sha256` |
| Evaluation / table scripts | `scripts/` |
| Tables | `tables/` |
| Frozen split lists | `splits/` |
| Figures | `figures/` |
| Analysis-plan notes | `supplementary/`, `notes/` |

## Zenodo contents (large)

- Fine-tuned checkpoints for fractions 10/25/50/100% × {base, pSMP}
- Checkpoint checksum manifest
- Optional: extended audit dumps

## Protocol of record

Same train200 / val50 / spotlight10 · seed-42 low-data subsets · five-sample best-of-ipTM primary · failure-aware full-cohort F1 · paired-intersection sensitivity · disclose `n_ok` / exclusions.
