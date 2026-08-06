# REVIEWER_ACCESS.md — pSMP-rnp-bench

**Purpose:** temporary access instructions for editors/reviewers before the public Zenodo/GitHub freeze is finalized.

## Preferred route (public freeze)

Once the Zenodo DOI and GitHub URL are inserted in the manuscript *Data and code availability* section, use those links. The public archive is the authoritative freeze (`pSMP-rnp-bench`).

## Private reviewer package (pre-publication)

If the public deposit is not yet live at the time of review:

1. Request the frozen reviewer tarball from the corresponding author (email in the manuscript header).
2. Verify integrity against `CHECKSUMS.sha256` in the package root (`shasum -a 256 -c CHECKSUMS.sha256`).
3. Start from `RESOURCE_MANIFEST.md` (inventory) and `DATA_PATHS.md` (splits/checkpoints path map).
4. Follow `TUTORIAL.md` for a 30-minute smoke test of the evaluation scripts.

## What must be present for main-text reproducibility

- Frozen train200 / val50 partitions and seed-42 low-data subsets (10/25/50/100%)
- Evaluation scripts under `scripts/` and tables under `tables/`
- Figure assets under `figures/` (or regenerated via `figures/generate_figures_server.py`)
- Checkpoint download pointers listed in `RESOURCE_MANIFEST.md` / `DATA_PATHS.md`

Large checkpoint binaries may be provided as download links rather than inside the manuscript ZIP; hashes for those files, when available, are recorded in `CHECKSUMS.sha256` or linked from the manifest.
