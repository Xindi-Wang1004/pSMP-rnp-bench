# Reproduce main tables (RNP-ContactBench / pSMP-rnp-bench)

This checklist reproduces the **machine-readable companions** of the main tables from deposited artifacts.
Large CIF/checkpoint binaries live on Zenodo (`DATA_PATHS.md`); this tree keeps splits, scores, and builders.

## Environment

```bash
# Python ≥3.9; key deps used by scoring helpers
pip install pandas numpy biopython edlib
# Optional (exposure rebuild): mmseqs2 in PATH
```

Checkpoint / Protenix inference requires the authors' Protenix environment (see `TUTORIAL.md`).
**Table reproduction below does not require GPUs** if deposited score JSON/CSV tables are present.

## One-command smoke check

From the repository root:

```bash
python3 scripts/reproduce_main_tables.py --tables-dir tables --out /tmp/rnp_contactbench_repro
```

Expected: prints `OK` for Table 4/5 companions, Temporal34 (S9d), low-homology track (S9e), and exposure compact (S19d).

## Map: manuscript ↔ files

| Main-text item | Primary file(s) |
|---|---|
| Table 3 FA-all (val50) | `tables/Table2_fivesample_val50_paired.csv` (or deposited five-sample ext JSON) |
| Table 4 geometry–coverage–failure decomp (10%) | `tables/TableS6l_geometry_coverage_failure_decomp.csv` |
| Table 5 ≤4k jointly completable | `tables/TableS6m_token_stratum_FA_pct10.csv` |
| Figure 2 composition/exposure | `figures/Figure2_composition_exposure.{png,pdf}` |
| Temporal34 locked | `tables/TableS9d_temporal34_locked_summary.csv`, `tables/TableS7c_temporal34_locked.tsv` |
| Protein low-homology track | `tables/TableS9e_temporal34_lowhomology_tracks.csv`, `tables/temporal34_proteinlt40_track_v0.1.json` |
| Exposure audits | `tables/TableS11_partner_sequence_audit.tsv`, `tables/TableS19d_temporal34_exposure_compact.csv` |

## Frozen hashes

```bash
shasum -a 256 splits/val50/cases.tsv \
  tables/TableS9d_temporal34_locked_summary.csv \
  tables/TableS19d_temporal34_exposure_compact.csv
# compare to CHECKSUMS.sha256 when present
```

## What is *not* claimed

- Public `val50` is a **development** cohort (may inform recipe selection).
- Temporal34 validates the **frozen protocol / scorer**, not confirmatory pSMP SOTA.
- Modality-specific tracks are **not** a compute-matched unified leaderboard.

## License / provenance

Code: MIT. PDB-derived structures: PDB terms. Checkpoints: see Zenodo record and model licenses.
