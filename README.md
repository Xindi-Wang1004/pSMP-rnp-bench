# pSMP-rnp-resource-v0.1

Frozen benchmark for **low-data RNA–protein interface contact recovery**.

- **Primary deliverable:** evaluation protocol (frozen splits, disclosed evaluability `n_ok`, failure-aware full-cohort scoring, paired-intersection sensitivity, neighbor-exposure audits, reusable scoring scripts).
- **Bundled reference implementation:** pSMP (pseudo-complex pre-training on Protenix-base).
- **Large binaries (checkpoints):** deposited on Zenodo (see `DATA_PATHS.md` / release notes). This GitHub tree keeps code, splits, tables, and docs.

Manuscript target: *Nucleic Acids Research* — Methods and Resources.

## Quick start

See **`TUTORIAL.md`**.

```bash
# Contact metrics from predicted CIFs
python3 scripts/compute_extended_metrics.py \
  --pred_root /path/to/pred_<tag> \
  --cases splits/val50/cases.tsv \
  --out /path/to/interface_<tag>_ext.json
```

## Layout

```
pSMP-rnp-resource-v0.1/
├── README.md
├── TUTORIAL.md
├── RESOURCE_MANIFEST.md
├── DATA_PATHS.md
├── REVIEWER_ACCESS.md
├── LICENSE
├── scripts/          # scoring / table builders / reproducibility helpers
├── tables/           # machine-readable Table 2 / S* companions
├── splits/           # frozen partition manifests (lists; not large CIFs)
├── figures/          # main + supplementary figure assets
├── supplementary/    # metric/analysis plan notes
└── notes/            # deposited audit notes
```

## Citation

Please cite the NAR Methods and Resources article (DOI to be inserted) and this resource freeze `pSMP-rnp-resource-v0.1`.

## License

MIT (code and scripts). Structural data remain subject to PDB terms.
