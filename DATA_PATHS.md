# Data paths — pSMP-rnp-resource-v0.1

## In this GitHub tree

| Artifact | Path |
|----------|------|
| val50 case list | `splits/val50/cases.tsv` |
| val50 PDB list | `splits/val50/pdb_list.txt` |
| spotlight10 (test10) | `splits/spotlight10/cases_test10.tsv`, `test_split_manifest.json` |
| train200 PDB list | `splits/train200/pdb_list.txt` |
| low-data fractions (seed 42) | `splits/lowdata/{pct10,pct25,pct50,pct100}/` (`pdb_list.txt`, `rnp_indices.csv`) |
| lowdata / sweep manifests | `splits/lowdata/manifest.json`, `splits/sweep_manifest.json` |
| machine-readable tables | `tables/` |
| scoring scripts | `scripts/` |

## Large artifacts (Zenodo; not in GitHub)

| Artifact | Role | Notes |
|----------|------|------|
| `pct{10,25,50,100}_{base,psmp}` EMA checkpoints | fine-tuned weights for Table 2 | ~4.2 GB each; checksums in Zenodo `CHECKSUMS_checkpoints.sha256` |
| `protenix_base_default_v1.0.0.pt` | Protenix backbone | obtain from Protenix release; not redistributed here |
| Optional prediction CIF trees | five-sample val50 artifacts | deferred / partial for ≥4k-token cases (see manuscript §5) |

Cluster development paths (for maintainers only):

```
/home/wangxindi/RNA_Protein/fusai/train/runs/rnp_real_lowdata/
/home/wangxindi/RNA_Protein/fusai/train/data/rnp_real/
```

Frozen ID: `pSMP-rnp-resource-v0.1`
