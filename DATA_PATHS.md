# Data paths — pSMP-rnp-bench

## External-user entry

**Read [`BENCHMARK_DATA.md`](BENCHMARK_DATA.md) first.**  
Portable case lists: `splits/val50/cases_portable.tsv` · natives via `scripts/download_pdb_natives.py`.

## In this GitHub tree

| Artifact | Path |
|----------|------|
| val50 portable cases | `splits/val50/cases_portable.tsv` |
| val50 full cases (legacy absolute paths) | `splits/val50/cases.tsv` |
| val50 PDB list | `splits/val50/pdb_list.txt` |
| spotlight10 | `splits/spotlight10/` |
| train200 PDB list | `splits/train200/pdb_list.txt` |
| low-data fractions (seed 42) | `splits/lowdata/{pct10,pct25,pct50,pct100}/` |
| Temporal34 locked cohort | `splits/temporal34/` |
| machine-readable tables | `tables/` |
| scoring / download scripts | `scripts/` |

## Large artifacts (Zenodo; checkpoints only)

**DOI:** https://doi.org/10.5281/zenodo.21822700  
**Not included:** benchmark splits (those are on GitHub).

| Artifact | Role | Notes |
|----------|------|------|
| `pct{10,25,50,100}_{base,psmp}` EMA checkpoints | reference weights | often `*.pt.part.*`; see `zenodo/README_ZENODO.md` |
| `protenix_base_default_v1.0.0.pt` | Protenix backbone | from Protenix release; not redistributed here |

Frozen ID: `pSMP-rnp-bench`
