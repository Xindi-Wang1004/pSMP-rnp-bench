# splits/ — frozen RNP-ContactBench membership

Start here if you want to **use the benchmark in your own code**.

## Recommended files (portable)

| Path | Use |
|---|---|
| `val50/cases_portable.tsv` | 50 public development cases (`name`, `pdb_id`, chains, lengths) |
| `train200/pdb_list.txt` | 200 training PDB IDs |
| `lowdata/pct10|pct25|pct50|pct100/` | Seed-42 low-data fractions |
| `temporal34/` | Locked Temporal34 protocol-check cohort + low-homology track JSON |

## Download natives

```bash
python3 scripts/download_pdb_natives.py \
  --cases splits/val50/cases_portable.tsv \
  --out_dir data/natives/val50
```

Full narrative: [`../BENCHMARK_DATA.md`](../BENCHMARK_DATA.md).

## Note on `val50/cases.tsv`

Contains author-cluster absolute paths (`json_path`, `native_cif_gz`). Kept for provenance / internal reproducibility. External integrations should use **`cases_portable.tsv`**.
