# Supplementary Note 19 | n1100 pretrain-source vs val50 sequence-identity audit

## Scope
MMseqs2 easy-search of frozen val50 protein/RNA partner sequences against unique sequences in the n1100 pretrain manifest:
- **p_split gallery**: unique protein sequences from p_split rows
- **r_split gallery**: unique RNA sequences from r_split rows
- **rnp_real gallery**: unique protein and RNA sequences from rnp_real rows
- **all_n1100**: union used for primary nearest-neighbor reporting

Coverage rule matches Tables S11–S12 (qcov≥0.7 & tcov≥0.7, or alnlen/minlen≥0.7 with min(qcov,tcov)≥0.5).

## Summary (val50, n=50)

| Threshold | Protein neighbors | RNA neighbors | Both partners |
|-----------|-------------------|---------------|---------------|
| ≥30% | 13 | 7 | 3 |
| ≥40% | 12 | 7 | 3 |
| ≥70% | 10 | 7 | 2 |

Decomposition columns in Table S17 separate p_split / r_split / rnp_real contributions.

## Interpretation
This audit closes the gap left by Note 16 (exact PDB-ID dedup only). Non-zero neighbors at moderate identity mean some val partners share sequence neighborhood with pretrain sources; gains should not be interpreted as homology-free. Exact-ID overlap remains zero (Note 16).

## Outputs
- `tables/TableS17_pretrain_vs_val50_sequence_audit.tsv`
- `tables/TableS17_pretrain_vs_val50_threshold_summary.tsv`
- Workdir: `nar_paper_audit/mmseqs_pretrain_vs_val/`

_Generated 2026-08-05_
