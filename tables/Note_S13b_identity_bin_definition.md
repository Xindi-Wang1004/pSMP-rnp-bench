# Identity-binned ΔF1 (Table S13b)

- **Source identities**: Table S11 partner-level MMseqs2 audit (same as S12 threshold summaries).
- **Primary axis**: `max_partner_identity = max(protein_identity, rna_identity)`.
- **Secondary axes**: protein-only and RNA-only (reported for completeness).
- **Bins**: [0,0.3), [0.3,0.4), [0.4,0.7), [0.7,1.0]; plus pooled mid [0.3,0.7) when atomic mid bins are sparse.
- **Estimand**: paired-intersection ΔF1 (pSMP − base) under the primary five-sample round.
- **Uncertainty**: bootstrap 95% CI on mean ΔF1 (2000 resamples); one-sided Wilcoxon (greater).
