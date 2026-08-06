# Primary analysis plan | pSMP-rnp-resource-v0.1 (locked 2026-08-03)

This plan governs **reporting of already-collected single-sample results** and **pre-registers** the next confirmatory round. It cannot retroactively make current p-values confirmatory (Note 14).

## Estimands

| Role | Estimand | Protocol |
|------|----------|----------|
| **Preferred full-cohort (v0.1 addendum)** | Mean contact F1 @5 Å with failures as 0 | Failure-aware val50 (Note 17 / Table S6b) |
| **Historical conditional** | Mean ΔF1 on paired evaluable intersection | Table 2 / S5 (sensitivity) |
| **Operational coverage** | Best-of-five by predicted ipTM | Five-sample val50 (in progress; secondary) |

`no_interface` cases are excluded from F1 denominators but counted in success-rate tables.

## Primary endpoint

- **One primary**: contact **F1 @5 Å**, failure-aware, on val50.
- Fractions 10/25/50/100% are four related contrasts of the same endpoint.
- Confirmatory multiplicity (next round only): **Holm** over the four fraction-wise one-sided Wilcoxon tests on case-level failure-aware ΔF1 (or hierarchical: test 10% first, then 25→50→100 only if previous passes). Current published p-values remain **nominal / exploratory**.

## Secondary / sensitivity

- Recall, precision, interface lDDT
- Complete-metric rate; rate recovering ≥1 / ≥5 native contacts
- Paired-intersection ΔF1 (legacy Table 2)
- Exposure strata (Tables S11–S12)
- Top-K (with large-case exclusion counts)
- Five-sample / best-of-ipTM (different estimand; do not silently replace FA primary)

## Randomness (next GPU round)

Minimum: 3 fine-tune init seeds × frozen subset seed 42; optionally 3 subset seeds. Report mean/median ΔF1, IQR, per-case sign consistency, failure rates.

## External / retrieval baselines

- k-NN contact transfer (Table S14; contacts export in progress)
- Protocol-matched external model only if identical inputs, sample budget, and FA rule; else context-only

## Split disclosure

val50 is a **public development benchmark** because all-mix / step schedule were informed by development readouts overlapping dev40 (see `MODEL_SELECTION_TIMELINE.md`). A temporal or cluster-disjoint holdout will be frozen before recipe changes for the next version.
