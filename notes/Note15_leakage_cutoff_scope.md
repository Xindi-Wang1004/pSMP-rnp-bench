# Supplementary Note 15 | Foundation-model cutoff / leakage audit scope

## What is reported in this resource
- Within-freeze partner-level MMseqs2 neighbor exposure (Note 13; Tables S11–S13).
- Cluster-label overlap between train200 and val50 (Note 12; Table S10).
- Temporal extension candidates use PDB release ≥2024-07-01 and exclude IDs already in the frozen 250-complex set (Notes 10–11).

## What cannot be completed from public metadata alone
Protenix-base and Chai-1 training corpora are not fully enumerable as public PDB-ID lists aligned to the exact weight files used here. Therefore membership of each val50 case in those pre-training sets cannot be certified case-by-case from open sources. We treat this as an explicit limitation: neighbor-exposure audits bound sequence relatedness within the resource freeze, but do not replace a vendor-complete cutoff audit.

## Operational stance for users
Downstream methods should disclose their own training cutoffs when comparing against this freeze, and may optionally provide a homology-hard companion split. The present release prioritizes a realistic-adaptation freeze with transparent within-resource audits over an unauditable claim of zero foundation-model leakage.

_Generated: 2026-08-03_
