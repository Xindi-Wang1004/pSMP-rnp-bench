# Related resources positioning (draft for manuscript Table / Box)

Purpose: clarify what pSMP-rnp-resource-v0.1 adds vs existing RNA–protein structure / interface resources (review §4.6-23).

| Resource / class | Typical goal | Split & protocol | Failure transparency | Low-data adaptation focus |
|------------------|--------------|------------------|----------------------|---------------------------|
| CASP / CAPRI nucleic tracks | Absolute structure / docking accuracy under data-rich regimes | Challenge-defined | Organizer-controlled | No |
| RoseTTAFoldNA / AF3 / Chai / Boltz benchmarks | Zero-shot or standard eval of foundation models | Author-defined, often static | Often conditional on successful preds | Rarely explicit |
| RNA–protein docking benchmarks | Pose recovery from unbound/bound partners | Docking-specific | Varies | No |
| Interface residue / site predictors | Per-residue interface labels | ML splits | Usually complete labels | Sometimes few-shot, not structure F1 |
| **pSMP-rnp-resource-v0.1** | **Frozen evaluation contract for low-data RNP contact recovery** | Fixed train200/val50 + seed-42 subsets; paired + FA rules | **n_ok taxonomy + failure-aware F1** | **Yes (primary)** |

## What this resource is not
- Not a SOTA leaderboard for absolute DockQ/lDDT.
- Not a homology-hard generalization claim (family exposure is audited, not removed).
- Not a substitute for vendor-complete foundation-model cutoff audits.

## Differentiation one-liner
> Existing resources measure how well models fold or dock RNPs when evaluation is ad hoc; this freeze measures **fair, failure-aware, low-data adaptation** under a reusable protocol, with pSMP as a replaceable reference implementation.
