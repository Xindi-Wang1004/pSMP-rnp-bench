# pSMP resource package — minimal tutorial

This package supports **low-data adaptation** experiments for RNA–protein interface contact recovery with Protenix + pSMP.

**Design goal:** a ready-to-use, leakage-aware **low-data fine-tuning benchmark** (fixed splits + contact metrics + checkpoints), not only a paper-code dump.

## 1. Environment

```bash
# Recommended: conda env `protenix`
# Python 3.11, PyTorch 2.7.1+cu126, Protenix 2.0.0
# Backbone weights: protenix_base_default_v1.0.0.pt
```

See Supplementary Note 7 for the full version table.

## 2. Benchmark splits (resource)

| File | Role |
|------|------|
| `fusai/train/data/rnp_real/val_protenix_inputs/test_split_manifest.json` | test10 definition |
| `fusai/train/data/rnp_real/val_protenix_inputs/cases_test10.tsv` | test10 case list |
| `fusai/train/data/rnp_real/val_protenix_inputs/cases.tsv` | full validation cases |
| `fusai/train/data/rnp_real/lowdata/` | 10/25/50/100% fine-tuning subsets (seed 42) |

## 3. One-command inference

```bash
export CUDA_VISIBLE_DEVICES=0
export PROTENIX_ROOT_DIR=/path/to/eval_env   # must contain checkpoint/protenix_base_default_v1.0.0.pt
# For a fine-tuned model, replace that checkpoint with pct10_psmp (or pct10_base) weights.

protenix pred \
  -i /path/to/case.json \
  -o /path/to/outdir \
  -s 101 \
  -n protenix_base_default_v1.0.0 \
  --use_msa false --use_template false --use_rna_msa false \
  --use_default_params true \
  --triatt_kernel torch --trimul_kernel torch \
  -e 1
```

## 4. One-command contact metrics

```bash
python3 nar_paper/scripts/compute_extended_metrics.py \
  --pred_root /path/to/pred_<tag> \
  --cases fusai/train/data/rnp_real/val_protenix_inputs/cases.tsv \
  --out /path/to/interface_<tag>_ext.json
```

Paired-intersection control (Table S5):

```bash
python3 nar_paper/scripts/build_paired_intersection_table.py
```

## 5. One-command low-data fine-tuning (reference)

Fine-tuning uses the frozen `lowdata/` fractions (seed 42) and Protenix training entrypoints. A typical 10% run pattern:

```bash
# Paths are project-local; see DATA_PATHS.md / RESOURCE_MANIFEST.md in the freeze package.
# Example: fine-tune on the 10% subset from a pSMP-pretrained checkpoint.
protenix train \
  --config /path/to/lowdata_pct10_psmp.yaml \
  --data_list fusai/train/data/rnp_real/lowdata/pct10/cases.tsv
```

Exact YAML/config names follow the deposited `rnp_real_lowdata` sweep (Supplementary Note 7). Reviewers should treat the frozen splits and contact-metric scripts as the protocol of record even if training configs are adapted to local hardware.

## 6. What this resource is for

Use this package to compare **low-data RNP adaptation** methods under a shared protocol (same splits, same contact metrics, same evaluability rules), not only to re-run pSMP itself.

**Reporting protocol (short):** same train/val/test10 · 10/25/50/100% fractions · five-seed native ranking · contact metrics over *n_ok* · disclose Table-S6-style exclusions · paired-intersection when comparing two methods · optionally Top-*K* contact precision for mutagenesis-oriented claims (Supplementary Note 8).

Frozen package ID: `pSMP-rnp-resource-v0.1` (`RESOURCE_MANIFEST.md`).  
Reviewer-access archive: available upon request at submission.  
Public GitHub + Zenodo: required at submission (insert DOI/URL before upload).
