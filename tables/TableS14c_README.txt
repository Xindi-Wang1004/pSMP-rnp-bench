Table S14c PEFT harness notes
=============================
Primary PEFT-vs-pSMP verdict uses the 24-case pSMP-evaluable intersection, not FA.
PEFT-from-base n_ok=43 vs PEFT-on-pSMP n_ok=24: same pairformer-only fine-tuning axis, but
different frozen diffusion weights (Protenix-base init vs pSMP-pretrain init), which change
CIF yield. Do not interpret the n_ok gap as a recipe superiority claim.
PEFT-on-pSMP vs PEFT-from-base on the 24-case set: mean Δ≈+0.008; 6↑/5↓/13 ties; one-sided
sign test p≈0.50; Wilcoxon two-sided p≈0.37 (not significant).
