TableS5_paired_intersection.csv — recomputed 2026-07-31

Source JSON:
  train/runs/rnp_real_lowdata/eval_work/interface_pct{10,25,50,100}_{base,psmp}_ext.json

Test:
  scipy.stats.wilcoxon(psmp, base, alternative='greater'|'two-sided', zero_method='wilcox')

Canonical one-sided F1 p (now in main-text Table 1):
  10% 0.074 | 25% 0.016 | 50% 0.125 | 100% 0.026

Two-sided F1 p (S5 only):
  10% 0.148 | 25% 0.031 | 50% 0.250 | 100% 0.053

F1 pair means match prior Table 1 point estimates.
Old unpublished/draft Table 1 p-values (0.042/0.008/0.044/8e-4) were NOT reproducible.

Regenerate:
  /home/wangxindi/miniconda3/envs/protenix/bin/python nar_paper/scripts/build_paired_intersection_table.py
