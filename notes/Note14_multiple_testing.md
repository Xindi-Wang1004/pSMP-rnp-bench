# Supplementary Note 14 | Multiple-testing sensitivity for paired F1 p-values

One-sided Wilcoxon p values on paired contact F1 (Table S5; four training fractions) were adjusted by Benjamini–Hochberg (BH) FDR control and by Bonferroni correction (m=4).

| Fraction | n_pair | raw p (F1) | BH p | Bonferroni p | nominal <0.05 | BH <0.05 |
|---|---:|---:|---:|---:|---|---|
| 10% | 11 | 0.0742188 | 0.0989584 | 0.296875 | no | no |
| 25% | 7 | 0.015625 | 0.0526244 | 0.0625 | yes | no |
| 50% | 12 | 0.125 | 0.125 | 0.5 | no | no |
| 100% | 39 | 0.0263122 | 0.0526244 | 0.105249 | yes | no |

Interpretation: under BH (m=4), fractions that remain <0.05 are reported in the table; Bonferroni is conservative at m=4. Main-text claims emphasize directional paired means and treat unadjusted p values as supportive rather than multiplicity-proof.
