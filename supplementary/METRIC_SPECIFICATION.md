# Metric specification | pSMP-rnp-resource-v0.1

Frozen evaluation definitions for the low-data RNP contact-recovery resource.
Authoritative implementation: `train/psmp/eval_interface.py` (deposit: `data/stage12_raw/psmp/eval_interface.py`).

## 1. Entities and contacts

| Setting | Definition |
|---------|------------|
| Residue pair | One protein polymer residue × one RNA polymer residue on the nominated chain pair |
| Distance | Minimum heavy-atom Euclidean distance between the two residues |
| **Evaluation contact** | Native pair with distance **≤ 5.0 Å** |
| **Pseudo-pretrain contact** (supervision) | Remapped intra-chain contacts with distance in **3.5–4.0 Å** after p_split / r_split (training only; not used as the primary eval endpoint) |
| Representative atom (interface lDDT) | Mean heavy-atom coordinate of the residue (`_rep_coord`) |

Native interface residue sets are residues that participate in ≥1 native 5 Å contact (`protein_interface` / `rna_interface` masks in `build_residue_distance_maps`).

## 2. Primary predictive metrics (contact recovery @5 Å)

Let \(N\) = native contact set, \(P\) = predicted contact set (pred residue pairs with min heavy-atom distance ≤ 5.0 Å on the aligned chain pair).

| Symbol | Formula |
|--------|---------|
| Recall | \(\|N \cap P\| / \|N\|\) |
| Precision | \(\|N \cap P\| / \|P\|\) (undefined / excluded if \|P\|=0 under conditional protocol) |
| F1 | \(2PR/(P+R)\) when both defined |
| Recovered count | \(\|N \cap P\|\) |

Aggregation in resource tables is **macro-average over cases** (mean of per-case scores), not micro-pooled over all residue pairs.

## 3. Auxiliary geometric metric

**Interface lDDT**: lDDT on concatenated interface representative atoms after Kabsch superposition of predicted interface atoms onto native interface atoms (radius 15 Å; thresholds 0.5/1.0/2.0/4.0 Å). Secondary endpoint only.

## 4. Failure / exclusion taxonomy

| Status | Meaning | Failure-aware F1 rule |
|--------|---------|------------------------|
| `evaluable` | `contact_precision_5A` and `contact_f1_5A` present | use reported F1 |
| `missing_prediction` | pred CIF missing / unreadable | **F1 := 0** |
| `no_predicted_contact` | native contacts exist but \|P\|=0 | **F1 := 0** under FA rule; excluded under legacy conditional tables |
| `no_interface` | \|N\|=0 | F1 **undefined**; excluded from FA denominator |
| `pipeline_error` | other evaluator errors | **F1 := 0** |

Legacy Table 2 / S5 use **paired-intersection of evaluable cases** only (conditional estimand).  
Table S6b reports **failure-aware** means over all defined cases.

## 5. Analysis sets and protocols

| Protocol | Cases | Sampling | Role |
|----------|-------|----------|------|
| Single-sample val50 | 50 | seed 101, 1 sample | Historical primary (exploratory) |
| Paired-intersection | ⊆ val50 | same | Sensitivity / conditional contrast |
| Failure-aware val50 | 50 | same | Preferred full-cohort estimand |
| Five-sample val50 | 50 | seed 101, 5 samples; best-of by predicted ipTM when scored | Pre-registered operational / coverage protocol (does **not** automatically replace primary) |
| spotlight10 / test10 | 10 ⊂ val50 | as published | Descriptive; not used for model selection |
| dev40 | 40 = val50 − test10 | as published | Development / ablation set |

## 6. Primary endpoint (locked for next confirmatory round)

- **Primary**: contact F1 @5 Å under the failure-aware val50 rule.
- **Secondary**: recall, precision, interface lDDT, success rates (≥1 / ≥5 recovered contacts), paired-intersection ΔF1, exposure strata, Top-K.
- Multiplicity: Holm (or pre-declared hierarchical) over the four train fractions for the primary only; current published p-values remain **nominal / exploratory** (Note 14).

## 7. Top-K note

Top-K analyses skip cases with residue-pair product > 50,000. Report exclusion counts whenever Top-K is cited.

_Versioned with resource freeze pSMP-rnp-resource-v0.1; text drafted 2026-08-03._
