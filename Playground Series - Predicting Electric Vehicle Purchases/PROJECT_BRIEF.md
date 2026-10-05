# Kaggle Playground 2026 — Predicting EV Purchases: your second round of suggestions paid off

**Short version: your TE-smoothing suggestion produced the first real gain since round 3.** Lowering the
smoothing of the exact-value target encoding from m = 20 to m = 5 improved all three models. The public LB
went from **0.94582 to 0.94586**. The details are below, followed by the results of your other suggestions
and the background for reference.

The competition closes around **30 September 2026**.

**Current best: public LB 0.94586** (OOF AUC 0.945879).

---

## Your second suggestions — what happened

Screening method: each idea was added on its own to the submitted configuration and compared paired, fold by
fold, over 5 folds (baseline 0.945271). The acceptance bar is the same as before: every fold won, t ≥ 2.5,
and a bootstrap above zero in ≥ 95% of resamples.

| Your suggestion | Result |
|---|---|
| **Smoothed exact-value TE, try m ∈ [10, 50, 100]** | Larger m hurt: m = 50 −0.00006, m = 100 −0.00021. Going the *other* way worked: **m = 5 +0.000092** (t = 4.6, 5/5 folds). m = 1–3 formed a plateau with m = 5, so m = 5 was adopted. **Adopted.** |
| **Modulo / precision features** (income % 100 / 1000, last digit, commute decimals) | The raw target rates do differ (4.4% vs 18.8% for round-thousand incomes). But checked against the model's own predictions, the model already explains them: that gap is the 30,000 income floor. As features: +0.000023, noise. |
| **Row-index leakage check** | None. AUC of `id` = 0.5000, and target rates by `id` chunk match the model's predictions. |
| **Aggressive subsampling** | `subsample` 0.7: −0.000029, noise. `colsample` 0.5 and a higher `min_child_samples` had already measured at noise. |
| **Rank-power averaging** | p chosen nested per fold: mean Δ −0.0000000, nothing. |
| **Grouped TE keys** (concern × subsidy, subsidy × anxiety, …) | Not rerun. The finer concern × subsidy × anxiety key from your first round had measured +0.000008, noise. |
| **XGBoost pairwise ranking loss** | Built, but not yet measured. XGBoost's own AUC under a ranking objective counts every pair in a query group, O(n²), and took seconds per iteration on a 67k-row validation fold. It is fixed now, using an sklearn AUC as the eval metric. |
| **WoE** | Not run: WoE is a monotone transform of TE, so trees split on it identically. |
| **Platt / isotonic before rank averaging** | Not run: monotone transforms leave ranks unchanged, and isotonic only adds ties. |
| **Subgroup rank normalisation** | Not run: the order within each subgroup is the model's own. Only the cross-group order changes, and that is where the strongest signal (subsidy) lives. |
| **Contextual aggregations, focal loss, DAE** | Set aside: the hand-built features measured at zero, and there are only days left. |

### The winning change, round 8

All three models were retrained with m = 5:

| Model | before (m = 20) | after (m = 5) | Δ |
|---|---|---|---|
| LightGBM | 0.945726 | 0.945795 | +0.000069 |
| XGBoost | 0.945612 | 0.945726 | +0.000114 |
| CatBoost | 0.945720 | **0.945823** | +0.000103 |
| **Blend** | 0.945795 | **0.945879** | **+0.000084** |

- CatBoost improved on 10 of 10 folds, and LightGBM on 9 of 10.
- The public LB showed +0.00004. That is about half the CV gain, which is within public-board noise for a
  difference this small.
- This worked where my earlier tree-depth tuning had not: a feature change carries across all three
  libraries, while a LightGBM hyperparameter did not translate to XGBoost / CatBoost depth.

## Questions for the last day

1. **Per-column smoothing?** m = 5 is now one value for all 7 columns. Their support varies from ~50 rows
   per value (income) to ~170,000 (number of cars). Is per-column or empirical-Bayes smoothing worth
   trying at this point, or is m = 1–5 being flat a sign there's nothing left there?
2. **Last runs.** Two options are left, and there is time for one or two:
   - CatBoost with 3 seeds: it holds 53% of the blend on a single seed; ~40 min for an expected +0.00002–0.00004.
   - The pairwise XGBoost diversity model: ~10 min, uncertain return.

   Which would you spend it on?

---

## Your first suggestions — what happened (round 6)

**None of them moved the score.** Two results were informative.

| Your suggestion | Result |
|---|---|
| **Leakage / duplicates vs the original** | **Zero everywhere**: no duplicate rows in train, no test rows that occur in train, no rows identical to an original row. The generator copies single values (97.9% of incomes occur in the original), but never whole rows. |
| **Model trained on the original as a meta-feature** | **−0.000171**, t = −7.5, lost on 5/5 folds |
| **Shallow tree on the original, rules as features** | −0.000059, noise |
| **Pairwise TE of strong columns** | +0.000008 and +0.000015, noise |
| **`is_in_original_income` flag** | −0.000014, noise |
| **Extra Trees** | OOF 0.945498, correlation 0.9987 with the other trees, blend weight 0 |

**Finding 1 — the original's label is noisier than the competition's.** A model trained on the original
scores 0.903 AUC on the original's own held-out rows, but 0.937 on the competition rows. The generator seems
to have smoothed the label, which is why every original-derived feature hurt.

**Finding 2 — the rule is visible.** A depth-4 tree on the original predicts a purchase when:
- concern ≥ 4, subsidy = Yes, income above ~67.5k, and anxiety is not Medium; or
- concern = 3, subsidy = Yes, and income above ~134k.

The boosted models already capture this.

---

## Background

### The problem

- Predict `Will_Buy_EV` (Yes/No), scored with **ROC-AUC**, so only the ranking matters.
- 668,665 train / 286,571 test rows. 7 numerical + 6 categorical features, **no missing values**.
- Positive rate 17.5%.
- The data is synthetic, generated from the 10k-row *EV adoption Behavior* dataset (the "original").

### What the data looks like

- **Near-deterministic rules:**
  - `Subsidy_Available = No` → P(Yes) = **0.6%**. This covers 37% of rows.
  - `Range_Anxiety = High` → **0.1%**, `Medium` → 4.2%.
- **Strongest single features:** `Environmental_Concern_Level` (AUC 0.84) and income (0.67).
- **Non-monotonic columns:** age, the station counts and the number of cars sit at ~0.50 AUC. Their
  per-value target rates, however, vary 10–60× more than noise.
- **No train/test drift**, and no generator artifacts beyond the copied values.

### The pipeline (round 8)

1. **Folds:** 10-fold stratified CV, shared by everything.
2. **Features (39):**
   - 13 raw columns;
   - 7 **exact-value target encodings**, nested inside each CV fold, smoothing m = 5;
   - 7 **frequency encodings**;
   - 12 hand-built features.
3. **Models:**
   - LightGBM, 15 leaves, 3 seeds.
   - XGBoost, depth 5, 3 seeds.
   - CatBoost, depth 5.
4. **Blend:** rank-averaged, with weights optimised on OOF: 0.42 / 0.05 / 0.53. The models correlate at 0.999.

CV and LB have agreed to within ±0.00003 on all three submissions.

### Results

| Round | Change | OOF AUC | Public LB |
|---|---|---|---|
| 1 | Baseline, 64-leaf trees, TE m = 20 | 0.945741 | 0.94575 |
| 3 | Shallower trees (64 → 15 leaves) + seed bagging | 0.945795 | 0.94582 |
| 8 | TE smoothing m = 20 → 5 (your suggestion) | **0.945879** | **0.94586** |

All other rounds were rejected on CV and never submitted.

| What worked | Effect on AUC |
|---|---|
| Exact-value target encoding | **+0.0027** |
| Frequency encoding | +0.0005 |
| Shallower trees (127 → 15 leaves) | +0.00031 (LightGBM only) |
| **TE smoothing m = 20 → 5** | **+0.00008 on the blend** |
| 3-seed bagging | +0.00002 to +0.00005 per model |

| Did not work | Result |
|---|---|
| Hand-built features, TE of all categoricals, pairwise TE | noise |
| Anything derived from the original dataset | −0.0001 to −0.00017 |
| Neural net with exact-value embeddings | 0.0016 weaker, blend weight 0 |
| Extra Trees, rank power, subsample / colsample / regularisation | noise |
