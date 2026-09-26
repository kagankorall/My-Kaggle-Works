# Kaggle Playground 2026 — Predicting EV Purchases: update after your suggestions

Thanks for the suggestions. I tested every cheap one on CV. **Short version: none of them moved the
score**, but two of the results tell us something about the data. The details are below, followed by a
few new questions. The background from last time is kept at the bottom for reference.

The competition closes around **30 September 2026**.

**Current best is unchanged: public LB 0.94582** (OOF AUC 0.945795).

---

## Your suggestions — what happened

Each idea was added on its own to the submitted configuration and compared paired, fold by fold. The
comparison used 5 folds, and the baseline scored 0.945271. The acceptance bar is the same as before
(explained under *How decisions were made* below).

| Your suggestion | What I did | Result |
|---|---|---|
| **#6 Leakage / duplicates vs the original** | Counted exact 13-column row matches: within train, train ↔ test, and train/test ↔ original | **Zero everywhere.** No duplicate rows in train, no test row that occurs in train, no row identical to an original row. The generator copies single values (97.9% of incomes occur in the original) but never whole rows, so there is nothing to hardcode. |
| **#1 Model trained on the original as a meta-feature** | LightGBM trained on the 10k original rows only, its prediction added as a feature | **−0.000171**, t = −7.5, lost on 5/5 folds. A clear loss. |
| **#2 Shallow tree on the original, rules as features** | Depth-4 decision tree fitted on the original, its leaf probability added as a feature | −0.000059, noise |
| **#3 Pairwise TE of strong columns** | Nested exact-value TE of concern × subsidy × anxiety (30 keys) and income × anxiety (~31 rows per key) | +0.000008 and +0.000015, noise |
| **#6 `is_in_original_income` flag** | As suggested | −0.000014, noise |
| **#4 Extra Trees** | LightGBM `extra_trees=True`, same features and capacity | OOF 0.945498, but correlation **0.9987** with the other trees. Blend weight 0 on all 10 folds. |
| Also: full-row target encoding | Nested TE of the whole row | −0.000032, noise (every row is unique anyway) |

**Set aside, with my reasons — push back if you disagree:**

- **Appending the original rows with an `is_original` flag.** It adds only 1.5% more data, and the rows would need their own nested encodings. Every original-derived feature so far has hurt, so I don't expect this to be different.
- **TabNet / NODE / FT-Transformer.** A large build with a few days left. My embedding MLP was 0.0016 short of the trees, and I'd need a model within ~0.0005 of them to have any chance in the blend.
- **Focal loss.** I don't see why re-weighting hard examples would help a pure ranking metric.
- **Soft pseudo-labelling.** Test predictions averaged over all folds carry every fold's labels. Training on them inside CV would leak the validation labels, so it would need a per-fold rebuild, including the encodings, to measure honestly. The expected gain on Playground data is small.

## Two findings from this round

1. **The original's label is noisier than the competition's.** The model trained on the original scores
   0.903 AUC on the original's own held-out rows, but **0.937** on the competition rows. The generator
   seems to have *smoothed* the relationship between features and label. That explains why every
   original-derived feature has hurt: each is a noisier estimate of a signal the 668k-row models already
   learn more cleanly.
2. **The rule is visible.** The depth-4 tree on the original says an EV purchase needs all of:
   - `Environmental_Concern ≥ 4`
   - `Subsidy = Yes`
   - income above ~67.5k
   - `Range_Anxiety` not `Medium`

   There is also a second path: concern = 3, subsidy, and income above ~134k. This is consistent with
   the EDA, and the boosted models already capture it.

## New questions

1. **Is this the ceiling?** My tree models have converged: they correlate at 0.999, and capacity, features
   and diversity are all flat. Have you seen any class of trick move a Playground AUC by +0.0002 or more
   once a pipeline reaches this point?
2. **The smoothed label.** Does the 0.903 vs 0.937 gap suggest how the labels were generated, and would
   you exploit that differently?
3. **The non-monotonic columns.** Age, the station counts and the number of cars carry their signal only
   per exact value. Exact-value TE captures it (+0.0027). Is there a better way than TE + frequency
   encoding to get at that kind of per-value signal?
4. **The set-aside list.** Would you still do any of it with ~3 days left? Pseudo-labelling done properly
   per fold is the one I'm least sure about dismissing.

---

## Background (from the first brief)

### The problem

- Predict `Will_Buy_EV` (Yes/No), scored with **ROC-AUC**, so only the ranking matters.
- 668,665 train / 286,571 test rows. 7 numerical + 6 categorical features, **no missing values**.
- Positive rate 17.5%.
- Synthetic data generated from the 10k-row *EV adoption Behavior* dataset (the "original").

| Numerical | Categorical |
|---|---|
| Age, Annual_Income_USD, Daily_Commute_km, Number_of_Cars_Owned, Charging_Stations_Near_Home, Charging_Stations_Near_Work, Environmental_Concern_Level (1–5) | Gender, City_Type, Current_Car_Type, Home_Charging_Possible, Subsidy_Available, Range_Anxiety_Level |

### What the data looks like

- **Near-deterministic rules:**
  - `Subsidy_Available = No` → P(Yes) = **0.6%**. This covers 37% of rows.
  - `Range_Anxiety = High` → **0.1%**, `Medium` → 4.2%, `Low` → 18.9%.
- **Strongest single features:**
  - `Environmental_Concern_Level`: AUC 0.84, close to monotonic.
  - Income: AUC 0.67.
- **Non-monotonic columns.** Age, the station counts and the number of cars sit at ~0.50 AUC. But their target rate per exact value varies 10–60× more than binomial noise would explain.
- **Values, not rows, are copied from the original.** 97.9% of the income values in train appear verbatim in the original, but no whole row does.
- **No train/test drift:** the KS statistic is ≤ 0.003 on every column.

### The pipeline

1. **Folds:** 10-fold stratified CV. The same folds are used everywhere.
2. **Features: 39 in total.**
   - Raw columns: 13.
   - **Exact-value target encoding** for every numerical column: 7. It is nested inside each CV fold, so no validation target reaches a training feature.
   - **Frequency encoding** (count of each exact value over train + test): 7.
   - 12 hand-built features.
3. **Models:**
   - LightGBM with 15 leaves, `min_child_samples=300`, 3 seeds.
   - XGBoost with depth 5, 3 seeds.
   - CatBoost with depth 5.
4. **Blend:** rank-averaged, weights optimised on OOF. The weights come out at about 0.50 LightGBM / 0.50 CatBoost / 0 XGBoost.

**How decisions were made.** Every idea was screened on CV before spending a submission. I used a paired per-fold test against the current configuration. A change was accepted only if it met all three conditions:
- It won on every fold, or on at least 9 of 10.
- t ≥ 2.5.
- A paired row bootstrap was above zero in ≥ 95% of resamples.

CV and LB have agreed to within 0.00003 on both submissions.

### Results

| Round | Change | OOF AUC | Public LB |
|---|---|---|---|
| 1 | Baseline: pipeline above, 64-leaf trees | 0.945741 | 0.94575 |
| 3 | Shallower trees (64 → 15 leaves) + seed bagging | 0.945795 | **0.94582** |

Rounds 2, 4, 5 and 6 were rejected on CV and never submitted.

| What worked | Effect on AUC |
|---|---|
| Exact-value target encoding | **+0.0027** |
| Frequency encoding | +0.0005 |
| Shallower trees: 127 → 15 leaves | +0.00031 (15 → 7 not significant) |
| 3-seed bagging | +0.00002 to +0.00005 per model |

| Did not work, before your suggestions | Result |
|---|---|
| 12 hand-built features | 0.000 |
| TE of all 6 categoricals combined | −0.00004, noise |
| Original-dataset TE per exact value (all columns / well-supported only) | −0.00010 / −0.00011 |
| `colsample_bytree=0.5`, `reg_lambda=10`, CatBoost depth 4 | noise |
| Neural net with an embedding per exact value (MLP 256→128) | OOF 0.9442, corr 0.993, blend weight 0 |
