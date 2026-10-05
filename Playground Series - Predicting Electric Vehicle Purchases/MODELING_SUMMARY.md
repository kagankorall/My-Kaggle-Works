# Playground Series — Predicting Electric Vehicle Purchases

Working log for the competition. There is one row per submission, with what changed, so that no submission is
left unattributed (the S6E8 log lost one and could never explain its best score).

- Task: binary classification of `Will_Buy_EV` (Yes/No), scored with **ROC-AUC**. Closes around 30 September 2026.
- Data: 668,665 train / 286,571 test rows, 7 numerical + 6 categorical features, no missing values.
- Positive rate: 17.46%.
- Notebook: [`notebook/ev_purchase_prediction.ipynb`](notebook/ev_purchase_prediction.ipynb).

**Best: rounds 8 and 9 — both public LB 0.94586** (OOF 0.945879 / 0.945884). Round 9 adds CatBoost seed
bagging for +0.000005 on CV, and the leaderboard duly showed no difference — below what it can resolve.
Over the competition the public LB went from 0.94575 to 0.94586 (+0.00011).
**Final selections: round 9 first (highest CV, bagged), round 8 second.** Choose by CV, not by public-LB
differences of this size.

---

## 1. Leaderboard history

| Round | Change | OOF AUC | LB | Δ LB |
|---|---|---|---|---|
| 1 | Baseline: raw + engineered + nested TE + frequency encoding, 10-fold LGB / XGB / Cat, rank blend | 0.945741 | 0.94575 | — |
| 2 | Original-dataset TE (`orig`, `orig_lite`). Both hurt on CV; not submitted | — | — | — |
| 3 | Shallower trees (LightGBM 15 leaves, XGB / Cat depth 5) + 3-seed LGB / XGB | 0.945795 | 0.94582 | +0.00007 |
| 4 | 7-leaf edge check, CatBoost depth 4. Neither established; not submitted | 0.945800 | — | — |
| 5 | Neural net with exact-value embeddings. Blend weight 0; not submitted | 0.945795 | — | — |
| 6 | First outside review (exact matches, original-data model / tree, pairwise TE, `extra_trees`). Nothing established | 0.945795 | — | — |
| 7 | Second outside review (diagnostics, TE smoothing, subsample, digit patterns). **TE m = 5 adopted**, CV only | — | — | — |
| 8 | All three models retrained with TE smoothing m = 5 (was 20) | **0.945879** | **0.94586** | +0.00004 |
| 9 | CatBoost 3 seeds. Also measured: per-column smoothing (noise), pairwise XGBoost (+0.000003) | **0.945884** | **0.94586** | 0.00000 |

**CV / LB agreement**

| Round | blend OOF | LB | offset |
|---|---|---|---|
| 1 | 0.945741 | 0.94575 | +0.00001 |
| 3 | 0.945795 | 0.94582 | +0.00003 |
| 8 | 0.945879 | 0.94586 | −0.00002 |
| 9 | 0.945884 | 0.94586 | −0.00002 |

The OOF gain predicted +0.000054 for round 3, and the leaderboard gave +0.00007. For round 8 it predicted
+0.000084, and the leaderboard gave +0.00004. Both differences are within public-LB noise. The public board
scores only a subset of the 287k test rows, while the OOF comparison is paired over all 669k train rows. The
CV is therefore trusted over the public LB, and ideas are screened on it without spending submissions.

---

## 2. What the data turned out to be

- **Clean.** There are no missing values, and there is no train/test drift: the KS statistic is ≤ 0.003 on every numerical column.
- **Two near-deterministic rules:**
  - `Subsidy_Available = No` → P(Yes) = **0.0058** (37% of train).
  - `Range_Anxiety_Level` → High 0.0014, Medium 0.0417, Low 0.1890.
- **`Environmental_Concern_Level`** (5 values) is the strongest single column: AUC 0.8435, nearly monotonic.
- **`Annual_Income_USD`** comes next at AUC 0.6704. Its exact-value encoding adds +0.04 on top of that.
  Income has a floor at 30,000, and commute has one at 5.0, which looks like clipping in the source.
- **Non-monotonic columns.** `Age`, the station counts and `Number_of_Cars_Owned` sit at ~0.50 AUC. Their
  per-value target rates, however, disperse 10–60× beyond binomial noise, the same pattern as S6E8.
- **The generator copies values, not rows.** 97.9% of train incomes occur verbatim in the original. There
  are zero full-row duplicates in train, zero train/test row overlaps, and zero rows identical to an
  original row.
- **The original's label is noisier than the competition's.** A model trained only on the 10k original rows
  scores 0.903 AUC on the original's held-out rows, but 0.937 on the competition rows. The generator appears
  to have smoothed the label relationship. That explains why every original-derived feature hurt.
- **The recovered rule.** A depth-4 tree on the original predicts a purchase on two paths:
  - concern ≥ 4, subsidy = Yes, income > ~67.5k, and anxiety not Medium;
  - concern = 3, subsidy = Yes, and income > ~134k.

  The boosted models already capture this.
- **No generator artifacts beyond the values.** Row order (`id`) is pure noise: AUC 0.5000, and each
  per-chunk rate matches the model's own prediction. Income digit patterns are fully explained by the model.

---

## 3. Current pipeline (round 8)

```
load → EDA → features → nested exact-value TE (m = 5) + frequency encoding
     → 10-fold CV: LightGBM (3 seeds) / XGBoost (3 seeds) / CatBoost (1 seed)
     → rank-optimised blend → submission.csv
```

- **Folds:** `StratifiedKFold(10, shuffle=True, random_state=42)`, created once and shared by every
  model and every encoding.
- **Target encoding is nested.** For each outer fold, the training rows are encoded out-of-fold over 5
  inner folds, and the validation and test rows use the table fitted on that fold's training rows. No
  validation target ever reaches a feature. Smoothing is `(sum + m·prior) / (count + m)` with **m = 5**.
  The encodings are integer codes + `bincount`, cached per fold.
- **Features (39):**
  - 13 raw columns;
  - 12 hand-built features (measured at zero, kept anyway);
  - 7 exact-value TE columns, one per numerical column;
  - 7 frequency encodings, counted over train + test.
- **Models:**
  - LightGBM: 15 leaves, `min_child_samples=300`, lr 0.03.
  - XGBoost: depth 5, `min_child_weight=100`, lr 0.03.
  - CatBoost: depth 5, lr 0.05. It early-stops on Logloss, because an AUC eval made it ~10× slower.
- **Blend:** rank-averaged, with simplex weights optimised on OOF: LightGBM 0.42 / XGBoost 0.05 / CatBoost 0.53. The
  models correlate 0.9990–0.9994, so the blend adds +0.00006 over the best single model.
- **Runtime:** ~31 min in total (LightGBM 6.5 min, XGBoost 6.5 min, CatBoost 18 min).

**Decision rule.** Every idea was screened on CV before any submission, with a paired per-fold test
against the configuration actually submitted. A change is adopted only if it passes all three checks:

- it wins on every fold (or on at least 9 of 10 at 10 folds);
- t ≥ 2.5;
- a paired row bootstrap is above zero in ≥ 95% of resamples.

Screening runs used LightGBM at lr 0.08 on 5 of the 10 folds (~35 s per variant).

---

## 4. What moved the score

| Change | Effect on AUC | Round |
|---|---|---|
| Exact-value target encoding | **+0.0027** (removing it costs this much) | 1 |
| Frequency encoding | +0.0005 | 1 |
| Shallower trees, 64 → 15 leaves | +0.000135 (LightGBM, full run) | 3 |
| TE smoothing, m = 20 → 5 | **+0.00009** per model, +0.000084 on the blend | 7–8 |
| 3-seed bagging | +0.00002 – 0.00005 per model | 3 |

Everything else measured at noise or hurt (§6).

---

## 5. Round by round

### Round 1 — baseline and feature-group ablation

Ablation, 5 folds, paired against raw + eng + te + freq (0.945089):

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| `- te` | **−0.002691** | −28.1 | 0/5 | 0.00 | TE essential |
| `- freq` | **−0.000499** | −18.6 | 0/5 | 0.00 | freq helps |
| `- eng` | +0.000004 | 0.18 | 3/5 | 0.68 | noise, kept |
| `+ combo` (TE of all 6 categoricals) | −0.000041 | −1.11 | 1/5 | 0.17 | noise, skipped |

| Model | OOF AUC | typical best iter | wall time |
|---|---|---|---|
| LightGBM (64 leaves) | 0.945542 | 280–610 | 2 min |
| XGBoost (depth 7) | 0.945578 | 400–570 | 2 min |
| CatBoost (depth 7) | 0.945681 | 490–840 | 16 min |
| blend, rank (0.17 / 0.25 / 0.59) | 0.945741 | | |

`ev_readiness` was the top feature by LightGBM gain, yet the whole `eng` group measured at zero. Gain shows
where the trees split, not what the model would lose; the same information is recoverable from the raw
columns.

### Round 2 — the original dataset (closed)

| Variant | mean Δ | t | folds won | boot P(>0) |
|---|---|---|---|---|
| `+ orig`: per-value TE from the 10k original rows | −0.000104 | −2.78 | 0/5 | 0.00 |
| `+ orig_lite`: only keys with ≥ 20 original rows per value | −0.000114 | −6.28 | 0/5 | 0.00 |

Dropping the noisy columns (income: 1.1 rows per value) made the loss *more* consistent. The loss is
therefore not caused by those columns. The original's per-value rates are a 10k-row estimate of what the 668k-row TE
already measures precisely.

### Round 3 — tree capacity

LightGBM sweep, 5 folds at lr 0.08, paired against the 64-leaf setting (0.945089):

| Setting | mean Δ | t | folds won | verdict |
|---|---|---|---|---|
| **`nl15_mcs300`** | **+0.000181** | 5.20 | 5/5 | ADOPT (best) |
| `nl15_mcs1000` / `nl15_mcs60` | +0.000159 / +0.000155 | 5.84 / 3.58 | 5/5 | ADOPT |
| `nl31_mcs300` / `nl31_mcs60` | +0.000145 / +0.000097 | 7.03 / 5.72 | 5/5 | ADOPT |
| `col0.5` / `nl31_mcs1000` | +0.000082 / +0.000074 | 3.25 / 1.83 | 4/5 | — |
| `nl64_mcs300` / `l2_10` | −0.000010 / −0.000041 | | | — |
| `nl64_mcs1000` | −0.000094 | −4.76 | 0/5 | — |
| `nl127_*` | −0.00020 to −0.00031 | | 0/5 | — |

Leaf count is the whole effect. A label built from a few near-rule features plus per-value offsets is fitted
better by many shallow trees than by fewer deep ones.

| Model | round 1 | first seed | 3 seeds | from capacity | from bagging |
|---|---|---|---|---|---|
| LightGBM (15 leaves) | 0.945542 | 0.945677 | 0.945726 | **+0.000135** | +0.000049 |
| XGBoost (depth 5) | 0.945578 | 0.945591 | 0.945612 | +0.000013 | +0.000021 |
| CatBoost (depth 5) | 0.945681 | 0.945720 | — | +0.000039 | — |
| blend | 0.945741 | | **0.945795** | | |

The LightGBM result did **not** carry to the other two libraries. The depth translation kept round 1's
ratio (64 leaves ↔ depth 7), but depth 5 allows 32 leaves, which is not the same capacity.

### Round 4 — edge checks (closed)

| Test | vs | mean Δ | t | folds won | verdict |
|---|---|---|---|---|---|
| LightGBM 7 leaves (`mcs60`) | 15 leaves | +0.000053 | 1.74 | 4/5 | not established |
| LightGBM 7 leaves (`mcs300` / `mcs1000`) | 15 leaves | +0.000034 / −0.000001 | | | nothing |
| `nl15_mcs300` + `col0.5` | 15 leaves | +0.000004 | 0.11 | 1/5 | nothing |
| CatBoost depth 4, 10 folds | depth 5 | −0.000003 | −0.28 | 5/10 | nothing |

The capacity curve has flattened, and CatBoost's shortfall was not a capacity problem.

### Round 5 — neural net (closed)

The net used an MLP (256 → 128) with a learned embedding for every exact value, plus 32 numeric inputs, on MPS
at ≈ 9 s per fold. It reached OOF 0.944174 with a correlation of 0.993 to the trees. That is genuinely more
diverse than the trees are among themselves, but it is 0.0016 weaker. Its nested blend weight was **0 on
all 10 folds**.

The first attempt hung at 0% CPU. torch deadlocked on its first CPU op because the kernel had three OpenMP
copies loaded (torch's, sklearn's, and Homebrew's, used by LightGBM / XGBoost). The fix was
`torch.set_num_threads(1)` plus a numpy permutation. Tree predictions were saved to `preds/`, so the
forced kernel restart did not mean retraining.

### Round 6 — first outside review (`friend_suggestion.md`), section 3d — nothing established

The exact-match diagnostics are in §2: there are no duplicates, overlaps or original-row matches. The ablation
used 5 folds, paired against the submitted set (0.945271):

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| `+ pair_inc_anx` (income × anxiety TE) | +0.000015 | 0.32 | 3/5 | 0.70 | noise |
| `+ pair_csa` (concern × subsidy × anxiety TE) | +0.000008 | 0.35 | 2/5 | 0.60 | noise |
| `+ in_orig` (income value exists in original) | −0.000014 | −1.17 | 2/5 | 0.43 | noise |
| `+ row` (full-row TE) | −0.000032 | −2.24 | 1/5 | 0.17 | noise |
| `+ orig_tree` (depth-4 tree on original) | −0.000059 | −1.82 | 1/5 | 0.01 | noise |
| `+ orig_model` (LightGBM trained on original) | **−0.000171** | −7.46 | 0/5 | 0.00 | hurts |

LightGBM with `extra_trees` scored 0.945498, correlated 0.9987 with the other trees, and took a nested
blend weight of 0.

Several review suggestions were set aside:

- Appending the original rows: 1.5% more data, which would need its own encodings.
- TabNet / FT-Transformer: too large a build for the time left.
- Focal loss: no clear case for a ranking metric.
- Soft pseudo-labelling: it would leak validation labels unless rebuilt per fold.

### Round 7 — second outside review, section 3e — TE smoothing pays

The diagnostics found nothing (§2). Each grouping was checked against the model's own OOF, using a
per-group z of observed vs predicted positives.

Ablation, paired against the submitted set with m = 20 (0.945271):

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| **TE smoothing m = 5** | **+0.000092** | 4.60 | 5/5 | 1.00 | **ADOPT** |
| `+ precision` (income mod 100/1000, last digit, commute tenths) | +0.000023 | 2.31 | 5/5 | 0.80 | noise |
| `subsample` 0.7 | −0.000029 | −0.82 | 1/5 | 0.23 | noise |
| m = 50 | −0.000059 | −2.60 | 1/5 | 0.04 | noise |
| m = 100 | −0.000214 | −9.15 | 0/5 | 0.00 | hurts |

m = 5 sat on the grid edge, so round 7b checked below it:

| m | vs m = 20 | vs m = 5 |
|---|---|---|
| 1 | +0.000086 (t 3.65, 5/5) | −0.000006 |
| 2 | +0.000098 (t 2.98, 4/5) | +0.000006 |
| 3 | +0.000080 (t 2.61, 5/5) | −0.000012 |
| **5** | **+0.000092 (t 4.60, 5/5)** | — |

m = 1–5 is a plateau. m = 5 was adopted: it clears the bar most cleanly and keeps some shrinkage.

The rest of the review was set aside, because the result is fixed by construction:

- WoE is a monotone transform of TE, so trees split on it identically.
- Platt / isotonic calibration before rank averaging does nothing: ranks are unchanged, and isotonic only adds ties.
- Subgroup rank normalisation only moves the cross-group order, which the model already sets.
- Coarser pairwise TE keys were not tried, because the finer one measured at noise.
- A denoising autoencoder is too large a build.

The XGBoost `rank:pairwise` model was interrupted after 17 min. XGBoost's own `auc` under a ranking
objective counts every pair within a query group (O(n²), single-threaded), so each iteration took seconds.
It was switched to a callable sklearn AUC, and **is still unmeasured**.

### Round 8 — retrain with m = 5

| Model | round 3 | first seed (r3 → r8) | round 8 | Δ | folds improved |
|---|---|---|---|---|---|
| LightGBM (3 seeds) | 0.945726 | 0.945677 → 0.945767 (**+0.000090**) | 0.945795 | +0.000069 | 9/10 |
| XGBoost (3 seeds) | 0.945612 | 0.945591 → 0.945704 (+0.000113) | 0.945726 | +0.000114 | — |
| CatBoost (1 seed) | 0.945720 | — | **0.945823** | +0.000103 | 10/10 |
| blend | 0.945795 | | **0.945879** | **+0.000084** | |

- **The screening harness predicted the full run almost exactly:** +0.000092 in the sweep against
  +0.000090 for the first-seed LightGBM.
- The gain carried to all three libraries, unlike round 3's leaf count. It is a feature change, not a
  hyperparameter that has to be translated between libraries.
- Rank-power averaging (5c) on the new blend gave mean Δ −0.0000000, t = −1.92 → closed.
- Public LB 0.94586, +0.00004 over round 3.

---

### Round 9 — per-column smoothing, pairwise XGBoost, CatBoost 3 seeds

A third review (pasted in chat) suggested per-column smoothing, with a *higher* m for the low-support
income column. The evidence runs the other way. m only matters where values have few rows, which in
practice means income (~50 rows per value) and, faintly, commute (~830). The round 7 sweep was therefore
already an income sweep, and less smoothing won. A back-of-envelope empirical-Bayes estimate from the EDA
dispersion ratios gives m ≈ 5–10 for income and ≈ 100 for commute. Section 3f computes it properly and
tests two variants against the submitted m = 5:

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| EB shape: income 5, rest 100 | −0.000031 | −1.17 | 1/5 | 0.11 | noise |
| commute m = 100 only | −0.000033 | −2.16 | 1/5 | 0.08 | noise |

**Closed: one m = 5 beats per-column smoothing.** Commute also prefers little smoothing, despite the
empirical-Bayes estimate. The baseline reproduced round 7's m = 5 score exactly (0.945363), days later.

The computed empirical-Bayes m (3f-1) was 9.4 for income, 123 for commute, and hundreds to thousands for
the rest. Where m is small against the rows per value, the shrinkage is under 2% and cannot matter.

| Also in round 9 | Result |
|---|---|
| CatBoost 3 seeds | 0.945823 → **0.945844** (+0.000021), 54 min |
| Blend with the 3-seed CatBoost | 0.945879 → **0.945884** (+0.000005). Weights 0.40 / 0.02 / 0.59 |
| XGBoost `rank:pairwise` | OOF **0.945733** (as strong as the log-loss XGBoost), rank correlation 0.9962–0.9967 with the trees. Nested weight 0.15 on 9/10 folds, but mean Δ **+0.000003**, t = 2.21, boot 0.93. Not established |
| Rank-power averaging (again) | mean Δ −0.0000000, t = −3.07. Nothing |

- **Seed bagging a model that already sits in an average buys little.** CatBoost alone gained +0.000021,
  but the blend was already averaging most of that seed noise away.
- **A different objective is not a different model here.** The pairwise ranker is exactly as accurate as
  the log-loss XGBoost, and it makes the same errors. On this data the diversity has to come from the
  features, not from the loss.
- The first pairwise attempt crashed: XGBRanker evaluates a callable metric per query group in a thread
  pool sized by `n_jobs`, and `n_jobs=-1` raises `max_workers must be greater than 0`. The fix was to pass
  `n_jobs=os.cpu_count()`.
- Predictions: `preds_new/` now holds the 3-seed CatBoost (it replaced the round 8 single-seed file) and
  `xgb_pw`. `submissions/submission.csv` is the round 9 blend.

## 6. Closed directions

| Direction | Evidence |
|---|---|
| Hand-built features, TE of all categoricals, pairwise TE, full-row TE | all noise (rounds 1, 6) |
| Per-column / empirical-Bayes TE smoothing | income 5 + rest 100 and commute 100: both −0.00003, noise (round 9) |
| Anything derived from the original dataset | per-value TE −0.0001, model on the original −0.00017, tree / flag noise (rounds 2, 6) |
| Tree capacity beyond 15 leaves / depth 5 | 7 leaves and CatBoost depth 4 not established (round 4) |
| `colsample_bytree`, `reg_lambda`, `subsample` | noise (rounds 3, 4, 7) |
| Blend diversity | NN (corr 0.993, 0.0016 weaker) and `extra_trees` (corr 0.9987): weight 0 on every fold (rounds 5, 6); pairwise-ranking XGBoost +0.000003 (round 9) |
| Blend post-processing | rank power: nothing; calibration / subgroup ranks: zero by construction (round 7) |
| Generator artifacts | no duplicates, no row overlap, no `id` signal, digit patterns explained by the model (rounds 6, 7) |

## 7. Open items

None. Every item raised during the competition has been measured. The pipeline is saturated: feature,
capacity, encoding, bagging and diversity changes now move the blend by ≤ 0.00001.

## 8. Running the notebook

```
notebook/     ev_purchase_prediction.ipynb   — Run All reproduces round 8
preds/        round 3 model predictions (m = 20)
preds_new/    round 8 configuration (m = 5): lgb, xgb, 3-seed cat, xgb_pw
submissions/  submission.csv                 — written by the last cell
```

- Every finished experiment is behind a flag that is now off (`RUN_ABLATION`, `RUN_TUNE`,
  `RUN_R6_ABLATION`, `RUN_R7_ABLATION`, `RUN_CAT_D4`, `RUN_NN`, `RUN_LGB_ET`, `RUN_XGB_PAIRWISE`). Its result
  is written in the cell's comments.
- Adoption flags: `USE_GROUPS` (3b), `TUNE_ADOPT` (3c), `ADOPT_R6` (3d-4), and `ADOPT_R7_*` (3e-4). When any
  of them changes the configuration, `CONFIG_CHANGED` switches `REUSE_SAVED_TREES` off. The trees are then
  retrained and written to `preds_new/`. With the configuration unchanged, the trees are reloaded from disk
  in seconds.
- The diversity candidates (5b) and rank-power (5c) write their own file only if they clear the bar.
  `submission.csv` always holds the main blend.

## 9. Transferable lessons (new since S6E8)

- **Tune the encoding, not only the model.** TE smoothing was set once, at m = 20, and never questioned until an
  outside review asked. With hundreds of rows behind most values, less shrinkage was worth as much as the
  entire capacity sweep.
- **Feature changes carry across libraries; hyperparameters do not.** A LightGBM leaf count did not
  translate into XGBoost / CatBoost depth, while the TE change improved all three.
- **A cheap harness can predict the expensive run.** Five folds at lr 0.08 predicted the 10-fold lr 0.03
  gain to within 0.000002 (round 8), and to the right order of magnitude in round 3.
- **Check the grid edge.** Both adopted settings (15 leaves, m = 5) sat on the edge of their first grid.
  Both edge checks found a plateau.
- **Many review suggestions can be answered without running anything.** WoE and calibration are monotone
  transforms, and subgroup ranking cannot beat the model's own cross-group order. Measure the rest; one
  paid.
- **Diversity has to come from the features, not the loss.** A pairwise-ranking XGBoost matched the
  log-loss one in accuracy and still correlated 0.997 with it. Seed bagging a model that already sits in
  a blend gains a quarter of what it gains alone (+0.000005 vs +0.000021).
- **The original dataset can be noisier than the synthetic one.** If a model trained on the original
  scores higher on the competition data than on the original's own held-out rows, original-derived
  features will only add noise.
- **Environment traps on this machine:**
  - torch + LightGBM / XGBoost in one kernel can deadlock (multiple OpenMP copies);
  - pandas 3 `.map()` on string Series returns Arrow arrays (use `.to_numpy(dtype=...)`);
  - XGBoost's AUC under a ranking objective is O(n²) per query group;
  - XGBRanker with a callable `eval_metric` needs a positive `n_jobs` (−1 crashes its per-group thread pool).
