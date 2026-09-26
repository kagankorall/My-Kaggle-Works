# Playground Series S6E8 — Predicting Smartphone Addiction

Working log for the competition. Records what was tried, what it measured, and what it cost —
including the things that did not work and the method mistakes that cost a round each.

**Current best: LB 0.96987** — from a submission the log failed to record (see §1). The best
*attributable* score is **0.96986** (round 9). For reference, a strong public solution scored 0.96891.

- Task: binary classification of `addicted_label`, scored with **ROC-AUC**
- Data: 691,369 train / 296,302 test rows, 9 numerical + 3 categorical features
- Positive rate: 70.94%
- Notebook: [`notebook/smartphone_addiction_prediction.ipynb`](notebook/smartphone_addiction_prediction.ipynb)

---

## 1. Leaderboard history

| Round | Change | OOF AUC | LB | Δ LB |
|---|---|---|---|---|
| 1 | Baseline: 5-fold, LGB/XGB/CatBoost, 3000-iteration cap — XGBoost | 0.964864 | 0.96636 | — |
| 1 | Baseline blend (0.69 xgb / 0.18 lgb / 0.13 cat) | 0.964974 | 0.96632 | −0.00004 |
| 2 | Raised iteration ceilings, added rank-blending option | — | 0.96647 | +0.00015 |
| 3 | Constraint imputation + exact-value target encoding + 10-fold | 0.968456 | 0.96963 | **+0.00316** |
| 4 | `TE_SKIP` + pairwise TE, shipped together | — | 0.96743 | **−0.00220** |
| 5–6 | CV-only ablation rounds, no submissions spent | — | — | — |
| 7 | + frequency encoding | — | 0.96984 | +0.00021 |
| — | **Unattributed** — a submission the log never recorded | — | **0.96987** | +0.00003 |
| 8 | Context-conditional TE / jagged categoricals — all rejected on CV | — | not submitted | — |
| 9 | + `lgb_raw` / `lgb_enc` diversity models | 0.968727 | 0.96986 | +0.00002 |

Round 3 produced more than every other round combined. Round 4 was a regression and was rolled back.
Rounds 5, 6 and 8 spent no submissions — the questions were settled on cross-validation.

**One submission is unattributed.** It scored 0.96987, one day after round 7 and above it, and no record
was kept of what it changed. It is the best score on the board and nothing in this log explains it. That
is the failure mode this whole document exists to prevent, so it is recorded as a gap rather than quietly
folded into round 7.

**CV/LB agreement.** The offset has been stable everywhere both numbers were recorded:

| Round | blend OOF | LB | offset |
|---|---|---|---|
| 3 | 0.968456 | 0.96963 | +0.00117 |
| 9 | 0.968727 | 0.96986 | +0.00113 |

At round 7 the paired ablation predicted +0.000224 and the leaderboard delivered +0.00021. The harness now
predicts leaderboard movement in both direction and magnitude — **including predicting nothing**, which is
what rounds 8 and 9 measured and what the leaderboard confirmed. That is worth more than any single
round's gain: ideas can be screened without spending submissions.

Working backwards through the offset puts round 7's blend at roughly 0.96869 OOF, which makes round 9's
five-model blend about +0.00004 better — the same figure the blend-versus-best-single comparison gives
from the other direction (§6).

---

## 2. What the data turned out to be

**Heavy missingness, MCAR.** Every column except `id` has gaps, up to 19.4% (`social_media_hours`).
The rates differ between train and test, so the mask is drawn per split. Missingness carries no target
signal — the largest gap between `P(y|NA)` and `P(y|not NA)` is 0.004 against a 0.709 base rate.

**A hard arithmetic identity**, verified on 100% of complete rows:

```
daily_screen_time = social_media + gaming + work_study + other,   other >= 0
```

**Several features relate to the target non-monotonically.** This turned out to be the single most
important property of the dataset, and it was missed in the first pass — see section 3.

**A soft reference point, not a ceiling.** A LightGBM trained only on complete rows scores ~0.9698,
which is roughly where the full pipeline sits. This was read as a ceiling; it is not one. Complete rows
are **44.3%** of train (306k of 691k), so that model is missing over half its data — its score is neither
an upper nor a lower bound on what the full pipeline can reach.

---

## 3. The finding that mattered

The EDA scored every numerical column by single-feature AUC and concluded that `notifications_per_day`
(0.492), `app_opens_per_day` (0.541), `sleep_hours` (0.529) and `age` (0.503) were at chance level.

That conclusion was wrong, and the decile plot in the same section already disproved it:

```
app_opens_per_day, target rate by decile:
0.655, 0.734, 0.673, 0.645, 0.726, 0.716, 0.657, 0.763, 0.770, 0.755
```

Each decile holds ~61,000 rows, so the standard error of a rate is about 0.0019. The swing from 0.645 to
0.770 is roughly **65 standard errors** — a strong relationship that simply is not monotonic.

**Single-feature AUC measures monotonic separation, not information.** A zig-zag relationship scores 0.50
while being highly predictive.

How extreme the jaggedness is: on `notifications_per_day` the per-value target rate runs from 0.372 to
0.932 with ~2,200 rows behind each value, i.e. a standard error of about 0.010 per value. The label
depends on the exact integer the way it would on a lookup table, not on a numeric trend. Summing the
exact-value logit offsets of the four "chance-level" columns and nothing else already gives AUC **0.830**.

Exact-value target encoding is the instrument that extracts it. Each numerical column is treated as a
high-cardinality categorical and replaced by `P(y=1 | that exact value)`, estimated out-of-fold:

| Feature | raw AUC | TE AUC | Δ |
|---|---|---|---|
| `notifications_per_day` | 0.4921 | **0.7485** | **+0.2564** |
| `app_opens_per_day` | 0.5409 | **0.7349** | **+0.1940** |
| `sleep_hours` | 0.5270 | 0.5914 | +0.0645 |
| `age` | 0.5023 | 0.5500 | +0.0477 |
| `work_study_hours` | 0.6549 | 0.6676 | +0.0127 |
| `gaming_hours` | 0.6220 | 0.6325 | +0.0105 |
| `daily_screen_time_hours` | 0.8896 | 0.8769 | −0.0126 |
| `weekend_screen_time` | 0.8810 | 0.8646 | −0.0164 |
| `social_media_hours` | 0.8578 | 0.8237 | −0.0341 |

**Why it works here.** Values repeat heavily — roughly 475 rows per distinct value — so each per-value
estimate rests on real support. And the data is synthetic: generators reproduce values from a source
dataset, so both the exact value and its rate of occurrence are traces of that source. Frequency encoding
(section 4) is the same insight from a second angle, and it is the only other idea that produced a gain.

---

## 4. Everything that was measured

All figures from the section 3b ablation harness: one LightGBM per feature set over shared folds.

| Idea | Effect | Verdict |
|---|---|---|
| Exact-value target encoding | −0.002532 to remove from the smooth columns | **essential** |
| Frequency encoding | +0.000224, t = 4.95, 5/5 folds | **adopted** → LB +0.00021 |
| Sharper TE smoothing (`te5`) | +0.000064, t = 2.30, 4/5 folds | rejected — fails the bar |
| Two-granularity TE (`te1d`) | −0.000028 | nothing |
| Pairwise TE, quantile-binned — jagged × jagged | −0.000034 | nothing |
| Pairwise TE, exact values — jagged × jagged | −0.000121 | nothing |
| Context-conditional TE, 8 bins (`ctx8`) | +0.000047, t = 2.25, 4/5, boot 0.84 | rejected *(§5b)* |
| Context-conditional TE, 4 bins (`ctx4`) | +0.000024, t = 0.54, 2/5, boot 0.68 | nothing |
| Jagged columns as LightGBM categoricals | **−0.001156**, t = −15.8, 0/5 | **regression** *(§5b)* |
| Constraint-aware imputation (15 features) | −0.000062 to remove | nothing measurable |
| Hand-built ratio features (17 features) | −0.000047 to remove | nothing measurable |

**Target encoding and frequency encoding are the entire story.** Every hand-built domain feature —
composition ratios, waking-hour shares, weekend rhythm, the constraint-derived bounds — measured at zero
once the encodings were present. 36 features score the same as 53.

Features that measured at zero were **kept anyway**. They are not harmful, and moving a known-good
configuration on a noise-level difference is exactly what caused the round 4 regression.

### Directions checked and ruled out

These follow naturally from the generator-fingerprint theme and are dead ends here. Recorded so they do
not get re-tried:

- **Decimal-precision fingerprint.** Every float column is ~90% two-decimal / ~9% one-decimal with no
  target relationship — the AUC of the decimal count runs 0.496–0.507 across all nine columns — and the
  three integer columns are integers throughout.
- **Row-level duplicates / near-duplicate group size.** 987,671 train+test rows produce 987,654 distinct
  9-column combinations. No duplicate structure to count; AUC of the group size is 0.5000.
- **Encoding coverage gaps.** Fewer than 0.01% of test rows carry an exact value never seen in train, in
  every column. There are no thin regions in the encoding tables, which removes the main rationale for
  original-dataset augmentation.

---

## 5. Method mistakes

**Round 4: two changes in one submission.** `TE_SKIP` and pairwise TE shipped together, lost 0.0022, and
neither was attributable. The later ablation showed `TE_SKIP` caused all of it and the pairs none.

The reasoning behind `TE_SKIP` had been that the encodings for `daily_screen_time_hours`,
`weekend_screen_time` and `social_media_hours` scored *below their raw columns* as standalone predictors.
That was never evidence that a model holding both is better off without one: the raw column supplies the
smooth monotone trend, the encoding supplies each exact value's deviation from it, and neither substitutes
for the other. Dropping features from a known-good configuration needs stronger evidence than a
standalone-AUC comparison.

**Rounds 5–6: the ablation threshold was too strict.** Each variant's mean AUC was compared against a
"noise floor" set by the largest fold-to-fold standard deviation. But `fold_std` measures how much harder
one fold is than another, and **every variant sees the same folds**, so that difficulty cancels in a
between-variant comparison. Using it as between-variant noise buried real effects.

Switching to a **paired per-fold test** — take `(variant − baseline)` on each fold, then judge the
consistency of those deltas — turned frequency encoding from "+0.000219, below the bar" into
"t = 4.95, 5/5 folds". Same data, correctly read, and the leaderboard then confirmed it.

**A third one, caught before it cost a round: a negative result is only as broad as the variant that
produced it.** "Pairs measured at zero" was written up as "pairwise encoding is closed" without checking
whether the pairs tested covered the idea. They did not — see section 5b.

**Rules adopted since:**
- One change per submission, proven on CV first.
- Compare variants with a paired test on shared folds; between-fold variance is not between-variant noise.
- Keep the ablation baseline pinned to the configuration actually being submitted.
- Adopt only if a variant wins 5/5 folds, |t| ≥ 2.5, and a paired row bootstrap puts ≥95% of resamples
  above zero.

---

## 5b. The interaction question, reopened and settled

Rounds 4 and 5 tested three pairs — `notif × opens`, `age × notif`, `age × opens` — and all three cross
one **jagged** column with another. That fails for reasons unrelated to whether interactions exist:
231 × 166 = 38k cells at ~18 rows each is not estimable, and both sides carry the same kind of signal
anyway. **Jagged × smooth was never tested.**

A direct diagnostic says there is something there. Split train at the median of `daily_screen_time_hours`
and compute each exact value's logit offset from its own group's base rate. A purely additive fingerprint
would reproduce the same offsets in both halves:

| column | values | corr(low, high) | sd low / high |
|---|---|---|---|
| `notifications_per_day` | 197 | **+0.878** | 0.95 / 1.14 |
| `app_opens_per_day` | 158 | **+0.882** | 0.91 / 1.16 |
| `age` | 18 | +0.857 | 0.20 / 0.22 |
| `sleep_hours` | 313 | +0.655 | 0.33 / 0.48 |

Cells hold ~1000 rows, so sampling noise attenuates these by at most ~1%: the true agreement is ~0.88,
not 1.0. Correlation is scale-invariant, so this is not one shared pattern at two strengths — the pattern
itself moves with the rest of the row.

### What round 8 tested

Three variants, all against the round 7 baseline (0.967865 at 5 folds), no submissions spent:

| Variant | What it adds | mean Δ | t | folds won | boot P(Δ>0) |
|---|---|---|---|---|---|
| `+ ctx8` | exact value × 8 quantile bins of `daily_screen_time`, as deviation from the marginal TE | +0.000047 | 2.25 | 4/5 | 0.84 |
| `+ ctx4` | the same at 4 bins — more support per cell, less resolution | +0.000024 | 0.54 | 2/5 | 0.68 |
| `+ jagged cat` | `age` / `notifications` / `app_opens` as categorical dtype beside their numeric selves | **−0.001156** | −15.8 | 0/5 | 0.00 |

**Nothing was adopted.** Three things came out of it.

**1. The interaction is real in the data, but the model already has it.** The offset diagnostic above
still stands. But a GBDT holding `notifications_per_day_te` and `daily_screen_time_hours` as separate
columns can split on one and then the other, and *that is* the interaction term. An explicit conditional
encoding only pays when the tree cannot find that partition, and evidently it can. **Verified structure in
the data does not imply a gain from an explicit feature for it** — the prediction that motivated this
round was wrong, and wrong for a reason worth keeping.

**2. `+ jagged cat` failed on leakage, not capacity**, and it is the sharpest result of the round.
LightGBM fits its categorical split **in-fold**: at each node it sorts the categories by their gradient
statistics and partitions the sorted set, which on a 231-level column amounts to an in-fold target
statistic. The TE columns already carry that same information out-of-fold, so the second, leaky route to
the same signal costs 0.0012. That is indirect confirmation that the OOF discipline around the encodings
is doing real work.

This condemns LightGBM's mechanism, not CatBoost's ordered target statistics, which exist precisely to
avoid that leak (`max_ctr_complexity=2` is wired into the CatBoost cell and switches on only if
`ADOPT_JAGGED_CAT` is set). That variant cannot be screened in the 3b harness, and it is not worth a
multi-hour run to find out: CatBoost is the weakest of the three models and adds almost nothing to the
blend as it stands.

**3. The harness has reached its resolution limit.** `ctx8`'s fold SE is 0.000021 and its bootstrap SE is
~0.00004 — the effect sits at the edge of both instruments, which is also why the two disagree (t = 2.25
reads "marginal", boot = 0.84 reads "probably"). Ten folds would shrink the fold SE by only √2, for 80
minutes. An effect this size cannot be established at any affordable cost, which settles the question
rather than leaving it open.

**The encoding axis is closed for real now** — not by assumption, but by having tested the family rounds 4
and 5 skipped. What remains is model-side.

---

## 6. Current pipeline

```
load → EDA → feature engineering → constraint imputation → target + frequency encoding
     → section 3b ablation (decide the feature set)
     → 10-fold CV: LightGBM / XGBoost / CatBoost  + lgb_raw / lgb_enc (diversity)
     → blend candidate comparison → single submission file
```

- **Folds:** `StratifiedKFold(10, shuffle=True, random_state=42)`, created once in section 3 and reused by
  every model. The encodings are built on those same folds; a different split anywhere would let a model
  validate on rows whose encoding saw their own targets.
- **Missing values:** never imputed for the models — LightGBM, XGBoost and CatBoost handle NaN natively.
  CatBoost rejects NaN in categorical columns, so there missingness is encoded as its own level.
- **Features (62, the round 7 set):** 9 raw numerical + 3 categorical + 17 engineered + 15 constraint +
  9 target-encoded + 9 frequency-encoded. Section 3b adds to this only via `ADOPT_CTX` /
  `ADOPT_JAGGED_CAT`, which stay off until the ablation clears them.
- **Round 3 model detail:** LightGBM 0.968094, XGBoost 0.968265, CatBoost 0.968071, blend 0.968456 at
  weights 0.22 / 0.44 / 0.34.

### Round 9: the ensemble cannot be fixed, and now we know why

OOF correlations between the three main models are **0.9961–0.9976**. They make the same mistakes, so
the blend had never beaten the best single model by more than 0.0002. Round 9 tested whether two
**feature-set split** models could break that — `lgb_raw` (44 features: raw, engineered and constraint
columns, encodings removed) and `lgb_enc` (21: target and frequency encodings only).

| Model | OOF AUC | vs lgb | vs xgb | vs cat |
|---|---|---|---|---|
| `lgb` | 0.968324 | — | 0.9975 | 0.9960 |
| `xgb` | **0.968487** | 0.9975 | — | 0.9963 |
| `cat` | 0.968258 | 0.9960 | 0.9963 | — |
| `lgb_raw` | 0.964924 | **0.9816** | **0.9820** | **0.9821** |
| `lgb_enc` | 0.963759 | **0.9800** | **0.9799** | **0.9801** |

**The decorrelation worked exactly as designed.** Both landed at 0.980–0.982 against the main three and
0.9647 against each other, while the main three did not move. The blend used them rather than zeroing
them out — simplex weights 0.094 for `lgb_raw` and 0.015 for `lgb_enc`.

**It bought nothing.** The blend reached 0.968727 against a best single model of 0.968487, an edge of
+0.000240 — against +0.000191 for the three-model blend in round 3. Two genuinely decorrelated models
moved the blend's edge by +0.00005, and the leaderboard delivered +0.00002.

The reason is structural, and it closes the direction:

> The encodings **are** the signal (+0.0025 in the 3b ablation). Any model that holds them correlates
> ~0.996 with the others; any model that drops them is ~0.004 weaker. There is no middle.

Decorrelation is necessary for a blend to gain, but it is not sufficient — the diversity has to be cheap,
and in this dataset it cannot be. At ρ = 0.98 only ~2% of a model's variance is its own, and here that 2%
arrives attached to a 0.0036–0.0047 accuracy deficit.

**The meta-learner closed along with it.** Ridge coefficients came out all-positive and essentially
identical to the simplex solution:

```
ridge      lgb 0.1905  xgb 0.3855  cat 0.3033  lgb_raw 0.1092  lgb_enc 0.0162
simplex    lgb 0.1968  xgb 0.4063  cat 0.2877  lgb_raw 0.0937  lgb_enc 0.0154
```

The entire argument for Ridge over simplex weights was that negative coefficients let the combiner cancel
shared error instead of only averaging. With no opposing error to cancel, it rediscovered the convex
combination — and `ridge_prob` (0.968716) scored *below* `prob_optimized` (0.968722).

The top four candidates sit within 0.00001 of each other (`rank_optimized` 0.968727, `prob_optimized`
0.968722, `ridge_prob` 0.968716, `ridge_rank` 0.968632). Which one the cell selects is not meaningful.

Both diversity models are **kept** despite the near-zero gain, by the same rule that keeps the zero-effect
features: moving a known-good configuration on a noise-level difference is what caused round 4. They cost
~20 minutes; `RUN_DIVERSITY = False` skips them while iterating on something else.

`RUN_LGB_ET` stays off. It was never measured, so this is a judgement call rather than a recorded
rejection — it varies the algorithm while holding the features fixed, and round 9 showed that even a much
sharper cut along the feature axis buys nothing.

---

## 7. Open items, ranked

Feature engineering closed in round 8 (§5b). Stacking closed in round 9 (§6). Two items remain live and
both are worth roughly +0.0002.

1. **Targeted hyperparameter sweep before opening Optuna.** The TE columns are jagged, so `num_leaves=64`
   with `min_child_samples=60` may be clipping their resolution. Sweep `num_leaves ∈ {64, 128, 256}` ×
   `min_child_samples ∈ {20, 60, 150}` at 3 folds first, and open a full Optuna run on XGBoost only if
   that axis shows signal. XGBoost is where any gain should be spent: it is both the best single model
   (0.968487) and the largest blend weight (0.40), so port the direction there via `max_depth` /
   `min_child_weight` once LightGBM says whether the axis is live at all.
2. **Multi-seed bagging.** Mechanical, reliable, roughly +0.0002, costs runtime linearly.
3. **Nested target encoding.** The encoding and the CV share folds, so training rows carry encodings
   fitted on tables that include the validation fold. The effect is diluted and CV/LB have agreed at
   every round, so this is a correctness cleanup rather than an expected gain.
4. **Original dataset augmentation** — demoted. The usual rationale is extra support for the encoding
   tables in thin regions, and the coverage check in §4 found no thin regions: under 0.01% of test
   rows carry a value unseen in train.

**Closed, with the evidence:** diversity for the blend and meta-learner stacking, both in §6. The public
solution that suggested them stacks many diverse models and fits a single Ridge on top of pre-computed OOF
matrices. That works where the base models are diverse *and* comparable in strength; here they cannot be
both at once, and the Ridge reduced to the simplex weights we already had.

The one door left open on that axis is a model with a genuinely different functional form that still holds
the full feature set — a neural net with embeddings over the exact-value columns would be the candidate,
strong *and* decorrelated. At this plateau it is a large build for an uncertain return.

---

## 8. Running it

```
notebook/  smartphone_addiction_prediction.ipynb   — run cells in order
data/      train.csv, test.csv, sample_submission.csv
submissions/  submission.csv                       — single file, written by the last cell
```

Paths are relative to the notebook directory. Two long stages:

1. **Section 3b ablation** (~40 min at 5 folds × 4 variants, plus ~30 s for the row bootstrap).
   Currently `RUN_ABLATION = False`; the round 8 result is recorded in the cell comment. Set it to `True`
   to screen a new feature set, then set `ADOPT_CTX` / `ADOPT_JAGGED_CAT` at the bottom of that cell.
   Both are off, which reproduces the round 7 configuration exactly.
2. **Section 4 training.** The "3–4 hours" figure carried in earlier revisions is **unverified and
   probably stale** — at 10 folds LightGBM measured 10 min 45 s wall and the two diversity models 20 min
   together on the round 9 pass. Re-measure the total on the next full run and replace this note.
   `RUN_CATBOOST`, `RUN_DIVERSITY` and `RUN_LGB_ET` each gate a stage. If `ADOPT_JAGGED_CAT = True`, CatBoost switches on
   combination CTRs and gets noticeably slower — check the per-fold timings before committing to all 10
   folds.

Section 3's encoding cell only builds the two families that are actually shipped. The four measured and
rejected ones (`te5`, `te1d`, `ctx4`/`ctx8`, jagged-categorical) plus both pairwise families sit behind
`BUILD_REJECTED = False` — 32 out-of-fold encodings over 691k rows that nothing downstream reads. Set it
to `True` before running a new section 3b ablation; the ablation cell asserts on it.

The blend cell scores every candidate on OOF AUC and picks the winner automatically; the submission cell
writes only that one file.

---

## 9. Transferable lessons

- On synthetic tabular data, check whether numerical columns behave like high-cardinality categoricals
  before treating them as continuous. Exact values and their frequencies carry the generator's fingerprint.
- Read the binned response curve, not the correlation or the single-feature AUC. Both are blind to
  non-monotonic structure, and that structure was worth +0.003 here.
- Compare model variants with a paired test on shared folds, and set the acceptance bar from the paired
  standard error rather than from between-fold spread. Five folds is a thin sample at the +0.0002 scale;
  a paired bootstrap over the OOF rows costs seconds and measures the uncertainty that actually decides
  whether a CV gain survives on the test set.
- One change per submission. Two changes cost a round and taught nothing.
- "This feature is a weaker standalone predictor" does not imply "the model is better off without it".
- A negative result is only as broad as the variant that produced it. "Pairs measured at zero" became
  "pairwise encoding is closed" while an entire family — jagged × smooth — had never been tried. Testing
  it reached the same verdict, which is the only way that verdict was ever going to be worth relying on.
- Structure verified in the data is not the same as a gain from a feature encoding it. Trees build
  interactions natively; an explicit interaction feature only pays where they cannot.
- Decorrelation is necessary for a blend to gain, but it is not sufficient: the diversity has to be cheap.
  A model 0.98-correlated contributes ~2% of its own variance, which cannot pay for a 0.004 accuracy
  deficit. Check what the diversity *costs* before building it, not only whether it works.
- A meta-learner only beats optimised simplex weights when there is opposing error to cancel. All-positive
  Ridge coefficients that match the simplex solution are the signal that there is not.
- In-fold categorical splits on a high-cardinality column are an in-fold target statistic. Where a
  properly out-of-fold encoding of the same column already exists, adding them cost 0.0012.
- Know the instrument's resolution. Once an effect sits at the edge of both the fold test and the row
  bootstrap, "run it again with more folds" does not rescue it — that is the signal to stop.
- Keep the ablation baseline pinned to what is actually being submitted, or every t-statistic starts
  referring to a configuration that has already moved on.
