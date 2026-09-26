# Playground Series — Predicting Electric Vehicle Purchases

Working log for the competition. One row per submission, with what changed — no unattributed
submissions (the S6E8 log lost one and could never explain its best score).

- Task: binary classification of `Will_Buy_EV` (Yes/No), scored with **ROC-AUC**
- Data: 668,665 train / 286,571 test rows, 7 numerical + 6 categorical features, no missing values
- Positive rate: 17.46%
- Notebook: [`notebook/ev_purchase_prediction.ipynb`](notebook/ev_purchase_prediction.ipynb)

---

## 1. Leaderboard history

| Round | Change | OOF AUC | LB | Δ LB |
|---|---|---|---|---|
| 1 | Baseline: raw + eng + nested TE + freq, 10-fold LGB/XGB/Cat, `rank_optimized` blend | 0.945741 | 0.94575 | — |
| 2 | `+ orig` / `+ orig_lite` (original-dataset TE) — both rejected on CV, not submitted | — | — | — |
| 3 | `nl15_mcs300` capacity (XGB depth 5, Cat depth 5) + 3-seed LGB/XGB, `rank_optimized` blend | 0.945795 | **0.94582** | +0.00007 |
| 4 | 7-leaf edge check + CatBoost depth 4 — neither established, blend +0.000005, not submitted | 0.945800 | — | — |
| 5 | Neural net with exact-value embeddings — nested blend weight 0 on every fold, not submitted | 0.945795 | — | — |
| 6 | Outside-review ideas (exact matches, original-data model/tree, pairwise TE, `extra_trees`) — none established, not submitted | 0.945795 | — | — |
| 7 | Second review: diagnostics, TE smoothing grid, subsample, digit patterns — **m = 5 adopted** (CV only) | — | — | — |
| 8 | Retrain all three models with TE smoothing m = 5 (was 20) | **0.945879** | **0.94586** | +0.00004 |

**CV/LB agreement:**

| Round | blend OOF | LB | offset |
|---|---|---|---|
| 1 | 0.945741 | 0.94575 | +0.00001 |
| 3 | 0.945795 | 0.94582 | +0.00003 |
| 8 | 0.945879 | 0.94586 | −0.00002 |

OOF predicted +0.000054 for round 3 and the leaderboard gave +0.00007; for round 8 OOF predicted +0.000084
and the leaderboard gave +0.00004. Both are within public-LB noise of the prediction — the public board is a
subset of the 287k test rows, while the OOF comparison is paired over all 669k train rows. The CV is a faithful proxy —
ideas can be screened on OOF without spending submissions.

---

## 2. What the data turned out to be

- **No missing values** in either split (unlike S6E8). `n_missing` is constant zero.
- **No train/test drift:** KS ≤ 0.003 on every numerical column.
- **Two near-deterministic rules:**
  - `Subsidy_Available = No` → P(Yes) = **0.0058** (248,756 rows, 37% of train)
  - `Range_Anxiety_Level = High` → P(Yes) = 0.0014, `Medium` → 0.0417, `Low` → 0.1890
- **`Environmental_Concern_Level`** (5 values) is the strongest single column, AUC 0.8435, and nearly
  monotonic — its per-value dispersion ratio is ~33,000.
- **`Annual_Income_USD`** carries the second-strongest signal (AUC 0.6704), and its exact-value encoding
  adds +0.04 on top. Floors at 30,000 (and commute at 5.0) look like clipping in the source.
- `Age`, the station counts and `Number_of_Cars_Owned` sit at ~0.50 AUC but have dispersion ratios of
  10–60 — non-monotonic signal, the same pattern as S6E8.

---

## 3. Round 1 ablation (5 folds, LightGBM, paired against raw + eng + te + freq = 0.945089)

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| `- te` | **−0.002691** | −28.1 | 0/5 | 0.00 | TE essential |
| `- freq` | **−0.000499** | −18.6 | 0/5 | 0.00 | freq helps |
| `- eng` | +0.000004 | 0.18 | 3/5 | 0.68 | noise, kept |
| `+ combo` | −0.000041 | −1.11 | 1/5 | 0.17 | noise, skipped |
| `+ orig` *(round 2)* | **−0.000104** | −2.78 | 0/5 | 0.00 | **hurts**, rejected |
| `+ orig_lite` *(round 2)* | **−0.000114** | −6.28 | 0/5 | 0.00 | **hurts**, rejected |

Target and frequency encoding again carry the gain; hand-built features measure at zero, as in S6E8.

Note that `ev_readiness` is the **top feature by LightGBM gain** while the whole `eng` group measures
at zero. Gain importance shows where the trees chose to split, not what the model would lose without
the column — the information in `ev_readiness` is recoverable from the raw columns.

---

## 4. Models (round 1)

| Model | OOF AUC | best iter (typical) | wall time |
|---|---|---|---|
| LightGBM | 0.945542 | 280–610 | 2 min |
| XGBoost | 0.945578 | 400–570 | 2 min |
| CatBoost | **0.945681** | 490–840 | 16 min |
| blend `rank_optimized` (0.17 / 0.25 / 0.59) | 0.945741 | | |

OOF correlations are 0.9986–0.9991, so the blend's edge over the best single model is only +0.00006.

---

## 5. Rounds 4–5 and closed directions

**Round 4 (CV only) — both axes closed:**

| Test | vs | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|---|
| LightGBM `nl7_mcs60` (5 folds, lr 0.08) | `nl15_mcs300` | +0.000053 | 1.74 | 4/5 | 0.97 | not established |
| LightGBM `nl7_mcs300` | `nl15_mcs300` | +0.000034 | 1.24 | 3/5 | 0.90 | not established |
| LightGBM `nl7_mcs1000` | `nl15_mcs300` | −0.000001 | −0.04 | 2/5 | 0.55 | nothing |
| LightGBM `nl15_mcs300_col0.5` | `nl15_mcs300` | +0.000004 | 0.11 | 1/5 | 0.66 | nothing |
| CatBoost depth 4 (10 folds) | round 3 depth 5 | −0.000003 | −0.28 | 5/10 | 0.43 | nothing |

- **The capacity curve has flattened.** 127 → 64 → 31 → 15 leaves bought +0.00031 in total; 15 → 7 buys
  at most +0.00005 and does not clear the bar. The grid edge is checked, and 15 leaves stays.
- **CatBoost does not care about depth 4 vs 5.** Its round 3 shortfall against LightGBM is not a
  capacity problem.
- Blending `cat_d4` in beside `cat` gives 0.945800 (+0.000005) — the two correlate 0.9997, so this is
  two CatBoost runs acting as seed bagging. Far below what the leaderboard can resolve; not submitted.
  `submissions/submission.csv` was not rewritten and is still the round 3 file.

**Where this leaves the pipeline:** features (§3), the original dataset, capacity (§5b–c) and blend
diversity (correlations ≥ 0.9989) are all measured and closed. What remains is a model with a genuinely
different functional form (a neural net with embeddings over the exact-value columns) — the same single
open door S6E8 ended on, and a large build for an uncertain return.

**Round 5 — neural net, closed:**

| | OOF AUC | corr with lgb / xgb / cat | nested blend weight | vs round 3 blend |
|---|---|---|---|---|
| NN, exact-value embeddings + 32 numerics, 1 seed | 0.944174 | 0.9931 / 0.9934 / 0.9931 | **0.0 on all 10 folds** | +0.000000 |

The net trained fast (≈9 s per fold on MPS, best epoch 4–7) and is genuinely more different from the
trees than they are from each other (0.993 vs ≥ 0.9989) — but it is 0.0016 weaker, and even a 5% share
lowers the blend on every fold. Same structure as S6E8's diversity models: the decorrelation is real,
it just costs more accuracy than it brings.

The first attempt hung: torch deadlocked on its first CPU op (`torch.randperm`) because the kernel had
three copies of OpenMP loaded (torch's, sklearn's and Homebrew's, the last used by LightGBM / XGBoost).
Fixed with `torch.set_num_threads(1)` (the net runs on MPS anyway) and a numpy permutation; the round 3
trees were reloaded from `preds/` instead of retrained.

**Round 6 — ideas from an outside review** (`friend_suggestion.md`), section 3d. **Nothing established.**

*Exact-match diagnostics:* **zero** full-row duplicates inside train, **zero** test rows that occur in train,
**zero** train or test rows identical to an original row. The generator copies single values (97.9% of
incomes occur in the original) but never whole rows — there is no lookup to exploit.

*Ablation* (5 folds, lr 0.08, baseline = submitted set at 15 leaves, 0.945271):

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| `+ pair_inc_anx` (income × anxiety TE, 31 rows/key) | +0.000015 | 0.32 | 3/5 | 0.70 | noise |
| `+ pair_csa` (concern × subsidy × anxiety TE, 30 keys) | +0.000008 | 0.35 | 2/5 | 0.60 | noise |
| `+ in_orig` (income value exists in original) | −0.000014 | −1.17 | 2/5 | 0.43 | noise |
| `+ row` (full-row TE — every row is unique) | −0.000032 | −2.24 | 1/5 | 0.17 | noise |
| `+ orig_tree` (depth-4 tree fitted on original) | −0.000059 | −1.82 | 1/5 | 0.01 | noise |
| `+ orig_model` (LightGBM trained on original only) | **−0.000171** | −7.46 | 0/5 | 0.00 | **hurts** |

*Diversity:* LightGBM `extra_trees` scored 0.945498 alone, correlated 0.9986–0.9988 with the other trees,
and took nested blend weight 0 on all 10 folds.

Two findings worth keeping:
- **The original's label is noisier than the competition's.** The original-only model scores 0.903 AUC
  on the original's own held-out rows but **0.937** on the competition rows. The generator appears to have
  smoothed the label relationship — which is also why every original-derived feature hurt: it is a
  noisier estimate of a signal the 668k-row models already learn more cleanly.
- **The rule the original's tree recovers** (the yes-leaves): `Environmental_Concern ≥ 4` and
  `Subsidy = Yes` and income > ~67.5k and `Range_Anxiety ≠ Medium`; or concern = 3, subsidy, income > ~134k. (The tree lumps the rare
  `High` level with `Low` only because the ordinal codes are alphabetical — High = 0, Low = 1, Medium = 2.)
  Consistent with the EDA; the boosted models already capture it.

Set aside from the review, with reasons: appending the 10k original rows (1.5% more data, and they would
need their own nested encodings); TabNet / NODE / FT-Transformer (a large build four days out, after an
MLP that was 0.0016 short); focal loss (unclear benefit for a ranking metric); soft pseudo-labelling (test
predictions from all folds would leak validation labels into training unless rebuilt per fold, and
Playground gains from it are usually small).

**Round 7 — second outside review**, section 3e.

*Diagnostics — nothing.* Each grouping was checked against the round 3 LightGBM OOF (per-group z of
observed vs predicted positives). `id` order: AUC 0.5000, chi2/df 0.85. Income digit patterns
(% 100, % 1000, last digit, the 30000 floor): chi2/df ≤ 0.23 — their raw rate gaps (4.4% vs 18.8%) are the
low-income floor, which the model already explains. Commute tenths digit: chi2/df 1.97, max |z| 2.7 —
borderline, and the digit-pattern features then measured at noise.

*Ablation* (5 folds, lr 0.08, baseline = submitted set at m = 20, 0.945271):

| Variant | mean Δ | t | folds won | boot P(>0) | verdict |
|---|---|---|---|---|---|
| **TE smoothing m = 5** | **+0.000092** | 4.60 | 5/5 | 1.00 | **ADOPT** |
| `+ precision` (digit patterns) | +0.000023 | 2.31 | 5/5 | 0.80 | noise |
| `subsample` 0.7 | −0.000029 | −0.82 | 1/5 | 0.23 | noise |
| m = 50 | −0.000059 | −2.60 | 1/5 | 0.04 | noise |
| m = 100 | −0.000214 | −9.15 | 0/5 | 0.00 | hurts |

**The first gain since round 3 — from the review.** Monotone in m down to the smallest value tried, so
round 7b checked below the grid edge:

| m | vs m = 20 | vs m = 5 |
|---|---|---|
| 1 | +0.000086 (t 3.65, 5/5) | −0.000006, noise |
| 2 | +0.000098 (t 2.98, 4/5) | +0.000006, noise |
| 3 | +0.000080 (t 2.61, 5/5) | −0.000012, noise |
| **5** | **+0.000092 (t 4.60, 5/5)** | — |

m = 1–5 is a plateau. **m = 5 adopted**: it clears the bar most cleanly and keeps some shrinkage.

**Round 8 — retrain with m = 5:**

| Model | round 3 | first seed (r3 → r8) | round 8 | Δ | folds improved |
|---|---|---|---|---|---|
| LightGBM (3 seeds) | 0.945726 | 0.945677 → 0.945767 (**+0.000090**) | 0.945795 | +0.000069 | 9/10 |
| XGBoost (3 seeds) | 0.945612 | 0.945591 → 0.945704 (+0.000113) | 0.945726 | +0.000114 | — |
| CatBoost (1 seed) | 0.945720 | — | **0.945823** | +0.000103 | 10/10 |
| blend `rank_optimized` | 0.945795 | | **0.945879** | **+0.000084** | |

- **The sweep predicted the full run almost exactly.** 3e-3 measured +0.000092 for LightGBM at lr 0.08 /
  5 folds; the first-seed LightGBM at lr 0.03 / 10 folds gained +0.000090.
- The gain carried to all three families this time (unlike the leaf-count change in round 3), because it
  is a feature change, not a hyperparameter that needs translating.
- CatBoost is now the strongest single model. The blend weights moved to 0.42 / 0.05 / 0.53
  (XGBoost back from zero).
- Rank-power averaging (5c) on the new blend: mean Δ −0.0000000, t = −1.92, 4/10 folds → closed.
- Public LB **0.94586** (+0.00004 over round 3) — the new best.
- The pairwise XGBoost was switched off for this run and is still unmeasured.

*XGBoost `rank:pairwise`* — first attempt interrupted after 17 min: XGBoost's own `auc` under a ranking
objective counts every pair inside a query group (O(n²), single-threaded for one group), so each
iteration's validation AUC on 67k rows took seconds. Replaced by a callable sklearn AUC.

**Final state (as of round 6 — superseded by rounds 7–8 above):** every axis tried — features, original data, capacity, blend diversity, a different model
family, and the outside review's cheap ideas — is measured and closed. **Round 3 (LB 0.94582, OOF 0.945795) is the final submission.**

**Closed: the original dataset.** The full `orig` group hurt (−0.000104). `orig_lite` kept only the six
encodings with ≥20 original rows per value (Age, cars, both station counts, concern level, `cat_combo`),
dropping income (1.1 rows/value) and commute (10.1), and it hurt **more consistently** (−0.000114,
t = −6.28). So the loss did not come from the noisy columns. The original's per-value rates are a
10k-row estimate of what the 668k-row nested TE already measures far more precisely — a noisier copy of
the same signal, which the trees overfit. Income values are nonetheless reproduced from the original
(97.9% of train values occur there), so the generator copies values; it just does not copy labels any
more usefully than the competition data already does.

## 5b. Round 3 capacity sweep (LightGBM, lr 0.08, 5 folds, paired vs `nl64_mcs60` = 0.945089)

| Setting | mean Δ | t | folds won | boot P(>0) | mean iter | verdict |
|---|---|---|---|---|---|---|
| `nl15_mcs300` | **+0.000181** | 5.20 | 5/5 | 1.00 | 409 | **ADOPT** (best) |
| `nl15_mcs1000` | +0.000159 | 5.84 | 5/5 | 1.00 | 435 | ADOPT |
| `nl15_mcs60` | +0.000155 | 3.58 | 5/5 | 1.00 | 377 | ADOPT |
| `nl31_mcs300` | +0.000145 | 7.03 | 5/5 | 1.00 | 232 | ADOPT |
| `nl31_mcs60` | +0.000097 | 5.72 | 5/5 | 1.00 | 229 | ADOPT |
| `col0.5` | +0.000082 | 3.25 | 4/5 | 0.98 | 145 | — |
| `nl31_mcs1000` | +0.000074 | 1.83 | 4/5 | 0.96 | 220 | — |
| `nl64_mcs300` | −0.000010 | −0.64 | 2/5 | 0.60 | 158 | — |
| `l2_10` | −0.000041 | −1.82 | 1/5 | 0.21 | 160 | — |
| `nl64_mcs1000` | −0.000094 | −4.76 | 0/5 | 0.03 | 141 | — |
| `nl127_mcs60` | −0.000199 | −6.33 | 0/5 | 0.00 | 84 | — |
| `nl127_mcs300` | −0.000209 | −5.16 | 0/5 | 0.00 | 98 | — |
| `nl127_mcs1000` | −0.000307 | −12.72 | 0/5 | 0.00 | 92 | — |

**Leaf count is the whole effect.** Every 15- and 31-leaf setting beats 64, every 127-leaf setting
loses, and `min_child_samples` barely matters at a fixed leaf count. The round 1 models were too deep:
a label built from a few near-rule features (subsidy, anxiety, concern level) plus per-value offsets is
fitted better by many shallow trees than by fewer deep ones — the 15-leaf models also run ~3x more
iterations before early stopping. The top four settings are within ~0.00004 of each other, so which
one wins among them is not meaningful; `nl15_mcs300` is taken as the best mean.

## 5c. Round 3 models (10 folds, lr 0.03)

| Model | round 1 | first seed | 3 seeds | from capacity | from bagging |
|---|---|---|---|---|---|
| LightGBM (15 leaves, mcs 300) | 0.945542 | 0.945677 | **0.945726** | **+0.000135** | +0.000049 |
| XGBoost (depth 5, mcw 100) | 0.945578 | 0.945591 | 0.945612 | +0.000013 | +0.000021 |
| CatBoost (depth 5, 1 seed) | 0.945681 | 0.945720 | — | +0.000039 | — |
| blend `rank_optimized` | 0.945741 | | **0.945795** | | |

- **LightGBM confirmed the sweep.** The 3c harness predicted +0.000181 at lr 0.08 / 5 folds; the full
  10-fold lr 0.03 run gave +0.000135 from the setting alone. Same direction, same order of magnitude.
- **The translation to the other two mostly failed.** XGBoost gained +0.000013 from depth 5 and
  CatBoost +0.000039. Depth 5 means up to 32 leaves per tree — twice LightGBM's 15, the same ratio as
  round 1's 64 ↔ depth 7, so the formula was consistent but it did not transfer the *capacity*.
- **The blend gained only +0.000054** despite LightGBM's +0.000184: the models are now more correlated
  (0.9989–0.9994), and the weights went to LightGBM 0.50 / CatBoost 0.50 / XGBoost **0.00**.
- Bagging is worth +0.00002–0.00005 per model — small, as expected.
