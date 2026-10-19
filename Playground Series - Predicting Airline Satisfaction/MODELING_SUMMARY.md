# Playground Series — Predicting Airline Satisfaction

Working log for the competition: one row per submission and what changed, plus what the data turned out to be,
what moved the score, what was rejected, and where the remaining bottlenecks are.

- Task: binary classification of `satisfaction` (True/False), scored with **ROC-AUC**.
- Data: 699,635 train / 299,844 test rows. 21 features: 4 categorical, 4 continuous (age, flight distance, two
  delays) and 13 service ratings on a 0–5 scale. Synthetic, generated from the *Airline Passenger Satisfaction*
  survey; `Inflight service`, the original's 14th rating, is not included.
- Positive rate: 44.36%.
- Notebooks: [`notebook/airline_satisfaction_prediction.ipynb`](notebook/airline_satisfaction_prediction.ipynb)
  (features, tree models, blend, submission) and [`notebook/realmlp.ipynb`](notebook/realmlp.ipynb) (a RealMLP
  neural network whose predictions the main notebook loads).

**Best public LB: round 13, 0.96187**, a fixed ½ + ½ average of our meta-stack and the public 0.96188 stack (no CV
possible for it). **Best by CV: round 14, nested CV 0.962234** (public LB 0.96184): the round 11 stack with a
LightGBM stacker blended in. Round 11 itself: nested CV 0.962224, public LB 0.96185: a meta-stack of 227 public OOF members plus 17 of
ours that were never published. **Best from our own models alone: round 9, public LB 0.96102** (OOF 0.961496).
Over eleven rounds the public LB went from 0.95851 to 0.96185 (+0.00334). Two steps carried most of it: the exact-value encodings of round 3 (+0.00170) and
the RealMLP of round 8 (+0.00029).

---

## 1. Leaderboard history

| Round | Change | OOF AUC (blend) | LB | Δ LB |
|---|---|---|---|---|
| 1 | Raw features, 5-fold LightGBM + XGBoost, rank blend | 0.959144 | 0.95851 | — |
| 2 | + CatBoost on the raw features | 0.959183 | 0.95859 | +0.00008 |
| 3 | + original-data features (`orig_pred`, `orig_te`) and exact-value encodings (`freq`, `te`) | 0.960780 | 0.96029 | **+0.00170** |
| 4 | + pairwise target encoding (`te_pair`) | 0.960949 | 0.96044 | +0.00015 |
| 5 | 10 folds, learning rates halved, early-stopping patience 200 → 300 | 0.961129 | 0.96057 | +0.00013 |
| 6 | + CatBoost with every raw column also as a categorical (diversity model, 5 folds) | 0.961235 | 0.96068 | +0.00011 |
| 7 | All-categorical CatBoost without our own TE and with `l2_leaf_reg` 10; three outside feature ideas tested and rejected | 0.961290 | 0.96071 | +0.00003 |
| 8 | + RealMLP neural network (separate notebook, 5 folds) | 0.961487 | 0.96100 | **+0.00029** |
| 9 | RealMLP averaged over two network seeds | **0.961496** | **0.96102** | +0.00002 |
| 10 | RealMLP with the engineered features as inputs: +0.000003 in screening, not adopted, not submitted | — | — | — |
| 11 | Meta-stack on Kaggle: LR (C = 0.003) on the logits of 227 public OOF members + our 17 unpublished ones | **0.962224** (nested) | **0.96185** | **+0.00083** |
| 12 | Five new all-categorical CatBoosts (Kaggle GPU) screened in the meta-stack: best +0.000001, not adopted, not submitted | 0.962224 (nested) | — | — |
| 13 | Twin: ½ our meta-stack (now 248 members; golem's 4 new ones change nested CV by −0.000004) + ½ nina2025's `0.96188.csv` (ewtherka's stack), percentile ranks, weight fixed in advance | — (test-only half) | 0.96187 | +0.00002 |
| 14 | Meta-stack with a LightGBM stacker blended in: 0.8 × LR + 0.2 × LightGBM on the same logits (section 2c), 248 members | **0.962234** (nested) | 0.96184 | −0.00001 vs round 11 |
| 15 | Twin of round 14: ½ that stack + ½ nina2025's `0.96188.csv`, percentile ranks | — (test-only half) | 0.96186 | — |

**CV / LB agreement**

| Round | OOF gain | LB gain | OOF − LB offset |
|---|---|---|---|
| 1 | — | — | −0.00063 |
| 2 | +0.00004 | +0.00008 | −0.00059 |
| 3 | +0.00160 | +0.00170 | −0.00049 |
| 4 | +0.00017 | +0.00015 | −0.00051 |
| 5 | +0.00018 | +0.00013 | −0.00056 |
| 6 | +0.00011 | +0.00011 | −0.00056 |
| 7 | +0.00006 | +0.00003 | −0.00058 |
| 8 | +0.00020 | +0.00029 | −0.00049 |
| 9 | +0.00001 | +0.00002 | −0.00048 |
| 11 | (meta-stack, nested CV) | +0.00083 | −0.00037 |
| 14 | +0.00001 (nested, vs round 11) | −0.00001 | −0.00039 |

The public LB has sat 0.0005–0.0006 below OOF in every submission, and every OOF gain has shown up on the
leaderboard at roughly its own size. Gains below about 0.0003 are within the public LB's noise (section 6), so CV
is the decision tool and the public LB only confirms.

---

## 2. What the data turned out to be

- **Clean and drift-free.** `Arrival Delay in Minutes` is the only column with gaps (292 train / 130 test rows,
  0.04%). There are no duplicated rows, and train vs test KS is ≤ 0.006 on every column.
- **0 is not a rating.** On the 0–5 scale, 0 means "not applicable". For the digital services it behaves like a
  *good* rating: wifi 0 → 0.89 satisfied, against 0.39 at a rating of 1 and 0.95 at 5 (a U-shape).
- **`Online boarding` is the strongest single column** (AUC ≈ 0.84), with a threshold rather than a slope:
  0.09–0.13 satisfied for ratings 1–3, then 0.64 at 4 and 0.88 at 5.
- **`Type of Travel` × `Class` is an interaction, not two effects.** Personal travellers are about 10% satisfied
  in every class, Business included. For business travellers class decides: 73.2% in Business, 26.0% in Eco.
- **The delays are almost useless.** More than 75% are 0 and the mean is 1.2 minutes, against about 15 in the
  original survey. Single-feature AUC is 0.49 / 0.48.
- **Label noise sets the ceiling.** A model trained on the original survey scores 0.9952 on its own labels, but
  0.9547 on the synthetic ones, only 0.0043 below a model trained on the synthetic data itself. The generator kept
  the original's decision function and added label noise. That is why everything after round 3 moves in
  ten-thousandths.
- **The generator left value-level traces.** Encoding exact values of `Flight Distance` and `Age` (round 3), and
  of category × number pairs (round 4), carried more signal than any other change. Trees cannot isolate
  thousands of distinct values on their own. goodpjw2008 reads `Flight Distance` as a route ID (3,521 distinct
  values over train + test), which fits this.
- **No copies of original rows.** Only 17 train and 4 test rows match an original row exactly, so a label
  lookup has nothing to work with.

---

## 3. Current pipeline (round 9)

**Features for the tree models (54).**

| Block | Columns | What it is |
|---|---|---|
| raw | 21 | binary categoricals as 0/1, `Class` ordinal (Eco 0, Eco Plus 1, Business 2), numerics as-is, NaN left in place |
| `orig_pred` | 1 | prediction of a LightGBM trained on the original survey only (leakage-free: it never sees a synthetic label) |
| `orig_te` | 21 | per-value satisfaction rate in the original data, for every raw column |
| `freq` | 4 | value counts over train + test for `Age`, `Flight Distance` and both delays |
| `te` | 2 | out-of-fold target encoding of `Age` and `Flight Distance` on the synthetic labels |
| `te_pair` | 5 | the same for `Class` / `Type of Travel` × `Flight Distance` and `Class` / `Type of Travel` / `Customer Type` × `Age` |

`te` and `te_pair` are rebuilt inside every fold, with smoothing 20. Validation, test and original rows are
encoded with the fold's training rows; the training rows themselves get an inner 5-fold out-of-fold encoding.

**Models.** Early stopping on each validation fold with patience 300 (trees); RealMLP keeps its best epoch by
validation AUC.

| Model | Folds | Key settings | OOF AUC | Wall time |
|---|---|---|---|---|
| LightGBM | 10 | lr 0.015, 63 leaves, min_child_samples 50, colsample 0.7, subsample 0.8 | 0.960987 | 6 min |
| XGBoost | 10 | lr 0.015, depth 7, min_child_weight 10, colsample 0.7, subsample 0.8 | 0.960972 | 6 min |
| CatBoost | 10 | lr 0.04, depth 7, the 4 categoricals as `cat_features`, early stopping on Logloss | 0.961022 | 28 min |
| CatBoost, all categorical | 5 | lr 0.06, depth 6, `l2_leaf_reg` 10, a string copy of every raw column as `cat_features`, `max_ctr_complexity` 2, our `te_*` columns dropped | 0.961127 | 55 min |
| RealMLP (`realmlp.ipynb`) | 5 | public recipe: raw columns + a categorical twin of every numeric column, `flat_anneal`, 4 epochs, `n_ens` 4, on the Apple GPU (MPS); average of two network seeds | 0.960948 | 2 × 16 min |

**Blend.** Rank average with weights optimised on OOF AUC (Nelder–Mead): LightGBM 0.11 / XGBoost 0.12 /
CatBoost 0.12 / all-categorical CatBoost 0.27 / RealMLP 0.37 → OOF 0.961496. A logistic-regression stack on logits
ties it (0.961489, scored nested), which also shows that the weight optimisation carries almost no optimism.
Models on different fold counts blend fine: every OOF prediction comes from a model that did not see that row.

**Ablation harness (section 3c).** A fast LightGBM (lr 0.1) on a fixed 5-fold split (`ABL_FOLDS`, which is also
the split goodpjw2008 and `realmlp.ipynb` use). Every variant is compared with the baseline fold by fold (paired),
so fold difficulty cancels out. An addition is adopted if its mean delta is positive with t > 2; a removal only
has to not hurt. Each round's baseline must reproduce the previous round's score exactly, and so far it always has.

---

## 4. What moved the score

| Change | LB gain | Evidence |
|---|---|---|
| Exact-value encodings + original-data features (round 3) | +0.00170 | all four blocks together +0.001702 on the harness, t = 27.98 |
| RealMLP, a different model family (round 8) | +0.00029 | weakest member alone, Spearman 0.967–0.972 against the trees, largest blend weight; blend +0.00020 OOF |
| Pairwise target encoding (round 4) | +0.00015 | +0.000129, t = 4.08, 5/5 folds |
| 10 folds + lower learning rates (round 5) | +0.00013 | every model +0.00018–0.00026 OOF (but see the 10-fold caveat in section 8) |
| All-categorical CatBoost (round 6) | +0.00011 | best single model, least correlated at the time, 0.47 of the blend |
| CatBoost in the blend (round 2) | +0.00008 | weakest model alone, but the least correlated at the time |
| All-categorical CatBoost settings (round 7) | +0.00003 | that model +0.000032 OOF with the features unchanged; blend +0.000055 |

Inside round 3 the blocks overlap: alone they were worth `te` +0.001285, `orig_te` +0.000896, `freq` +0.000636
and `orig_pred` +0.000214 (about +0.0030 in total), but together +0.0017.

---

## 5. Closed directions

| Idea | Result | Why it failed |
|---|---|---|
| Row-wise rating aggregates (means, std, counts of 0 / low / top ratings, group means), delay features, `travel_class` | −0.000311, t = −4.21, 0/5 folds | The signal sits in specific thresholds of specific ratings, not in the overall rating level; the smooth aggregates most likely win greedy splits and dilute the precise ones |
| Original survey appended as training rows | −0.000201, t = −3.33, 0/5 | Covariate shift: the original's delays (KS 0.35) and distances (KS 0.22) look nothing like the synthetic ones |
| Target encoding of the delays | −0.000075 | Over 75% zeros, so there is no value-level structure to encode |
| TE smoothing 5 / 50 (instead of 20) | −0.000028 / −0.000060 | The optimum is flat around 20 (unlike the EV competition, S6E9, where 5 won) |
| Dropping `orig_pred` | −0.000341, t = −5.59 | Kept: it complements the encodings with the smooth, clean-label decision function |
| Exact-match label lookup from the original | 17 matching rows | Nothing to look up |
| Route profile (per-`Flight Distance` mean of every other column) | +0.000055, t = 1.00, 3/5 | The distance's target encodings, its pairs, `freq` and `orig_te` already describe each route |
| Second original-data model (XGBoost depth 8 on all original rows) | −0.000038 | Ranks almost exactly like `orig_pred` (Spearman 0.9887) |
| TE keys `Age` × `Flight Distance` and `Age` × `Class` × `Type of Travel` × `Customer Type` | −0.000028 | The pair is sparse; the four-way key is within reach of the trees given `te_pair` |
| RealMLP with the engineered features as inputs (round 10) | +0.000003 to the blend | Better alone (0.960956), but it now ranks like the trees (Spearman 0.974–0.978, from 0.968–0.973) and loses the difference that made RealMLP useful |

The route profile, the second original-data model and the extra TE keys came from goodpjw2008's notebook, where each
was a real gain on a weaker starting set; so was the engineered-input RealMLP, their best single model.

Not tried, because goodpjw2008 measured them on this data (section 8): target encoding of rating combinations
and crossing `Flight Distance` with every column (both overfit), KNN (0.956), a GLM on target rates, FT-Transformer
and GPT-2 as classifiers (all weak).

---

## 6. Bottlenecks

1. **The feature side is exhausted.** Round 7 took the three feature ideas behind the strongest public notebook
   and none passed: our encodings already carry what they add.
2. **Diversity was the lever, and RealMLP pulled it.** The four trees correlate at 0.982–0.995; RealMLP at
   0.967–0.972 against them, and it took the largest blend weight. The blend is now +0.00036 above the best single
   model (it was +0.00016). goodpjw2008's 29-member stack is still about +0.0004–0.0005 above its best single model
   and reaches LB 0.96129 against our 0.96100, so more members of other families (RealMLP variants, DeepFM) remain
   the direction.
3. **Gains near the public LB's resolution.** thisray's resampling estimate: the public LB cannot order gaps
   below about 0.0003, while 700k OOF rows resolve about 0.00005. Round 8's +0.00029 is at that edge; rounds 4–7
   were below it. Decisions have to be made on paired CV.
4. **Runtime is dominated by CatBoost.** The all-categorical CatBoost takes 55 minutes on 5 folds and the regular
   one 28 on 10, against 6 each for LightGBM and XGBoost and 16 for RealMLP. A full retrain of everything is about
   two hours. The ablation harness uses LightGBM and cannot screen CatBoost- or network-specific ideas.
5. **Memory on a 16 GB Mac.** CPU and GPU share it. RealMLP with the recipe's `n_ens=8` ran out of GPU memory on
   the full data; `n_ens=4` runs at 3.3 GB. Close the main notebook's kernel before running `realmlp.ipynb`.
6. **Fold encodings are recomputed for every model.** `fold_te` rebuilds the `te` / `te_pair` encodings
   separately for each model and fold. The cost has not been measured; it is small next to CatBoost.

---

## 7. Open items

- **Auxiliary-task features** (`aux_ev`): predict each rating from the other 20 columns and use the expected
  value. About +0.0001 per tree model in goodpjw2008's measurements, but their other feature ideas did not
  transfer here, so expectations are modest.
- **Does 10-fold help on test?** goodpjw2008 measured +0.00012 in CV but no test gain; our round 5 LB gain is
  below the public LB's resolution.
- **Final selection** (two submissions, ticked by hand: unticked, Kaggle takes the two best public scores):
  round 14 (`submission.csv` of meta-stack Version 6, the highest nested CV) and round 13 (the Version 5 twin,
  public 0.96187). The round 13 twin is chosen over round 15 because it differs more from round 14 (plain LR half,
  no LightGBM), so the pair covers more ground on the private board.
- **Kaggle notebook** (`notebook/kaggle_s6e10_airline_satisfaction.ipynb`): Version 1 committed on Kaggle (GPU T4 x2),
  public LB **0.96091**. All five models on one 5-fold split, CatBoost on GPU, RealMLP with `n_ens` 8 and one seed;
  0.00011 below the local pipeline, within the public LB's noise. Open: make it public, post a discussion topic.

---

## 8. Outside notebooks reviewed

**goodpjw2008, *S6E10 | Auxiliary-Task Features | LB 0.96129* (Part 4 of a series).** A careful, CV-driven
pipeline on the same 5-fold split as our ablation harness. What it does that we did not, and what came of it here:

| Idea | Their measurement | Here |
|---|---|---|
| Route profile: per-`Flight Distance` mean of every other column, no target | alone ≈ target encoding of the distance; with TE + counts +0.00034 (fold 0) | +0.000055, t = 1.00, rejected |
| Original-data XGBoost (depth 8) on all original rows | 0.95514 on the synthetic labels; +0.0008 next to route features | 0.955163, but −0.000038 as a block, rejected |
| TE keys `Flight Distance` × `Age` and `Age` × `Class` × `Type of Travel` × `Customer Type` | part of their main feature set | −0.000028, rejected |
| All-categorical CatBoost without own TE, `l2_leaf_reg` 10 | fold 0: 0.961321, against 0.961173 with own TE and `l2` 3 | +0.000032 for that model, adopted |
| Logistic regression on logits as the stacker | 29 members, negative weights allowed | ties our weighted rank (0.961483 nested) |
| RealMLP, raw inputs + categorical twins, `flat_anneal` | 0.96094–0.96098 with `n_ens` 8; their stack's main diversity | 0.960883 with `n_ens` 4; blend +0.00020 OOF, LB +0.00029, adopted |
| Auxiliary-task features: each rating predicted from the other 20 columns, expected value | +0.00013 for one XGBoost, +0.00006 in the stack | not tried yet |
| RealMLP with engineered inputs ("v4"), DeepFM, a second feature view | v4 is their best single model; DeepFM and view B add a little to the stack | not tried yet |
| 10 folds instead of 5 | +0.00012 in CV, no test gain | open question |

**Notebooks whose score is a blend of public submissions.** The "Vanilla CatBoost" notebook (LB 0.96165)
submitted `1.001 × najiama − 0.001 × vanilla`; `ps6e10-airline-satisfaction-baseline` rank-blends ten public
submission files and gives its own model a weight of 0.0018. Their own models use practices we measured as
harmful (original rows in the folds, rating aggregates). Nothing to adopt.

---

## 9. Running the notebooks

- **Order:** `realmlp.ipynb` first if its predictions are missing (they are saved in `preds/`), then the main
  notebook. Close the main notebook's kernel while RealMLP trains: the two share 16 GB.
- **`realmlp.ipynb`:** `SMOKE = True` runs one epoch on the full fold-0 training set (about a minute) to check
  memory and time; `SMOKE = False` is the full run. Each fold is saved as it finishes, and a re-run resumes at the
  first missing fold.
- **Main notebook: Restart Kernel → Run All.** `RUN_ABLATION = False` applies the recorded round 7 decision (the
  round 4 configuration). `EXTERNAL_PREDS` lists the predictions loaded from other notebooks.
- **`REUSE_SAVED_PREDS = True` is safe to leave on.** Prediction files carry the configuration, the fold count and
  a hash of the model parameters (`lgb_<config>_k10_<hash>_oof.npy`), so any change retrains, and an unchanged
  model loads in seconds. With nothing changed, a full Run All of the main notebook takes about 5 minutes.
- **Typical runtimes when training:**

  | Step | Time |
  |---|---|
  | EDA + original-data models | ~4 min |
  | Ablation (when on) | ~2–5 min |
  | LightGBM | 6 min |
  | XGBoost | 6 min |
  | CatBoost | 28 min |
  | All-categorical CatBoost (5 folds) | 55 min |
  | RealMLP (5 folds, `realmlp.ipynb`) | 16 min |

- **Save with Cmd+S before reading results from the file.** Outputs exist only in the editor until then.
- Outputs go to `preds/` (OOF and test predictions per model, plus LightGBM importances from round 7 on) and
  `submission/submission.csv`. Only the current submission is kept there.

---

## 10. Lessons

- **Paired CV on fixed folds has been reliable.** Nine out of nine submissions moved the LB by about the
  OOF-predicted amount, with a stable offset.
- **A diversity model must stay different.** Feeding RealMLP the trees' engineered features made it a better model
  and a worse blend member: the correlation rose and its contribution vanished.
- **When the features are exhausted, change the model family.** Four rounds of tree-side work added about
  +0.0001 each; one neural network that disagreed with the trees added +0.0003, while being the weakest model
  alone.
- **A weaker-looking but different model is what lifts a converged blend.** Both the all-categorical CatBoost and
  RealMLP won on correlation, not on their own AUC.
- **The original dataset is worth more as features than as rows** when its covariates have shifted. It was the
  reverse in the EV competition (S6E9), where original-data TE hurt. This has to be checked per competition.
- **Someone else's gain depends on their baseline.** goodpjw2008's three feature ideas were real gains on their
  feature set and nothing on ours. Outside ideas go through the same ablation as our own.
- **On Apple silicon, size the smoke run like the real one.** A 60k-row smoke run of RealMLP passed and the full
  run then ran out of unified memory; one full-size epoch is the test that predicts the real run.
- **Gain importance is not contribution.** `orig_pred` dominates the gain chart but is worth +0.0002 alone; only
  the ablation shows what a feature adds.
- **Plot integer columns with one bin per value.** 60 equal-width bins over `Age` drew regular false spikes.
- **A public notebook's LB score is not evidence of its method.** Two of the high-scoring notebooks reviewed were
  blends of other people's submission files. Read the last cells before reading the first.

---

## 11. The meta-stack (round 11)

The top of the public LB (0.96180 – 0.96184) is not built from better single models: it is one logistic regression
over the community's published out-of-fold predictions, often blended at the end with another public submission
file. Our version, `notebook/kaggle_s6e10_meta_stack.ipynb`, runs on Kaggle (CPU, about 4 minutes):

- **Pool:** 16 sources, 245 members; one exact copy dropped, 244 used. The 227 public members include the
  tabular foundation models (TabPFN-3.5, TabFM, LimiX-2, …) that need a CUDA GPU to train.
- **Our 17:** every model from rounds 1–10 that had not been published, uploaded as the private dataset
  `s6e10-ablation-members-oof` (folder `PythonWorks/kaggle_uploads/`, outside the repo).
- **Stacker:** `LogisticRegression(C=0.003)` on logits, scored with nested 5-fold CV on a different seed (7). No
  submission files are blended in.

| stack | members | nested CV |
|---|---|---|
| public pool | 227 | 0.962205 (equal to goodpjw2008's 228-member stack: the loader is right) |
| **public pool + our 17** | **244** | **0.962224** (+0.000020, 5 of 5 folds) → **public LB 0.96185** |
| our 17 alone (C = 1) | 17 | 0.961540 |

- **Our edge is the unpublished members.** Once published, a member is in everyone's pool; our 17 add as much as
  goodpjw2008's own unpublished members did (+0.000022).
- **The two all-categorical CatBoosts carry it:** `cat_allcat_v1` has the 5th-largest absolute weight of all 244
  members, `cat_allcat_v2` is in the top 15. The weaker of the two gets the larger weight — diversity again.
- **The foundation models get the largest weights** (TabFM 0.225, 15-fold TabPFN-3.5 0.206, LimiX-2 0.176). A
  CUDA GPU (Kaggle) TabPFN-3.5 member of our own would be the next unpublished edge.
- The 124-member collection contains one blend-level OOF (najiama's blend, 0.961744); it is kept, as in the other
  public stacks on that collection.

### 11.1 Round 12: more all-categorical CatBoosts (not adopted)

The two all-categorical CatBoosts carry our 17, so five more variants were trained on Kaggle's GPU in
`notebook/kaggle_s6e10_allcat_variants.ipynb` (about 1.5 hours; on Kaggle the notebook kept its default name,
`notebook775e4113b6`, so the meta-stack finds it by its files, not its name) and screened one at a time on top of the
244-member stack (section 2b of the meta-stack notebook):

| variant | depth | `l2_leaf_reg` | `max_ctr_complexity` | inputs (plus a string twin of every raw column) |
|---|---|---|---|---|
| `d8_l10` | 8 | 10 | 2 | engineered features, own TE dropped |
| `d6_l10_c3` | 6 | 10 | 3 | engineered features, own TE dropped |
| `d7_l30_te` | 7 | 30 | 2 | engineered features, own TE kept |
| `d6_l3_te_s7` | 6 | 3 | 2 | engineered features, own TE kept, seed 7 |
| `d6_l10_raw` | 6 | 10 | 2 | the 21 raw columns only |

- Only `d6_l10_raw` passed the gate then in place (any rise, 4 of 5 folds up): nested CV 0.962224, **+0.000001**,
  stack weight −0.002. The other four did not raise the nested CV on at least 4 folds.
- **Not adopted, not submitted.** +0.000001 is noise and the public LB shows five decimals. The gate now asks for at
  least `MIN_GAIN` = 0.000005 (a quarter of what the 17 add), so the notebook reproduces the round 11 stack.
- **Why it failed:** `cat_allcat_v1` and `v2` (weights 0.105 and 0.077) already cover what an all-categorical
  CatBoost sees; tree depth, regularisation and seed changes give the stacker nothing new. A new member has to be a
  different kind of model, not a variant of one already in the pool.
- **Next edge:** the largest weights are still the GPU foundation models (TabFM 0.223, TabPFN-3.5 0.199, LimiX-2
  0.172), all trained by others on the raw columns. A TabPFN-3.5 member of our own, fed our encodings, is the
  remaining unpublished edge; new public OOF libraries posted before the deadline are the other.

### 11.2 The 0.96187 – 0.96188 notebooks (reviewed 2026-10-14)

Four new public notebooks claim LB 0.96187 – 0.96188 (`other notebooks/2026-10-14/`, with the two
source stacks; the earlier five are in `other notebooks/2026-10-11/`). None of them gets there with a model of its own:

| notebook | what its submission is |
|---|---|
| `airline-passenger-satisfaction-0-96188` | nina2025's `0.96188.csv` resubmitted as is (its own LightGBM: 0.95920) |
| `s6e10-airline-satisfaction-roadmap-lb-0-96188` | 0.96 × a 0.96188 file + 0.025 / 0.015 × two golem 10-fold test predictions, weights set on the LB |
| `s6e10-orange-blend` | 0.9 × goodpjw2008's twin stack (0.96187) + 0.1 × one XGBoost |
| `secrets-to-satisfaction-it-s-stacking` v12 | 0.9 × `0.96188.csv` + 0.1 × samanyu1808's Megastack, weights set on the LB; its own LR stack (202 members) has nested CV 0.962172 |

- **The real sources** are three stacks: dariushafshar's 247 public OOFs (0.96187), goodpjw2008's twin stacks (a
  rank average of their 245-member stack and dariushafshar's, 0.96187) and ewtherka's family-wise L2 stack (0.96188,
  the file nina2025 republished). Their reported nested CV (0.96222 – 0.96223) is ours (0.962224); 0.96185 vs 0.96188
  is within the public LB's noise (about 1e-4 between two strong stacks on a public-LB-sized subset).
- **Family-wise L2** (TabPFN columns × 3, other foundation models × 1.5 before the LR, so the penalty shrinks them
  less): goodpjw2008 measured it on three meta-fold seeds of a pool almost identical to ours: +6.5, +2.4 and
  +0.4 × 1e-6 (4/5, 3/5, 2/5 folds), public LB unchanged at 0.96186. Noise; not added.
- **Our 17 members are in every pool now.** The ablation dataset is public, and both goodpjw2008's 245 (Part 10:
  "+ kagankoral's 17", LB 0.96186) and dariushafshar's 247 load it. That is why they caught up: the unpublished edge
  of round 11 is gone. The two collections agree to a test rank correlation of 0.99990, and the public LB spread
  for near-identical predictions is about ±0.000015, so 0.96185 – 0.96188 is one score.
- **A stack-level member to keep out:** goodpjw2008's extra-members dataset now also holds their frozen stack
  (`stack245`, OOF 0.962224). dariushafshar's loader picked it up, and with C = 1 it lifted his nested CV to
  0.962420 (+0.000272 from that one column): a stack of stacks inflates nested CV. Our loader reads `members.csv`
  and now also skips any member named `stack`.
- **Twin averages:** goodpjw2008's ½ + ½ of two near-identical stacks landed exactly on the better parent (0.96187).
  Ours (round 13: our stack + nina2025's `0.96188.csv`, rank correlation 0.999776) scored **0.96187**, between its
  parents (0.96185 and 0.96188): the same kind of hedge, not a gain.
- **Golem's library grew to 7 members outside the 124** (4 new); with them the stack's nested CV is 0.962219
  (−0.000004 against round 11, noise), so the round 11 file stays the CV choice.
- **What can still move the true score** is a member nobody else has, kept private until the competition closes:
  the full-context TabPFN-3.5 on our encodings (`kaggle_s6e10_tabpfn_members.ipynb`).
- **The golem library has grown** past `o`, `p`, `q` (a `s_realmlp_raw_auxev_10fold` member appears); the loader now
  takes every member from `o` on, once the input is updated to the latest dataset version.

### 11.3 TabPFN-3.5 on our encodings: smoke run (2026-10-14)

`notebook/kaggle_s6e10_tabpfn_members.ipynb` on Kaggle (GPU T4 x2, tabpfn 9.2.0, torch 2.11), member `pfn_enc`
(54 columns: raw + `orig_pred` + `orig_te` + `freq` + `te` / `te_pair`), `n_estimators` 4, `fit_with_cache`, one
full-size fold per GPU:

| fold | context | AUC | fit | 139,927 validation rows | peak memory |
|---|---|---|---|---|---|
| 0 | 279,854 (whole fold: out of memory) | 0.960720 | 49.7 min | 25.4 min | 12.5 GB |
| 1 | 279,854 | 0.959908 | 54.0 min | 27.1 min | 12.5 GB |

- **The whole training fold (560k rows) does not fit a T4** with the cache; half of it does, at 12.5 of 15 GB.
- **Slow:** one full job (fit + validation + 299,844 test rows) is about 140 min, so the full run is about 7 hours
  on both GPUs.
- **Weaker than the TabPFNs already in the pool**, on the same rows: abdullahsafwan333's 15-fold full-context
  TabPFN-3.5 scores 0.961897 / 0.960857 on these two folds (catfd10) and 0.961401 / 0.960503 (te + counts), so ours
  is 0.0006 – 0.0012 below. The pool already holds 16 TabPFN members; goodpjw2008's limited-context TabPFN
  (OOF 0.95978) moved their stack by −2.5 × 1e-6 ("the niche was already occupied").
- Decision pending: the full run costs 7 GPU hours for a member that is expected to add about nothing.

### 11.4 The 0.96197 – 0.96204 notebooks (reviewed 2026-10-19, `other notebooks/2026-10-19/`)

Nine new notebooks; **none brings a new OOF member.** The public top climbed 0.96188 → 0.96204 with test-only files:

- **LB-surface ascent** (mayankraaz, rabari9999 "rank 6" 0.96201, the surface probes, the 0.96203 "measured
  notebook"): a quadratic model of public score fitted over 60 – 96 scored files, then climbed. Public-LB tuning by
  construction; the authors say so themselves.
- **Copies and consensus files** (hanifnoerrofiq's `agentic-ensemble` copies a `latest_result.csv` with no code;
  the 0.96204 audit reproduces it; the 0.96201 consensus and nina2025's h-blend 7 rank-average the above).
- **goodpjw2008's 0.96197 (the useful one):** the 0.96190 – 0.96193 chain is an extrapolation along one axis,
  ewtherka's plain-LR file minus his 0.8 LR + 0.2 LightGBM-stacker file, i.e. *removing* the LightGBM stacker, pushed
  1.5× past zero. Rebuilt on the 182-member OOF pool, that direction **loses** nested CV at every strength
  (−13 × 1e-6 at 0.1, −933 × 1e-6 at 1.5, 0/5 folds), and the opposite direction **gains** (+7.6 to +10.6 × 1e-6,
  5/5 folds) while the public LB moves down (0.96198 → 0.96196). The public 20% and the 700k-row CV disagree in
  both directions on this axis: the public top is likely to reshuffle on the private board.
- **Taken from it:** the LightGBM-stacker blend, now section 2c of the meta-stack (LightGBM on the same logits, out
  of fold on the meta folds, shares 0.1 / 0.2 / 0.3 screened under the `MIN_GAIN` rule). Three independent
  measurements (yeashusemwal, goodpjw2008 Part 9 and 10-19) found +8 to +11 × 1e-6 on 5 of 5 folds. Expect the
  public LB not to show it.
- **Result (round 14):** LR 0.962219, LightGBM stacker alone 0.962035 (iterations 210 – 519), share 0.2 adopted:
  **nested CV 0.962234 (+0.000015)**, our best. Public LB 0.96184, one digit below round 11's 0.96185: the same
  direction goodpjw2008 saw. Its twin with `0.96188.csv` (rank correlation 0.999473) scored 0.96186.
