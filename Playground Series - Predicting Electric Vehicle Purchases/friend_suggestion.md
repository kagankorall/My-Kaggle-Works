# Kaggle Playground 2026 — EV Purchase Prediction: Final Days Action Plan

Based on your findings from `PROJECT_BRIEF_2.md`, your pipeline is operating with exceptional discipline. Since tree correlation has reached 0.999 and the noise level in the original dataset is clear, standard feature engineering and basic hyperparameter tuning have hit a ceiling.

To break through this plateau and capture the remaining AUC gains (+0.0002+) that top competitors achieve without relies on simple Optuna searches, use this high-leverage action plan for the final days before the **30 September 2026** deadline.

---

## 1. Exploiting Generator Artifacts & Mathematical Patterns
Synthetic data generators (e.g., CTGAN, Synthpop) leave subtle mathematical signatures that trees do not naturally split on unless explicitly engineered.

- **Modulo & Precision Features:**
  - `Income_mod_100` / `Income_mod_500` / `Income_mod_1000`: Quantization artifacts often exist in synthetic continuous variables like `Annual_Income_USD` and `Daily_Commute_km`.
  - Digit count / trailing zeroes: Count precision patterns or decimal signatures in continuous features.
- **Row Index Leakage Check:**
  - Create an `index` / `row_id` feature or test `Target Encoding` against chunks of row ordering. Check if the synthetic dataset was concatenated or generated without full shuffling.

---

## 2. Advanced Multi-Column Group Target Encoding
Exact-value TE on single columns gave +0.0027. Extend this to interaction keys to help 15-leaf shallow trees access complex combinations without needing deeper splits.

- **Grouped TE Keys:**
  - Compute nested OOF Exact-Value TE for high-impact interactions:
    - `Environmental_Concern` × `Subsidy_Available`
    - `Subsidy_Available` × `Range_Anxiety`
    - `City_Type` × `Charging_Stations_Near_Home`
  - Group continuous features (`Age`, `Income`) by categorical anchors (e.g., `Income` mean/median within `Environmental_Concern × Subsidy_Available` groups).

---

## 3. Non-Monotonic Column Enhancements

- **Bayesian / Smoothed Exact-Value TE:**
  Apply smoothing to nested exact-value encodings to stabilize low-frequency categories:
  $$S_i = rac{n_i \cdot ar{y}_i + m \cdot y_{	ext{global}}}{n_i + m}$$
  Test $m \in [10, 50, 100]$ using grid search across folds.
- **Contextual Group Aggregations & Weight of Evidence (WoE):**
  - Compute relative differences (e.g., `Age - Mean_Age_For_Income_Bracket`).
  - Test WoE encoding on non-monotonic columns (`Age`, `Charging_Stations`, `Number_of_Cars`) for stable log-odds ranking.

---

## 4. Pairwise & Surrogated Loss Functions (Objective-Level Diversity)
Log-loss models optimize probabilities rather than relative ranks. Switching objectives directly optimizes the ROC-AUC evaluation metric.

- **RankNet / Pairwise Loss:**
  Train LightGBM or XGBoost with pairwise ranking loss (`lambdarank` in LightGBM or `rank:pairwise` in XGBoost).
- **Focal Loss:**
  Train a LightGBM/XGBoost variant using Binary Focal Loss ($\gamma \in [1.0, 2.0]$) to force focus on borderline decisions.
- **Benefit:** Even if standalone OOF AUC is comparable, ranking-loss models produce un-correlated predictions vs log-loss models, significantly boosting rank ensemble performance.

---

## 5. Denoising Autoencoder (DAE) Latent Embeddings
Instead of feeding raw features to Neural Networks (which lagged by 0.0016), use a DAE for unsupervised feature extraction.

- Train a Denoising Autoencoder (or Masked Autoencoder) on train + test features to reconstruct corrupted inputs.
- Extract bottleneck layer representations (e.g., 16–32 dense features).
- Pass these latent embeddings back into LightGBM and CatBoost. This provides manifold representations that decision trees cannot natively construct.

---

## 6. Post-Processing & Grouped Rank Calibration

Since ROC-AUC strictly depends on prediction ordering, small adjustments near decision boundaries directly move the public/private LB score.

- **Subgroup Rank Normalization:**
  Transform raw probabilities into percentile ranks *within* major deterministic subgroups (e.g., rank separately inside `Subsidy_Available = Yes` vs `No`), then re-combine:
  $$	ext{Rank}_{	ext{final}} = w_1 \cdot 	ext{Rank}(	ext{Group}_{	ext{Subsidy=Yes}}) + w_2 \cdot 	ext{Rank}(	ext{Group}_{	ext{Subsidy=No}})$$
- **Rank Power Averaging ($P^p$):**
  When ensembling OOF predictions, convert probabilities to percentile ranks, raise them to power $p \in [0.5, 2.0]$, and optimize $p$ against OOF AUC:
  $$	ext{Ensemble Score} = w_1 \cdot 	ext{Rank}(M_1)^p + w_2 \cdot 	ext{Rank}(M_2)^p$$
- **Isotonic Regression / Spline Calibration:**
  Apply Isotonic Regression to fold predictions before rank-averaging to smooth boundary transitions.

---

## 7. Final Verdict on the Set-Aside List

| Technique | Recommendation | Reason |
|---|---|---|
| **Soft Pseudo-Labelling** | **Skip** | High implementation cost to prevent nested fold leakage with minimal expected gain on synthetic data. |
| **TabNet / FT-Transformer** | **Skip** | High compute time and unlikely to close the 0.0016 gap in the final days. |
| **Original Data Features** | **Skip** | Proven to introduce unwanted noise into the generator's smoothed distribution. |
