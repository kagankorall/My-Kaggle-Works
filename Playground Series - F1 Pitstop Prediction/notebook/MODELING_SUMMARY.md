# Modelleme Bölümü Özeti — F1 Pit Stop Prediction

Bu doküman, [formula1_pit_stop_prediction.ipynb](formula1_pit_stop_prediction.ipynb) defterindeki **Bölüm 8 — Modeling** ve **Bölüm 9 — Final Submission** kısımlarında yapılan işleri özetler.

## Hedef

Her satır bir sürücünün belirli bir turunu temsil ediyor; amaç bir sonraki tura pit yapıp yapmayacağını (`PitNextLap`, ikili 0/1) tahmin etmek. Sınıf dağılımı dengesiz: ~%20 pozitif.

---

## 8.1 Modelleme Setup

- **Atılan kolonlar:** `id`, ham string `Driver` / `Compound` / `Race`, target ve etiket-kodlanmış `Driver_enc` / `Race_enc`. Yüksek kardinaliteli `Driver` ve `Race`, fold içinde **smoothed target encoding** ile yeniden kodlanıyor (sızıntı önlemi).
- **Compound:** label encoding yerine **One-Hot Encoding** (5 binary kolon). Önceki ordinal varsayım (SOFT < MEDIUM < HARD) yanlıştı; OHE kaldırıldı.
- **Cross-sectional features:** `groupby([Race, Year, LapNumber])` üzerinden 4 yeni rank/oran feature (`TyreLife_pct_vs_lap_mean`, `TyreLife_rank_in_lap`, `CumDeg_rank_in_lap`, `LapTime_rank_in_lap`) — undercut/overcut sinyali.
- **Toplam:** ~20 engineered + 2 target-encoded.
- **2023 sample weighting:** `WEIGHT_2023 = 1.0` (devre dışı). 0.3'te denendi, LB'de düz sonuç (0.94697 → 0.94673).

## 8.2 5-Fold Cross-Validation — LightGBM (regülasyonlu)

- **Validasyon:** `StratifiedKFold(n_splits=5)` — test setinin satır-bazlı örnekleme desenine uyduğu için seçildi.
- **Daha önce denenen:** `StratifiedGroupKFold` (race bazında hold-out) çok karamsar bir tablo çiziyor, early-stopping erken tetikleniyordu — geri çevrildi.
- **Hiperparametreler (regülasyonlu):**
  - `num_leaves`: 127 → **63**
  - `learning_rate`: 0.05 → **0.02**
  - `min_data_in_leaf`: 20 → **100**
  - `lambda_l2`: 0.1 → **0.5**
  - `n_estimators`: 2000 → **5000**, `early_stopping(50 → 100)`
  - `feature_fraction=0.8`, `bagging_fraction=0.8`, `bagging_freq=5`, `lambda_l1=0.1`
- **Smoothed target encoding** her fold'ta sadece eğitim katlanından öğreniliyor (smooth=10). Test öngörüleri 5 fold ortalaması.
- **Çıktılar:** `oof`, `test_preds`, `fold_aucs`, `fold_models`. Diagnostic — submission yazılmıyor.

## 8.3 Threshold Tuning (F1)

- 0.05–0.95 aralığında 91 eşik tarandı. Dengesiz hedef nedeniyle optimal eşik genellikle 0.30–0.40 arasında.

## 8.4 Confusion Matrix (Best Threshold)

- En iyi eşikte OOF confusion matrix ve sınıf bazlı precision/recall/F1 raporu.

## 8.5 Feature Importance

- 5 fold ortalaması üzerinden LightGBM `feature_importances_`.
- **En güçlü sinyaller:** `Driver_te` ve `Race_te` (target-encoded) baskın → tarihsel pit oranı, label integer'larından çok daha bilgilendirici.
- **Fizik tabanlı sinyaller:** `Cumulative_Degradation`, `LapsRemaining`, `TyreLife_pct` öne çıkıyor.

## 8.6 Pseudo-Labeling (Deney — submit edilmedi)

- Baseline modelin yüksek-güvenli test tahminleri (prob ≥ 0.97 veya ≤ 0.03) ground truth kabul edilip eğitime eklendi.
- **LB sonucu:** 0.94672 (baseline 0.94697'den hafif düşüş). Kayıt için saklanıyor, submission yok.

## 8.7 Multi-Seed LGBM Averaging (10 seed)

- **Seedler:** `[42, 7, 2024, 123, 456, 789, 31, 99, 2025, 1337]`.
- Her seed hem fold split'lerini hem modelin kendisini değiştiriyor.
- Test tahminleri 10 seed × 5 fold = 50 modelin ortalaması → bagging-noise variansı ~1/N oranında düşüyor.
- **Çıktılar:** `oof_seedavg`, `test_preds_seedavg`.

## 8.9 CatBoost — Native Categorical + Rank Blend (Multi-Seed)

- **CatBoost'a ham string kategorikallar veriliyor:** `cat_features=['Driver', 'Race', 'Compound']`. LGBM'in OHE/target-encoded feature'larından bağımsız feature işleme mimarisi.
- **Hiperparametreler:** `iterations=3000`, `lr=0.03`, `depth=8`, `l2_leaf_reg=5`, `early_stopping_rounds=100`.
- **3-seed averaging:** `CB_SEEDS = [42, 7, 2024]`. Test tahminleri 3 seed × 5 fold = 15 modelin ortalaması.
- **Çıktılar:** `cb_oof_seedavg`, `cb_test_seedavg` (alias: `cb_oof`, `cb_test`).

### 8.9.2 Rank-Blend (grid-tuned w)

- **Grid search:** `np.arange(0.30, 0.81, 0.05)` üzerinden 11 ağırlık taranıyor (LGBM payı w). Her w için OOF AUC hesaplanıp en yüksek olan `best_w` seçiliyor.
- **Rank-blend formülü:** `w * rankdata(lgbm) + (1-w) * rankdata(catboost)` → normalize.
- **Diversity karar kuralı:** OOF korelasyonu < 0.95 ise blend gerçek diversity sinyali; ≥ 0.95 ise blend muhtemelen LB'ye yansımaz.
- **Önceki başarısız blend (LB 0.94705):** O zaman CatBoost'a LGBM için hazırlanan target-encoded feature'lar verilmişti — aynı sinyal, blend OOF'a overfit etti. Bu kez sıfırdan ham kategorikallerle.

## 8.10 Modelleme Özeti — LB Sonuçları

| Deney | LB Skoru | Δ | Not |
|---|---|---|---|
| Baseline LGBM (StratifiedGroupKFold, tek seed) | 0.94697 | — | İlk validasyon |
| + Pseudo-labelling | 0.94672 | -0.00025 | Hafif düşüş |
| + 3-seed averaging (StratifiedGroupKFold) | 0.94721 | +0.00024 | İlk gerçek iyileşme |
| + LGBM 3-seed + CatBoost (target-encoded) blend | 0.94705 | -0.00016 | Aynı feature → diversity yok |
| + 5-seed averaging (StratifiedKFold) | 0.94891 | +0.00170 | Validasyon değişikliği + seed |
| + 10-seed averaging | 0.94898 | +0.00007 | Compound OHE + cross-sectional ranks |
| + Hiperparametre regülasyonu | 0.94911 | +0.00013 | OOF/LB gap kapandı |
| + LGBM × CatBoost blend (60/40, 1-seed CB) | 0.94944 | +0.00033 | Native categorical → diversity LB'ye yansıdı |
| + CatBoost 3-seed + grid-tuned w | **0.94947** | +0.00003 | Plato — diminishing returns |
| Time-series rolling features (kaldırıldı) | 0.94564 | — | Overfit |
| 2023 sample-weight = 0.3 (devre dışı) | 0.94673 | — | Düz |

**Net kazanç (10-seed → final):** +0.00049

### Çıkarımlar

- Bu veri setinde **variance reduction (multi-seed)** + **regülasyon** + **heterojen feature işleme (LGBM target-encoded × CatBoost native categorical)** birlikte tutarlı LB iyileşmesi sağlıyor.
- **Cross-sectional ranks** (lap-içi rakip karşılaştırması) ufak ama temiz sinyal getirdi; time-series rolling'in aksine sızıntı riski yok.
- **Compound OHE**, label encoding'in ordinal varsayımını ortadan kaldırarak modelin her tipi bağımsız kalibre etmesini sağladı.
- **Validasyon stratejisi belirleyici:** GroupKFold gerçek görevden daha zor bir görevi çözüyordu, early stopping'i sabote ediyordu.
- **Blend başarısı feature işleme çeşitliliğine bağlı:** Aynı target-encoded feature'larla beslenen iki model overfit'e neden olur (eski 0.94705); LGBM (OHE+TE) × CatBoost (native cat.) gerçek diversity ürettiği için LB'ye yansıdı.
- **0.94944'ten sonra plato:** CatBoost multi-seed ve grid search marjinal kazanç (+0.00003) verdi — diminishing returns sınırına yaklaşıldı.

---

## 9. Final Submission

Bölüm yazılan **üç** CSV:

- `submission_lgbm.csv` — Bölüm 8.7'nin 10-seed LGBM ortalaması (regülasyonlu).
- `submission_blend.csv` — LGBM 10-seed × CatBoost 3-seed rank-blend (grid-tuned w), Bölüm 8.9.
- `submission_final.csv` — OOF AUC'si yüksek olan otomatik kopyalanır.

İki submission'u Kaggle'da seçili tutmak Public/Private LB ayrımına karşı sigorta görevi görür.

### 9.1 Pipeline Özeti

```
Train ──► LGBM × 10 seed avg ─────┐
                                   ├──► rank-blend (grid-tuned w) ──► submission_blend.csv
      ──► CatBoost × 3 seed avg ──┘

      ──► LGBM × 10 seed avg ───────────────────────────────────────► submission_lgbm.csv
```

- **Validasyon:** 5-fold `StratifiedKFold` (test örnekleme desenine uyumlu).
- **LGBM hiperparam (regülasyonlu):** `num_leaves=63`, `lr=0.02`, `min_data_in_leaf=100`, `lambda_l2=0.5`, `n_estimators=5000` + `early_stopping(100)`.
- **CatBoost:** ham string `cat_features=['Driver','Race','Compound']` + sayısal feature'lar; `iterations=3000`, `lr=0.03`, `depth=8`, `l2_leaf_reg=5`. **3 seed averaging.**
- **Featurelar:** ~20 engineered (4 agg, 3 flag, 2 interaction, **4 cross-sectional rank**, **5 Compound OHE**) + 2 target-encoded (Driver, Race, fold içinde).
- **Variance reduction:** LGBM 10 seed × 5 fold + CatBoost 3 seed × 5 fold; her seed hem fold split'lerini hem model bagging'ini değiştiriyor.
- **Blend ağırlığı:** OOF AUC üzerinde 0.30–0.80 aralığında 0.05 adımla grid search; en yüksek OOF AUC veren w sabitlenir.

**Neden bu konfigürasyon:** Sistematik denemeler sadece varyans azaltma + regülasyon + heterojen feature işleme blend'inin LB'ye yansıdığını gösterdi. Aynı feature'larla beslenen modellerin blend'i (eski 0.94705) ve time-series rolling features (0.94564) overfit etti — birinde diversity yok, diğerinde leak. Final LB: **0.94947**.
