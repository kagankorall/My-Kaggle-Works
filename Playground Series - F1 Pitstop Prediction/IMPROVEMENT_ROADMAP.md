# F1 Pit Stop Prediction — Improvement Roadmap

Bu dosya, ensemble/Optuna dışında AUC kazancı sağlayabilecek alternatifleri öncelik sırasıyla listeler.

---

## Öncelik Tablosu

| # | Adım | Beklenen AUC Kazancı | Efor | Durum |
|---|------|---------------------|------|-------|
| 1 | GroupKFold (validation fix) | gerçek skoru görmek | düşük | ⬜ |
| 2 | Rolling / lag features | +0.003–0.010 | orta | ⬜ |
| 3 | 2023 regime isolation | +0.002–0.005 | orta | ⬜ |
| 4 | Pseudo-labeling | +0.002–0.005 | düşük | ⬜ |
| 5 | Stacking (meta-model) | +0.001–0.003 | orta | ⬜ |
| 6 | CatBoost ekleme | +0.001–0.002 | düşük | ⬜ |
| 7 | Optuna hyperparameter search | +0.001–0.002 | yüksek | ⬜ |
| 8 | Ensemble blend | +0.001–0.002 | düşük | ✅ |

---

## 1. Validation Stratejisi — GroupKFold

**Problem:** StratifiedKFold aynı yarıştan ardışık lap'ları hem train hem validation'a koyar.  
Bu, OOF skorunu gerçekten iyi gösterir ama aslında "cheating" yapar — model aynı yarışın başka lap'larını görmüş olur.

**Çözüm:**
```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
groups = train_fe['Race'].astype(str) + '_' + train_fe['Year'].astype(str)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_base, y, groups=groups), 1):
    ...
```

**Neden önemli:** Validation stratejisi yanlışsa Optuna da yanlış optimize eder. Bu adım diğer her şeyin temelini oluşturur.

---

## 2. Rolling / Lag Features

**Fikir:** Bir sürücünün son 3–5 lap'ındaki performans trendini yakalamak.  
Mevcut `LapTime_Delta` tek bir lap'a bakıyor; rolling ortalama çok daha stabil bir degradasyon sinyali verir.

```python
# Sürücü + yarış bazında sıralama gerekiyor
all_df = all_df.sort_values(['Race', 'Year', 'Driver', 'LapNumber'])

grp = all_df.groupby(['Race', 'Year', 'Driver'])

# Son 3 ve 5 lap'ın ortalama lap süresi
all_df['LapTime_Roll3'] = grp['LapTime (s)'].transform(lambda x: x.shift(1).rolling(3).mean())
all_df['LapTime_Roll5'] = grp['LapTime (s)'].transform(lambda x: x.shift(1).rolling(5).mean())

# Mevcut lap ile rolling ortalama farkı → anlık yavaşlama
all_df['LapTime_vs_Roll5'] = all_df['LapTime (s)'] - all_df['LapTime_Roll5']

# Tyre life artış hızı (genelde 1, ama safety car altında sıçrar)
all_df['TyreLife_Delta'] = grp['TyreLife'].transform(lambda x: x.diff())

# Stint içinde kaçıncı lap (0-indexed)
all_df['LapInStint'] = grp['LapNumber'].transform(lambda x: x - x.min())
```

**Not:** `.shift(1)` kritik — hedefe bakan leak'i önler (gelecek lap bilgisi kullanma).

---

## 3. 2023 Regime Isolation

**Problem:** 2023 pit rate'i ~%1 vs diğer yıllar ~%28. Şu an `is_2023` flag'i var ama model hâlâ tüm veriyi aynı anda öğreniyor.

### Seçenek A — Yıl bazında ayrı model
```python
mask_2023 = train_fe['Year'] == 2023

model_2023    = train_and_predict(train_fe[mask_2023], ...)
model_non2023 = train_and_predict(train_fe[~mask_2023], ...)

# Test için yıla göre yönlendir
test_2023_preds    = model_2023.predict(test_fe[test_fe['Year'] == 2023])
test_non2023_preds = model_non2023.predict(test_fe[test_fe['Year'] != 2023])
```

### Seçenek B — Sample weight ile cezalandır
```python
# 2023 lap'larına düşük ağırlık ver, modelin onlara "güvenmesini" azalt
sample_weights = np.where(train_fe['Year'] == 2023, 0.3, 1.0)

model.fit(X_tr, y_tr, sample_weight=sample_weights[tr_idx], ...)
```

**Öneri:** Önce Seçenek B ile hızlı test et; kazanç varsa Seçenek A'ya geç.

---

## 4. Pseudo-Labeling

**Fikir:** Test setindeki yüksek güvenlik tahminleri büyük ihtimalle doğrudur — bunları eğitim setine ekle.

```python
# Ensemble tahminlerini kullan
test_fe['PitNextLap_pseudo'] = ensemble_test_preds

# Çok emin olduğumuz tahminleri seç
high_conf = test_fe[
    (ensemble_test_preds > 0.95) | (ensemble_test_preds < 0.05)
].copy()
high_conf['PitNextLap'] = (high_conf['PitNextLap_pseudo'] > 0.5).astype(int)

# Genişletilmiş train seti
train_pseudo = pd.concat([train_fe, high_conf], ignore_index=True)

# Aynı CV loop'unu train_pseudo üzerinde çalıştır
```

**Dikkat:** Pseudo-labeling iteratif uygulanabilir (1. turda %95 eşik, 2. turda %90 gibi).  
Genellikle yarışmanın 2–3. haftasında yapılır — çünkü önce iyi bir base model gerekir.

---

## 5. Stacking (Meta-Model)

**Blend vs Stacking farkı:**
- Blend: `0.508 * lgbm_oof + 0.492 * xgb_oof` → sabit ağırlık
- Stack: Her satır için farklı ağırlık — bir Logistic Regression veya Ridge öğrenir

```python
from sklearn.linear_model import LogisticRegression

# Level-1 OOF'ları birleştir
oof_stack = np.column_stack([lgbm_oof, xgb_oof])     # shape: (N, 2)
test_stack = np.column_stack([test_preds, xgb_test_preds])

# Level-2 meta-model (kendi içinde CV ile fit et)
meta = LogisticRegression(C=1.0)
meta.fit(oof_stack, y)

final_preds = meta.predict_proba(test_stack)[:, 1]
print(f'Stack OOF AUC: {roc_auc_score(y, meta.predict_proba(oof_stack)[:, 1]):.5f}')
```

**Daha güçlü meta-model:** LightGBM ile stacking (özellikle 3+ base model varsa).

---

## 6. CatBoost Ekleme

CatBoost'un farkı: kategorik sütunları (`Driver`, `Race`, `Compound`) string olarak alır, kendi içinde ordered target encoding uygular — fold içi manual TE'ye gerek kalmaz.

```python
from catboost import CatBoostClassifier, Pool

cat_features = ['Driver', 'Race', 'Compound']

# CatBoost string kolonları olduğu gibi alır
X_cat = train_fe[features_base_str + cat_features]  # str kolonları koru

model = CatBoostClassifier(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    eval_metric='AUC',
    random_seed=RANDOM_STATE,
    verbose=False,
)
model.fit(
    Pool(X_tr, y_tr, cat_features=cat_features),
    eval_set=Pool(X_val, y_val, cat_features=cat_features),
    early_stopping_rounds=50,
)
```

**Stack'e ekleme:** `[lgbm_oof, xgb_oof, cat_oof]` → 3 kolonlu meta-model.

---

## 7. Alan Spesifik Özellikler (Domain FE)

Bu özellikler F1 domain bilgisine dayanır ve başka kagglerların çoğu yazmaz:

```python
# Aynı yarışta sürücünün pit geçmişi
grp_driver_race = all_df.groupby(['Race', 'Year', 'Driver'])

# Şimdiye kadar kaç pit yaptı?
all_df['CumulativePits'] = grp_driver_race['PitStop'].transform('cumsum')

# Son pit'ten bu yana kaç lap geçti?
all_df['LapsSinceLastPit'] = all_df.groupby(
    ['Race', 'Year', 'Driver', 'Stint']
)['LapNumber'].transform(lambda x: x - x.min())

# Yarışta toplam beklenen pit sayısı (compound'a göre kaba tahmin)
compound_avg_stints = train_fe.groupby('Compound')['Stint'].max().to_dict()
all_df['ExpectedTotalPits'] = all_df['Compound'].map(compound_avg_stints) - 1

# Pit yapmak için kalan "bütçe"
all_df['PitsRemaining_estimate'] = (all_df['ExpectedTotalPits']
                                    - all_df['CumulativePits']).clip(0)
```

---

## 8. Probability Calibration

ROC-AUC metriğinde etkisi sınırlı ama submission kalitesini artırır.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# OOF üzerinde kalibrasyon eğrisi çiz
frac_pos, mean_pred = calibration_curve(y, blend_oof, n_bins=20)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(mean_pred, frac_pos, marker='o', label='model')
ax.plot([0, 1], [0, 1], 'k--', label='perfect')
ax.set_title('Calibration Curve')
ax.legend()
plt.show()

# Isotonic regression ile kalibre et
from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(blend_oof, y)
calibrated_test_preds = iso.predict(ensemble_test_preds)
```

---

## Önerilen Sıradaki Adım

```
1. GroupKFold uygula → gerçek validation skoru nedir bak
2. Rolling features (LapTime_Roll3/5, LapInStint) ekle → en yüksek ROI
3. Pseudo-labeling dene → düşük efor, ciddi kazanç olabilir
4. CatBoost ekle + 3'lü stack
```
