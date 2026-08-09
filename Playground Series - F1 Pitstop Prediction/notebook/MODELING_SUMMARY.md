# Çalışma Özeti — F1 Pit Stop Prediction

Bu doküman [formula1_pit_stop_prediction.ipynb](formula1_pit_stop_prediction.ipynb) defterinin tamamını (EDA, Feature Engineering, Modeling, Final Submission) özetler.

## Hedef ve Veri

- **Görev:** Her satır bir sürücünün belirli bir turunu temsil eder; amaç **bir sonraki tura pit yapıp yapmayacağını** (`PitNextLap`, ikili 0/1) tahmin etmektir.
- **Veri:** Train ~439K, test ~188K lap-level satır (~70/30 split). 16 train kolonu, 15 test kolonu. Hiçbir kolonda eksik değer yok.
- **Metrik:** Public LB skorlarının dağılımı (0.94+ aralığı) **ROC-AUC** olduğunu gösteriyor.

---

## Bölüm 6 — EDA Bulguları

### 6.1 Hedef Dağılımı
- `PitNextLap` ortalaması **~%20** → dengesiz sınıf.

### 6.3 Yıl Analizi (Kritik Anomali)
- **2023 yılında pit oranı ~%1**, diğer yıllarda (2022/2024/2025) ~%27–30.
- Olası açıklama: 2023 için ağır sentetik augmentation veya farklı etiketleme rejimi.
- Aksiyon: **`is_2023` flag feature'ı**; sample-weighting ile düşürmek denendi (LB 0.94697 → 0.94673, devre dışı).

### 6.4 Compound (Lastik) Analizi
- **HARD** baskın (~170K lap) ve en yüksek pit oranı (~%33).
- **WET / INTERMEDIATE** nadir pit yapıyor.

### 6.5 Race (Pist) Analizi
- Pit oranı pistler arasında **%9–39 aralığında** değişiyor (Chinese GP/Monaco zirvede, Mexico City/Miami en düşük).

### 6.6 Driver Kardinalitesi
- 887 unique driver. **131 gerçek 3-harfli kod** + **756 sentetik ID**.
- High cardinality → **smoothed target encoding**; `Driver_kind` flag'i eklendi.

### 6.7 TyreLife
- **Pit oranı `TyreLife` ile monoton artıyor**, ~40+ turda zirve. Tek başına en güçlü feature.

### 6.8–6.11 Diğer Patternler
- **RaceProgress:** erken/orta yarış penceresinde zirve, finalde düşüş.
- **Stint:** erken stint'ler (1, 2) çok daha fazla pit yapıyor.
- **PitStop ⊥ PitNextLap (leak kontrolü):** %19 vs %25 — leak değil, kullanışlı feature.

### Test Set Kompozisyonu (Sonradan Doğrulandı)
- **0 unseen year, race, driver veya (Race, Year) çifti.**
- Test'in **%100'ü** train'de görülen yarışlardan satır-bazlı örnekleme.
- **(Race, Year, LapNumber) hücre seviyesinde:** ortalama 70 train satırı/hücre, train_frac = 0.70 ± 0.14, %85 hücrede ≥ 15 train satırı, sparse hücreler test'in %0.2'sini etkiliyor.
- **Sonuç:** `StratifiedKFold` (per-row) optimal. Private LB shake-up riski düşük. Race-matrix aggregation feature'ları için zemin ideal.

---

## Bölüm 7 — Feature Engineering

Train + test birleşik DataFrame (`all_df`) üzerinde tüm feature'lar tek seferde üretildi.

### 7.1 / 7.2 Race Max Lap + LapsRemaining
- `Race_MaxLap = groupby('Race')['LapNumber'].transform('max')`
- `LapsRemaining = Race_MaxLap − LapNumber`

### 7.3 Compound TyreLife Q95 / TyreLife_pct
- `Compound_TyreLife_Q95`: her compound için 95. persantil TyreLife.
- `TyreLife_pct = TyreLife / Q95` → compound-normalize fraksiyon worn.

### 7.4 Year / Race / Driver Flag'leri
- `is_2023`, `is_pre_season`, `Driver_kind` (gerçek vs sentetik).

### 7.5 Etkileşim ve Cross-Sectional Feature'lar
- `LapTime_vs_RaceMedian` — pist-agnostik lap-time normalizasyonu.
- `TyreLife_x_Stint` — compound risk.
- **Cross-sectional (lap-içi rakip karşılaştırması)** — `groupby([Race, Year, LapNumber])`:
  - `TyreLife_pct_vs_lap_mean`
  - `TyreLife_rank_in_lap`
  - `CumDeg_rank_in_lap`
  - `LapTime_rank_in_lap` (undercut/overcut sinyali)

### 7.6 Race Matrix Aggregation (Safety Car / VSC Proxy) — YENİ
Her `(Race, Year, LapNumber)` hücresinin train satırlarını agregeleyerek o turun bağlamını test'e taşıma:
- `lap_train_pitstop_rate` — train'de hücre içi `PitStop` oranı (SC/VSC indicator).
- `lap_train_avg_laptime_delta` — train'de hücre içi ortalama `LapTime_Delta` (toplu yavaşlama).

**Leak-safe pipeline:**
- Train satırı için **leave-one-out** (kendi değeri hücreden çıkarılır)
- Test satırı için tüm train kullanılır
- Tek formül, `train_mask` çarpanı ile: `pit_n_eff = pit_n - 1[train]`, `pit_sum_eff = pit_sum - PitStop * 1[train]`
- Sparse hücreler (train_n = 0) → global ortalama fallback (test'in %0.2'si)

**Veri analizi destekli (Bölüm 6 sonu)**: %85 hücrede ≥15 train satırı, ortalama 70/hücre.

### 7.7 Categorical Encoding (Revize)
- **`Compound`**: **One-Hot Encoding** (5 binary kolon). Önceki ordinal label encoding kaldırıldı (`SOFT < MEDIUM < HARD` varsayımı yanlıştı).
- **`Race`, `Driver`**: label kodları sadece referans (`_enc`); modele giren değil. Fold içinde **smoothed target encoding** (smooth=10).

### 7.8 Split + Final Feature Set
- **~22 engineered feature toplamı:** 4 agregasyon, 3 flag, 2 interaction, 4 cross-sectional rank, **2 race-matrix aggregate**, 5 Compound OHE, 2 label-encoded referans.

### 7.9 Yeni Feature'ların Target Korelasyonu
- En güçlü: `is_2023` (−0.32), `LapsRemaining` (−0.27), `Compound_TyreLife_Q95` (+0.25), `TyreLife_pct` (+0.21).

### Denenip Atılan Feature'lar
- **Time-series rolling features:** LB 0.94697 → 0.94564, overfit. Yerini cross-sectional rank'ler aldı.
- **2023 sample-weighting (0.3):** LB düz (0.94697 → 0.94673), devre dışı.

---

## Bölüm 8 — Modeling

### 8.1 Modelleme Setup
- **Atılan:** `id`, ham string `Driver` / `Compound` / `Race`, target, label-encoded `Driver_enc` / `Race_enc`.
- **Modele giren:** numeric/aggregated/flag/interaction/OHE/cross-sectional + race-matrix + fold-içi `Driver_te`, `Race_te`.

### 8.2 5-Fold CV — Baseline LightGBM (Regülasyonlu)
- **Validasyon:** `StratifiedKFold(n_splits=5)` (test set kompozisyonu ile uyumlu).
- **Hiperparametreler (regülasyonlu):**
  - `num_leaves: 127 → 63`
  - `learning_rate: 0.05 → 0.02`
  - `min_data_in_leaf: 20 → 100`
  - `lambda_l2: 0.1 → 0.5`
  - `n_estimators: 2000 → 5000`, `early_stopping(50 → 100)`
- **Smoothed target encoding** her fold'ta yalnız train portionundan fit.

### 8.3–8.5 Diagnostic
- Threshold tuning (F1), confusion matrix, feature importance.
- `Driver_te` ve `Race_te` baskın; `Cumulative_Degradation`, `LapsRemaining`, `TyreLife_pct` fizik tabanlı.

### 8.6 Pseudo-Labeling (Deney)
- Yüksek-güvenli test tahminleri eğitime eklendi. **LB:** 0.94672 (baseline 0.94697'den düşük). Submit edilmedi.

### 8.7 LGBM 10-Seed Averaging
- `SEEDS = [42, 7, 2024, 123, 456, 789, 31, 99, 2025, 1337]`.
- 10 seed × 5 fold = 50 modelin ortalaması.

### 8.9 CatBoost — Native Categorical + 2-Way Rank Blend
- **Anahtar fikir:** LGBM'e (OHE + target-encoded) feature verilirken **CatBoost'a ham string** (`cat_features=['Driver','Race','Compound']`) → tamamen farklı feature işleme.
- **Önceki başarısız blend (LB 0.94705):** CatBoost'a LGBM için hazırlanan target-encoded feature'lar verilmişti — aynı sinyal, overfit. Bu kez native categorical ile çözüldü.
- **Hiperparametreler:** `iterations=3000`, `lr=0.03`, `depth=8`, `l2_leaf_reg=5`.
- **3-seed averaging:** `[42, 7, 2024]`.
- **2-way rank-blend:** OOF AUC üzerinde grid search (0.30–0.80, adım 0.05).

### 8.10 NN Blend — DENEME, BAŞARISIZ
- **Yapı:** `MLPClassifier(hidden=(128, 64), alpha=1e-3, early_stopping=True)` + **`QuantileTransformer(output_distribution='normal')`** (RankGauss benzeri) + 3-seed averaging.
- **Sonuçlar:**
  - NN OOF AUC: **0.94424** (LGBM 0.94977 ve CatBoost 0.94955'ten ~0.005 düşük)
  - Diversity check **FAIL** (NN-LGBM veya NN-CatBoost korelasyonu ≥ 0.95)
  - 3-way blend OOF: 0.94988 (2-way 0.95007'den **−0.00019**)
- **Teşhis:** Embedding'siz MLP, target-encoded kategorical sinyali ağaçlardan daha **az verimli** ama **aynı yönde** işliyor → aynı sinyali görüyor (yüksek korelasyon) ama daha kötü işliyor (düşük AUC) → blend'e net zarar.
- **Karar:** NN, `submission_nn_blend.csv` yazılmadı. Bölüm 9 otomatik 2-way blend kullanıyor.

---

## 8.11 LB Sonuçları — İlerleme Tablosu

| Deney | LB Skoru | Δ | Not |
|---|---|---|---|
| Baseline LGBM (StratifiedGroupKFold, tek seed) | 0.94697 | — | İlk validasyon |
| + Pseudo-labelling | 0.94672 | -0.00025 | Hafif düşüş, submit edilmedi |
| + 3-seed averaging (StratifiedGroupKFold) | 0.94721 | +0.00024 | İlk gerçek iyileşme |
| + LGBM 3-seed + CatBoost (target-encoded) blend | 0.94705 | -0.00016 | Aynı feature → diversity yok |
| + 5-seed averaging (StratifiedKFold) | 0.94891 | +0.00170 | Validasyon değişikliği + seed |
| + 10-seed averaging | 0.94898 | +0.00007 | Compound OHE + cross-sectional ranks |
| + Hiperparametre regülasyonu | 0.94911 | +0.00013 | OOF/LB gap kapandı |
| + LGBM × CatBoost blend (60/40, 1-seed CB) | 0.94944 | +0.00033 | Native categorical → diversity LB'ye yansıdı |
| + CatBoost 3-seed + grid-tuned w | **0.94947** | +0.00003 | Plato sinyali |
| + Race-matrix aggregates + NN deneme | 0.94945 | -0.00002 | Race-matrix net etki sıfır; NN FAIL |
| Time-series rolling features (kaldırıldı) | 0.94564 | — | Overfit |
| 2023 sample-weight = 0.3 (devre dışı) | 0.94673 | — | Düz |

**En iyi LB:** 0.94947 (önceki run). Mevcut state'te 0.94945 — istatistiksel olarak özdeş.

**Net kazanç (baseline 0.94697 → en iyi 0.94947):** **+0.00250**

### Plato Doğrulaması — OOF/LB Gap'i

| Metrik | OOF AUC | LB | Gap |
|---|---|---|---|
| LGBM 10-seed (race-matrix dahil) | 0.94977 | 0.94915 | **0.00062** |
| 2-way blend | 0.95007 | 0.94945 | **0.00062** |
| 3-way blend (NN dahil) | 0.94988 | (yazılmadı) | — |

OOF/LB gap **0.0006**'da sabit → modelin "kapasite tavanına" geldiğinin işareti. OOF'taki iyileşmeler LB'ye tutarlı şekilde ~0.0006 kayıpla yansıyor; daha fazla optimizasyon bu band içinde kalıyor.

### Race-Matrix Feature'ları — Detaylı

| Submission | Önceki (race-matrix YOK) | Mevcut (race-matrix VAR) | Δ |
|---|---|---|---|
| `submission_lgbm.csv` | 0.94911 | 0.94915 | +0.00004 |
| `submission_blend.csv` | 0.94944 | 0.94945 | +0.00001 |
| `submission_final.csv` | 0.94947 | 0.94945 | -0.00002 |

LGBM-only'ye marjinal yardım, blend'e net etki sıfır (noise band). Race-matrix sinyalini hem LGBM hem CatBoost benzer şekilde kullandığı için diversity hafifçe düştü, blend kazancı yendi.

### Ana Çıkarımlar

1. **Variance reduction (multi-seed)** + **regülasyon** + **heterojen feature işleme** üçlüsü tutarlı LB iyileşmesi sağladı.
2. **Blend başarısı feature işleme çeşitliliğine bağlı:** Aynı sinyali farklı işleyen modeller (LGBM OHE+TE × CatBoost native) gerçek diversity ürettir; aynı feature'larla beslenenler overfit eder.
3. **Cross-sectional ranks** time-series rolling'in başarısız olduğu yerde başarılı: yatay (lap-içi) karşılaştırma, sızıntı riski yok.
4. **Compound OHE** ordinal label encoding varsayımını kaldırdı.
5. **Test set kompozisyonu (kontrol edildi):** %100 (Race, Year) seen — `StratifiedKFold` optimal.
6. **Race-matrix aggregation** mantıken doğru ama bu plato'da LB'ye yansımadı — LGBM-only'ye +0.00004, blend'e ±0. Sinyal hem LGBM hem CatBoost tarafından aynı yönde işleniyor, diversity'yi azaltıyor.
7. **NN blend BAŞARISIZ:** embedding'siz MLP, target-encoded kategorical sinyali ağaçlardan daha az verimli ama benzer yönde işliyor → hem korelasyon yüksek hem AUC düşük → blend'e zarar. Tabular'da NN'i kazançlı yapmak için embedding layer + farklı feature işleme gerekiyor (sklearn MLPClassifier yetmiyor).
8. **Plato gerçek:** OOF/LB gap 0.0006'da sabit. Daha fazla model/feature optimizasyonu istatistiksel noise band'ında kalıyor.

---

## Bölüm 9 — Final Submission

Defterin yazdığı CSV'ler (NN diversity check FAIL olduğu için 3'üncüsü yazılmadı):

- **`submission_lgbm.csv`** — yalnız LGBM 10-seed ortalaması (LB: **0.94915**).
- **`submission_blend.csv`** — LGBM × CatBoost 2-way rank-blend (LB: **0.94945**).
- ~~`submission_nn_blend.csv`~~ — yazılmadı (NN diversity FAIL).
- **`submission_final.csv`** — en yüksek OOF olan = `submission_blend.csv` ile aynı (LB: 0.94945).

**Kaggle final-submission önerisi (2 final seçimi):**
- `submission_blend.csv` (2-way, en yüksek LB)
- `submission_lgbm.csv` (tek model, blend farklı sonuç verirse sigorta)

### 9.1 Pipeline Diyagramı

```
Train ──► LGBM × 10 seed avg ──┐
                                ├──► 2-way rank-blend (grid-tuned w) ──► submission_blend.csv
      ──► CatBoost × 3 seed ────┘

      ──► LGBM × 10 seed avg     ─────────────────────────────────────► submission_lgbm.csv

      ──► NN (MLP) × 3 seed       [DEVRE DIŞI — diversity FAIL]
```

### Pipeline Özet Parametreleri

| Bileşen | Konfigürasyon |
|---|---|
| Validasyon | 5-fold `StratifiedKFold` (test ile uyumlu, %100 seen Race×Year ile doğrulandı) |
| LGBM | `num_leaves=63`, `lr=0.02`, `min_data_in_leaf=100`, `lambda_l2=0.5`, `n_estimators=5000`, `early_stopping(100)`, **10 seed** |
| CatBoost | `cat_features=['Driver','Race','Compound']`, `iterations=3000`, `lr=0.03`, `depth=8`, `l2_leaf_reg=5`, **3 seed** |
| NN (MLP) — devre dışı | `hidden=(128, 64)`, `alpha=1e-3`, QuantileTransformer, **3 seed**; diversity FAIL nedeniyle submission'a girmiyor |
| Featurelar | ~22 engineered (4 agg, 3 flag, 2 interaction, 4 cross-sectional rank, **2 race-matrix**, 5 Compound OHE) + 2 target-encoded (Driver, Race) |
| Blend ağırlığı | OOF AUC üzerinde 0.30–0.80 grid search (adım 0.05); optimum w_lgbm ≈ 0.55 |
| Diversity check | NN ↔ LGBM ve NN ↔ CatBoost korelasyonu < 0.95 (FAIL → 3-way devre dışı) |

---

## Final Durum Özeti

- **En iyi LB skorumuz: 0.94947** (race-matrix öncesi 2-way blend) ya da **0.94945** (race-matrix dahil) — istatistiksel olarak özdeş.
- **Net ilerleme (baseline 0.94697 → 0.94947): +0.00250**, 9 kademe geliştirme ile.
- **Plato'da:** OOF/LB gap 0.0006'da sabit, ek ağaç-model optimizasyonları noise band'ında kalıyor.
- **NN yolu kapalı:** sklearn MLPClassifier tabular'da embedding olmadan diversity üretemiyor.
- **Sonraki potansiyel adımlar (denenmedi):** per-year target encoding (2023 hariç), Optuna ile ağırlık optimizasyonu, NN için PyTorch + categorical embedding layer. Plato gözleminde beklenti hepsi için düşük.
