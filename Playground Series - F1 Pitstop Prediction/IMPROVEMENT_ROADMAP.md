Önemli ön düzeltme — Compound aslında atılmadı
Bölüm 8.1'de drop_from_X = ['id', 'Driver', 'Compound', 'Race', 'PitNextLap', 'Driver_enc', 'Race_enc'] şu anki kullanımı: ham string Compound atıldı ama formula1_pit_stop_prediction.ipynb Bölüm 7.6'da üretilen Compound_enc (label-encoded) modelde var — features_base listesine giriyor. Yani sinyal mevcut, sadece ordinal kodlanmış (SOFT=2, HARD=0, vb.) — bu da yanlış bir sıralama varsayımı yapıyor. Önerin yine de geçerli: OHE bu varsayımı kaldırır.

1. Compound — OHE'ye geç
formula1_pit_stop_prediction.ipynb Bölüm 7.6'da Compound_enc üretimini kaldırıp yerine OHE:


# Bölüm 7.6 yerine
compound_dummies = pd.get_dummies(all_df['Compound'], prefix='Compound').astype(int)
all_df = pd.concat([all_df, compound_dummies], axis=1)
Bölüm 8.1'de drop_from_X listesinden Compound_enc'i çıkar, OHE kolonları otomatik feature listesine düşer. 5 ek binary kolon — kardinalite çok düşük, regülarizasyon etkisi yok.

2. CatBoost'u doğru beslemek + ensemble
Yeni Bölüm 8.9 olarak ekle (LGBM 10-seed bittikten sonra, blend için):


from catboost import CatBoostClassifier

cat_features = ['Driver', 'Race', 'Compound']  # ham string
X_cb       = train_fe[features_base + cat_features].drop(columns=['Compound_enc']).copy()
X_cb_test  = test_fe[features_base + cat_features].drop(columns=['Compound_enc']).copy()
# Driver/Race ham string olduğu için str_train/str_test'ten alıp birleştirmek gerekir

cb_oof, cb_test = np.zeros(len(X_cb)), np.zeros(len(X_cb_test))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_cb, y), 1):
    model = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=8,
        l2_leaf_reg=5, eval_metric='AUC',
        cat_features=cat_features, random_state=RANDOM_STATE, verbose=0
    )
    model.fit(X_cb.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=(X_cb.iloc[val_idx], y.iloc[val_idx]),
              early_stopping_rounds=100)
    cb_oof[val_idx] = model.predict_proba(X_cb.iloc[val_idx])[:, 1]
    cb_test += model.predict_proba(X_cb_test)[:, 1] / skf.n_splits

# Rank-blend (LB'de daha stabil)
from scipy.stats import rankdata
blend_test = 0.6 * rankdata(test_preds_seedavg) + 0.4 * rankdata(cb_test)
blend_test /= len(blend_test)
Önemli: korelasyonu önce OOF'ta ölçmen gerekir (np.corrcoef(oof_seedavg, cb_oof)); 0.95'in altıysa diversity gerçek, üstüyse blend yine LB'ye yansımayabilir.

3. Hiperparametre regülasyonu
Bölüm 8.2'de lgb_params güncellemesi:


lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02,         # 0.05 → 0.02
    'num_leaves': 63,              # 127 → 63
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 100,       # 20 → 100
    'lambda_l1': 0.1,
    'lambda_l2': 0.5,              # 0.1 → 0.5 (hafif artır)
    'verbose': -1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
}
# ve
model = lgb.LGBMClassifier(**lgb_params, n_estimators=5000)  # 2000 → 5000
Maliyet: lr yarıya inince eğitim ~2-3x uzar, 10-seed × 5-fold için ~60-70 dk. Tek seedde önce dene, OOF AUC düşmüyor/artıyorsa 10-seed'e geç.

4. Cross-sectional (lap-içi göreceli) feature'lar
Bölüm 7.5 (interactions) altına:


lap_grp = all_df.groupby(['Race', 'Year', 'LapNumber'])

all_df['TyreLife_pct_vs_lap_mean'] = (
    all_df['TyreLife_pct'] / lap_grp['TyreLife_pct'].transform('mean')
)
all_df['TyreLife_rank_in_lap']    = lap_grp['TyreLife'].rank(pct=True)
all_df['CumDeg_rank_in_lap']      = lap_grp['Cumulative_Degradation'].rank(pct=True)
all_df['LapTime_rank_in_lap']     = lap_grp['LapTime (s)'].rank(pct=True)
all_df['Position_vs_grid_size']   = all_df['Position'] / lap_grp['Position'].transform('max')
Bunlar undercut/overcut sinyali. Time-series rolling overfit ettiyse, cross-sectional rank'ler genelde daha güvenli — leak riski yok (aynı turdaki diğer sürücüler test setinde de gözlemleniyor).

5. Validation — Public/Private LB ikili strateji
Forumdan bölünme stratejisini kontrol etmen şart, ama koda iki sigorta ekleyebiliriz:


# A) Mevcut: StratifiedKFold ile en iyi best_iter ortalaması belirle
mean_best_iter = int(np.mean([m.best_iteration_ for m in fold_models]))

# B) Sigorta submission: GroupKFold + sabit iter (full-train final model)
gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
group_oof = np.zeros(len(X_base))
for tr_idx, val_idx in gkf.split(X_base, y, groups):
    # ... aynı pipeline ama n_estimators=mean_best_iter, early_stopping YOK
    pass
group_auc = roc_auc_score(y, group_oof)
print(f'GroupKFold OOF AUC: {group_auc:.5f}  (Private LB sigortası)')
İki submission hazırla: submission_skf.csv (mevcut, Public'e oynak) + submission_groupcv.csv (Private sigortası). Kaggle 2 final submission seçtiriyor — birini her stratejiye ayır.

Hangi sırayla denemeni öneririm
Sıra	Aksiyon	Risk	Beklenen kazanç
1	Compound OHE	Düşük	Ufak ama bedava
2	Cross-sectional ranks (madde 4)	Düşük-orta	Orta — undercut sinyali
3	Hiperparam reg. (madde 3)	Düşük	OOF/LB gap'i küçültür
4	CatBoost + rank-blend (madde 2)	Orta	OOF korelasyonu < 0.95 ise yüksek
5	Çift-validasyon submission (madde 5)	Sıfır	Private LB sigortası
Hangisinden başlamamı istersin? Ben madde 1+4 ikilisinin en hızlı geri dönüş vereceğini düşünüyorum (kod değişikliği küçük, training maliyeti aynı). 2 ve 3 birlikte denenmeli — yeni feature + daha az leaves = doğal tamamlayıcı. 5 ise sadece notebook'un sonuna ek hücre.