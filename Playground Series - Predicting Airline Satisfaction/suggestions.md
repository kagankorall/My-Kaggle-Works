# Havayolu Memnuniyet Tahmini - Darboğaz Çözüm Önerileri

## 1. Model Çeşitliliğini Artırma (Darboğaz 1)
Ağaç tabanlı modellerin (LightGBM, XGBoost, CatBoost) tahminlerinin 0.990-0.995 seviyesinde çok yüksek korelasyona sahip olması[span_1](start_span)[span_1](end_span), uyguladığın sıralama ortalamasının (rank blend) getirisini sınırlıyor[span_2](start_span)[span_2](end_span). Tüm özellikleri kategorik olarak CatBoost'a verme denemen model çeşitliliği sağlamak adına çok mantıklı bir adım[span_3](start_span)[span_3](end_span). Buna ek olarak:
* **Farklı Mimariler:** Ağaç tabanlı olmayan modelleri (örneğin TabNet, basit bir PyTorch MLP veya Keras tabanlı bir sinir ağı) denemek, ağaçların zorlandığı farklı yolcu profillerinde (örneğin iş/ekonomi sınıfı geçişlerindeki sınır vakalarında) daha iyi sıralama yapmanı sağlayabilir.
* **K-Nearest Neighbors (KNN):** Orijinal anket verilerindeki "0" değerlerinin bir derecelendirme olmadığını ve wifi hizmeti gibi dijital alanlarda U-şekilli bir etki yarattığını keşfetmişsin[span_4](start_span)[span_4](end_span). Bu tür doğrusal olmayan ve threshold odaklı yapıları, iyi ölçeklendirilmiş bir KNN veya SVM modeli ağaçlardan farklı bir şekilde ağırlıklandırabilir.

## 2. En İyi Notebook ile Aradaki Fark (Darboğaz 2)
`najiama` isimli notebook ile arandaki açıklanamayan ~0.0008'lik OOF farkı[span_5](start_span)[span_5](end_span), yarışmanın bu aşamasında oldukça kritik.
* **Sözde Etiketleme (Pseudo-labeling):** Sentetik Playground yarışmalarında en tepeye yerleşen modeller genellikle test setine yüksek güvenle yapılan tahminleri (pseudo-labels) eğitim setine dahil eder. Özellikle özellik dağılımının ve drift oranının çok düşük olduğu bu veri setinde[span_6](start_span)[span_6](end_span) pseudo-labeling çok iyi çalışabilir.
* **Etkileşim Özellikleri (Interaction Features):** `Online boarding` özelliğinin tek başına çok güçlü olduğunu (AUC ≈ 0.84) ve `Type of Travel` x `Class` etkileşiminin iyi çalıştığını analiz etmişsin[span_7](start_span)[span_7](end_span). Karar ağaçlarının bulmasını beklemeden `Online boarding` skoru ile diğer hizmetlerin (örneğin Inflight Wifi) arasındaki farkları veya oranları açık (explicit) bir özellik olarak modele verebilirsin.

## 3. Çalışma Süresi ve İş Akışı Otomasyonu (Darboğaz 4 & 5)
CatBoost'un 28 dakikalık eğitim süresi[span_8](start_span)[span_8](end_span) iterasyon hızını yavaşlatıyor. Rapor tasarımı ve iş akışı otomasyonu yeteneklerini buradaki pipeline'ı hızlandırmak için kullanabilirsin:
* **TE (Target Encoding) Önbellekleme:** Hedef kodlamalarının (fold encodings) her model için ayrı ayrı baştan hesaplanması[span_9](start_span)[span_9](end_span) ciddi bir hesaplama israfı yaratıyor. Fold döngüsü içinde oluşturulan `te` ve `te_pair` verilerini geçici bir klasörde (örn. `cached_folds/`) `.parquet` formatında diske kaydedip diğer modellerin doğrudan buradan okumasını sağlayan bir otomasyon yazabilirsin.
* **CatBoost Hızlandırması:** Eğer donanımın uygunsa `task_type='GPU'` parametresi CatBoost'un süresini dakikalara indirecektir. Sadece CPU kullanılıyorsa, doğruluğu çok bozmadan `rsm` (colsample_bylevel) veya `border_count` (max_bin) parametrelerini bir miktar düşürmeyi test edebilirsin.

## 4. LightGBM Özellik Önem Grafiği (Darboğaz 6)
Modelleri yeniden yüklerken importances değerlerinin tahminlerle beraber kaydedilmemesi nedeniyle grafiklerin kaybolması problemini[span_10](start_span)[span_10](end_span) çözmek için:
* Model eğitimini tamamladıktan ve OOF tahminlerini kaydettikten hemen sonra `booster.feature_importance()` çıktılarını basit bir JSON dosyası olarak (örneğin `preds/lgb_importances.json`) kaydet.
* Notebook'u yeniden başlattığında, modeli baştan eğitmek yerine `.npy` dosyasını ve bu JSON dosyasını okutup seaborn/matplotlib fonksiyonunu doğrudan çağırabilirsin.

## 5. Karar Mekanizması (Darboğaz 3)
Gelişimlerin LB üzerinde çok küçük farklılıklar yaratması ve OOF-LB sapmasının gönderimden gönderime oynaması nedeniyle[span_11](start_span)[span_11](end_span) kararları CV üzerinden verme stratejin en sağlıklı yoldur[span_12](start_span)[span_12](end_span). Bu süreçte sadece fold'ların ortalama OOF AUC'sine değil, 10 fold arasındaki standart sapmaya da bakarak (varyansı düşüren özellikleri tercih ederek) daha stabil bir model elde edebilirsin.
