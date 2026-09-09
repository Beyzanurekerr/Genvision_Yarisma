# GenVision — Ne Var, Nasıl Çalıştırılır?

**Takım NOVA** (Takım ID 918091, Başvuru ID 4885814) — TEKNOFEST 2026 Sağlıkta
Yapay Zeka Yarışması, Üniversite ve Üzeri Seviyesi. PSR ve PDR teslim edildi;
bu paket **final turu** (16-18 Eylül 2026, Dicle Üniversitesi) için hazırlanan
GÜNCEL modelleme sistemidir.

## Hızlı özet (5 dakikada ne olduğunu anlamak için)

- 4 panel (MASTER/KANSER/PAH/CFTR), her biri için ayrı model — yarışma şartı.
- Final günü **internet YOK**, model önceden hazır olmalı → `models/deploy/*.joblib`
  bunun için var, `predict_final.py` final JSON'unu üretiyor.
- Final resmi skor formülü: panel başına (F1+MCC)/2, sonra 4 panelin ortalaması.
- **Şu anki tahmini skor: 0.580** (bkz. aşağıdaki tablo) — `models/threshold_results.json`
  içinde tüm detay var.
- `reports/GenVision_Rapor.docx` **ESKİ/GÜNCEL DEĞİL** — bu paket sonrası yapılan
  tüm iyileştirmeler (CatBoost, ağırlıklı blend, dış veri denemesi) rapora
  yansıtılmadı. Rapor gerekirse `src/generate_report.py` yeniden çalıştırılmalı.

## Klasörler
- `data/` — yarışmadan gelen 4 panelin eğitim verisi
- `src/` — Python kodları (aşağıda ne işe yaradıkları var, ÇALIŞTIRMA SIRASIYLA)
- `models/` — kodları çalıştırınca üretilen SONUÇLAR (JSON dosyaları, sayılar) +
  `models/deploy/` (final günü kullanılacak, diske kaydedilmiş hazır modeller)
- `reports/` — veri sözlüğü tabloları, şekiller, örnek JSON teslim dosyası

## Kurulum (bilgisayarınızda ilk kez çalıştırırken)

```
pip install -r requirements.txt
```
Yalnızca final günü çalıştırmak (eğitim değil, tahmin) için minimal kurulum yeterli:
```
pip install -r requirements-inference.txt
```

## Kodlar hangi sırayla çalışır?

1. **`features.py`** — kod değil, diğer dosyaların kullandığı "araç kutusu" (eksik veri
   işleme, amino asit mesafesi hesaplama, AL×EK etkileşim özellikleri vb.). Tek başına
   çalıştırılmaz.

2. **`leakage_check.py`** — Veri sızıntısı var mı diye kontrol eder. Temiz çıktı =
   güvenilir sonuç demek.
   ```
   cd src && python leakage_check.py
   ```

3. **`tune.py`** — Modellerin ayarlarını (hiperparametre, Optuna ile) otomatik dener.
   Panel bazlı veya hepsi birden çalıştırılabilir (uzun sürebilir, MASTER en uzun):
   ```
   cd src && python tune.py            # tüm paneller, XGB+LGB+RF+CatBoost
   cd src && python tune.py MASTER     # yalnızca bir panel
   cd src && python tune.py --cat-only # yalnızca CatBoost'u arar, diğerlerini korur (hızlı)
   ```

4. **`train_final.py`** — tune.py'nin bulduğu en iyi ayarlarla SON eğitimi (3 seed × 5
   kat CV) yapar; 4 temel model (XGBoost/LightGBM/RandomForest/CatBoost) + Stacking +
   tüm ikili/üçlü/dörtlü blend kombinasyonlarının OOF tahminlerini üretir:
   ```
   cd src && python train_final.py
   ```

5. **`threshold.py`** — panel başına, TÜM aday modeller (4 temel + Stacking + sabit
   blend'ler + ağırlık-taranmış blend'ler) arasından, final resmi formülüne
   ((F1+MCC)/2, test prevalansına projekte edilmiş) göre EN İYİSİNİ seçer + eşiğini
   bulur + 2000 tekrarlı bootstrap ile doğrular:
   ```
   cd src && python threshold.py
   ```
   Çıktı: `models/threshold_results.json` — **en önemli sonuç dosyası**.

6. **`smote_experiment.py`** — sentetik veri çoğaltmanın (SMOTE) işe yarayıp yaramadığını
   test eder. Sonuç: yalnızca CFTR'de anlamlı katkı, diğerlerinde yok/zararlı —
   bu yüzden production'a eklenmedi.
   ```
   cd src && python smote_experiment.py
   ```

7. **`shap_analysis.py`** — hangi bilginin kararı en çok etkilediğine bakar.
   ```
   cd src && python shap_analysis.py
   ```

8. **`seed_stability.py`** — panel başına seçilen modeli, sabit test-eşiğinde 3 seed'in
   HER BİRİ için ayrı ayrı çalıştırıp Ortalama ± Std raporlar (3-seed bagged sonuca ek
   bir kararlılık kanıtı). CFTR'nin diğer panellere göre çok daha kararsız olduğunu
   gösteriyor (küçük örneklem, n=111) — final günü riskini bilerek kabul ettik.
   ```
   cd src && python seed_stability.py
   ```

9. **`plot_figures.py`** — rapor/sunum için 3 yüksek çözünürlüklü şekil üretir.
   ```
   cd src && python plot_figures.py
   ```

10. **`fetch_clinvar_grantham.py`** + **`ext_grantham_ablation.py`** — dış veri (ClinVar)
    denemesi: Grantham mesafesi → gerçek ClinVar patojenite ilişkisini öğrenip özellik
    olarak test eder. **SONUÇ: net katkı yok** (KANSER/PAH'ta zararlı), production'a
    eklenmedi — dürüst bir negatif sonuç olarak burada duruyor.
    ```
    cd src && python fetch_clinvar_grantham.py   # internet gerektirir
    cd src && python ext_grantham_ablation.py
    ```

11. **`train_deploy.py`** — KRİTİK (final günü için): panel başına SEÇİLEN modeli
    TÜM eğitim verisiyle yeniden eğitip diske kaydeder. Final günü internet YOK,
    model önceden hazır olmalı — 1-10 arasındaki adımlar sadece PERFORMANS ÖLÇÜYOR
    (CV/OOF), gerçek teslim edilecek modeli bu script üretiyor:
    ```
    cd src && python train_deploy.py
    ```
    Çıktı: `models/deploy/{PANEL}_deploy.joblib` (4 dosya, zaten üretilmiş halde bu
    pakette mevcut).

12. **`predict_final.py`** — final kılavuzundaki ZORUNLU JSON şemasında teslim dosyası
    üretir. Gerçek test verisi geldiğinde:
    ```
    cd src && python predict_final.py --out ../TEAM_918091_FINAL.json \
        --master <yol> --kanser <yol> --pah <yol> --cftr <yol>
    ```
    Gerçek veri gelmeden ÖNCE format+uçtan-uca doğrulama için:
    ```
    cd src && python predict_final.py --dry-run
    ```
    ÖNEMLİ: `--dry-run` çıktısındaki F1/MCC eğitim verisi üzerinde olduğu için
    iyimserdir — sadece boru hattının uçtan uca çalıştığını doğrular, gerçek
    performans göstergesi DEĞİLDİR.

13. **`generate_report.py`** — (ŞU AN GÜNCEL DEĞİL, bkz. yukarı) `models/*.json` +
    `reports/*.csv` + `reports/figures/*.png` okuyup tek bir `.docx` raporu üretir.
    ```
    cd src && python generate_report.py
    ```

## Final günü kontrol listesi (16 Eylül 2026, internet YOK)

- [ ] `pip install -r requirements-inference.txt` bu bilgisayarda ÖNCEDEN çalıştırılmış olmalı
- [ ] `models/deploy/*.joblib` (4 dosya) diskte hazır — ✅ bu pakette zaten mevcut
- [ ] `python predict_final.py --dry-run` en az bir kez sorunsuz çalışmış olmalı
- [ ] `TEAM_INFO` (predict_final.py içinde) KYS'deki takım adı/ID/başvuru ID ile
      final günü teyit edilmeli (şu an: NOVA / 918091 / 4885814)
- [ ] Gerçek test CSV'leri geldiğinde: `predict_final.py --master ... --kanser ... --pah ... --cftr ...`
      → çıktı JSON'u yükleme sistemine yükle

## Sonuçları nerede görürüm?

- `models/threshold_results.json` — **en güncel, en önemli sonuç tablosu** (hangi
  model, hangi eşik, tahmini test F1/MCC/(F1+MCC)/2, bootstrap güven aralığı)
- `models/seed_stability.json` — kararlılık/varyans kanıtı
- `models/final_cv_results.json` — tüm adayların (XGB/LGB/RF/CatBoost/Stacking/
  tüm blend kombinasyonları) ham 0.5-eşiğindeki CV performansı
- `models/shap_summary.json` — hangi bilgi kategorisi kararı ne kadar etkiliyor
- `models/smote_experiment.json`, `models/clinvar_grantham_*` — denenip
  production'a ALINMAYAN yaklaşımlar (dürüst negatif sonuçlar)

## Şu anki en güncel sonuçlar (özet)

| Panel  | Seçilen Model     | Karar Eşiği | Tahmini Test F1 | Tahmini Test MCC | (F1+MCC)/2 |
|--------|-------------------|-------------|------------------|-------------------|------------|
| MASTER | WBlend_RC_0.6     | 0.75        | 0.534            | 0.451             | 0.492      |
| KANSER | WBlend_XC_0.3     | 0.86        | 0.678            | 0.612             | 0.645      |
| PAH    | LightGBM          | 0.75        | 0.629            | 0.467             | 0.548      |
| CFTR   | Blend_XGB_CAT     | 0.93        | 0.626            | 0.641             | 0.633      |

**Ortalama (resmi final formülü): 0.580** — başlangıç değeri 0.546'ydı (+0.034, %6.2
göreli artış). "WBlend_XY_w" = X ve Y modellerinin w/(1-w) ağırlıklı ortalaması
(X=XGBoost, R=RandomForest, C=CatBoost, L=LightGBM). "Tahmini" diyoruz çünkü gerçek
yarışma test verisini henüz görmedik — bu, eğitim verisinden istatistiksel olarak
projekte edilmiş ve bootstrap+seed-varyansıyla doğrulanmış bir değer.

**CFTR notu:** En yüksek beklenti ama en yüksek kararsızlık burada (seed-arası
sonuçlar 0.40-0.79 arası değişiyor, n=111 küçük örneklem). Bilinçli bir risk kararı
olarak yüksek-beklenti seçeneği korundu (final formülü tek nokta tahminini
maksimize ediyor, varyansı cezalandırmıyor).
