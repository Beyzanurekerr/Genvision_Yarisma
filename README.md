# GenVision (Takım NOVA, 918091) — Kurulum ve Çalıştırma

## 1. Kurulum

Proje klasörünü bilgisayarına al:
- GitHub'dan: `git clone https://github.com/Beyzanurekerr/Genvision_Yarisma.git`
- ya da zip/USB ile kopyala.

Python kurulu mu kontrol et (PowerShell'de):
```
python --version
```
Sürüm numarası çıkmazsa python.org/downloads'tan indirip kur ("Add python.exe
to PATH" kutusunu işaretle).

## 2. Requirements / Ortam Kurulumu

Proje klasörünün içinde:
```
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```
Bu, sadece BİR KEZ (internet varken) yapılır. Bu bilgisayarda (Casper'ın
bilgisayarı) zaten yapıldı, `venv\` klasörü hazır — bu adımı atlayabilirsin.
Başka bir bilgisayarda ilk kez kuruyorsan bu iki komutu çalıştır.

## 3. Test Verilerini Projeye Koymak

USB'den gelen (ya da denemek için `data\test\` klasöründeki sahte) test
dosyalarını şu klasöre kopyala:
```
data\final_test\
```
(Önceden bir şey koyduysan, yenilerini koymadan önce içindeki eski .csv
dosyalarını sil — `README.md` dosyasına dokunma, o klasörün GitHub'da var
olabilmesi için orada duruyor, silersen bir şey bozulmaz ama gerek yok.)

## 4. Dosya İçine Girmek

Proje klasörünü Dosya Gezgini'nde aç, adres çubuğuna `powershell` yaz, Enter'a
bas — PowerShell doğrudan orada açılır. Sonra:
```
cd src
```

## 5. Çalıştırıp JSON'a Kaydetmek

```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json
```

Ekranda her panelin (MASTER/KANSER/PAH/CFTR) hangi dosyayla eşleştiğini
yazar. Sonunda:
```
Doğrulama OK: ... tahmin, ... benzersiz (panel,id).
Kaydedildi: ...TEAM_918091_FINAL.json ...
```
görürsen, `TEAM_918091_FINAL.json` proje kök klasöründe hazırdır. Bu dosyayı
organizasyonun web sistemine yükle — teslim budur.

## 6. Olası Durumlar (internet YOK, kendi başına karar vermen gerekecek)

**Panel eşleşmesi yanlış/eksik çıktıysa** (ekranda `eşleşme YOK` ya da
`BELİRSİZ` yazıyorsa) → dosyaları elle belirt:
```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --master <yol> --kanser <yol> --pah <yol> --cftr <yol>
```
(`<yol>` yerine dosyanın gerçek yolunu yaz; dosya adı önemli değil, hangi
dosyayı hangi panele verdiğin önemli.)

**Dosya yolunu bulmanın en kolay yolu (yazmana bile gerek yok):**
Terminalde `--master ` yazdıktan sonra (boşluk bırak, Enter'a basma), Dosya
Gezgini'nde o dosyayı bul ve **fare ile terminal penceresinin üzerine
sürükle bırak** — dosyanın tam yolu otomatik, tırnak içinde yapıştırılır.
Aynısını `--kanser`, `--pah`, `--cftr` için de yap.

*(İstersen elle de yazabilirsin: dosyaya Dosya Gezgini'nde SAĞ TIKLA →
"Yol olarak kopyala" / "Copy as path" seç, sonra terminale yapıştır.)*

**4 ayrı dosya değil, TEK dosyada "Panel" sütunu varsa:**
```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --input <tek dosya yolu>
```

**`ModuleNotFoundError` hatası** → yanlış Python kullanıyorsun, mutlaka
`venv\Scripts\python.exe` ile çalıştır (sistem Python'ı değil).

**`InconsistentVersionWarning` yazısı** → zararsız bir uyarı, göz ardı et,
sonucu bozmaz.

**"Sütun eksik" hatası** → hangi sütunların eksik olduğunu isim isim
yazar; doğru dosyayı/paneli verdiğinden emin ol.

**Ayraç/ondalık virgül/farklı ID sütun adı** (Excel'den farklı kaydedilmiş
dosya) → script bunu otomatik algılayıp düzeltir, ekstra bir şey yapmana
gerek yok, konsola ne yaptığını yazar.

**Hiçbiri uymuyorsa** → komutu AYNEN bir daha dene (bazen tek seferlik bir
yazım hatasıdır); olmuyorsa yukarıdaki "elle belirtme" komutuyla dene.
**`uret.py`'yi ÇALIŞTIRMA** — adına rağmen işe yaramaz, sabit/sahte
dosyalarla sınırlı.

**Birden fazla yükleme yapabilirsin** — süre bitene kadarki SON geçerli
JSON dosyası teslim sayılır, ilk denemede hata olması sorun değil.

## 7. Proje Dosyaları — Ne İşe Yararlar

Hepsi `..\venv\Scripts\python.exe <dosya>` ile çalıştırılır (önce `cd` ile o
dosyanın bulunduğu klasöre girmen gerekir).

### `src\` (asıl kullanılan pipeline)

| Dosya | Ne işe yarar | Çalıştırma |
|---|---|---|
| `predict_final.py` | **Final tahmin/JSON üretici** — bu README'nin konusu. | `predict_final.py --out ..\TEAM_918091_FINAL.json` |
| `features.py` | Öznitelik mühendisliği (Grantham mesafesi, eksiklik sinyalleri, kategorik kodlama). Kendi başına çalıştırılmaz, diğer dosyalar kullanır. | — |
| `tune.py` | Optuna ile hiperparametre arama (en iyi ayarları bulur). | `tune.py` |
| `train_final.py` | Ayarlanmış hiperparametrelerle çapraz doğrulama (CV/OOF) sonuçları üretir. | `train_final.py [PANEL]` |
| `train_deploy.py` | Seçilen modeli TÜM veriyle eğitip `models\deploy\*.joblib` paketlerini üretir. | `train_deploy.py` |
| `check_determinism.py` | Aynı veriyi iki kez tahmin ettirip sonucun birebir aynı çıktığını doğrular. | `check_determinism.py` |
| `guard_scan.py` | Kalibrasyon kayması güvenlik ağının (GUARD_HI/LO) en iyi değerini tarar/ölçer. | `guard_scan.py` |
| `overfit_check.py` | Aşırı öğrenme/veri sızıntısı denetimi (train-OOF boşluğu + etiket karıştırma testi). | `overfit_check.py [PANEL]` |
| `seed_stability.py` | Farklı rastgele tohumlarla sonucun ne kadar kararlı olduğunu ölçer. | `seed_stability.py` |
| `shap_analysis.py` | SHAP ile hangi özniteliğin kararı ne kadar etkilediğini analiz eder. | `shap_analysis.py` |
| `plot_figures.py` | Rapor için grafik/görsel üretir. | `plot_figures.py` |
| `generate_report.py` | Proje detay raporu (PDR) metnini otomatik üretir. | `generate_report.py` |
| `train.py` | Optuna'dan önceki temel/ilk eğitim scripti (artık `tune.py`+`train_final.py` kullanılıyor). | `train.py` |

### Proje kökü

| Dosya | Ne işe yarar | Çalıştırma |
|---|---|---|
| `uret.py` | **Final günü KULLANMA** — sadece `data\test\` içindeki sabit/sahte dosyalarla hızlı yerel deneme. | `uret.py` |
| `generate_dummy_test.py` | `data\test\` klasörü için eğitim verisinden 50 satırlık, etiketsiz sahte test dosyaları üretir. | `generate_dummy_test.py` |

### `Eski_Deneysel_Kodlar\` (çoğu arşiv/deney — bir istisna var)

| Dosya | Ne işe yarar | Çalıştırma |
|---|---|---|
| `threshold.py` | **DİKKAT — bu aslında aktif pipeline'ın parçası** (arşiv klasöründe duruyor ama gerçekten kullanılıyor): test-bilinçli eşik kalibrasyonu + panel başına model seçimi. `train_final.py`'den sonra, `train_deploy.py`'den ÖNCE çalıştırılmalı. | `threshold.py` |
| `fetch_clinvar_grantham.py` | ClinVar'dan Grantham-mesafe/patojenite ilişkisini öğrenir (dış veri denemesi). | `fetch_clinvar_grantham.py` |
| `ext_grantham_ablation.py` | Yukarıdakini kendi verimizde test eder — sonuç: eşiği geçemedi, modele eklenmedi. | `ext_grantham_ablation.py` |
| `leakage_check.py` | Sızıntı denetimi: tek bir özniteliğin tek başına etiketi "ele verip vermediğini" tarar. | `leakage_check.py` |
| `check_determinism.py` | `src\check_determinism.py`'nin eski/orijinal kopyası. | — (güncel olan `src\`'deki) |
| Diğerleri (`main.py`, `run_all.py`, `predict_test.py`, `check_*.py`, `smote_experiment.py`, `visualize_mapper.py`, `catboost_screen.py`) | Çeşitli deneysel kontrol/analiz scriptleri — aktif final pipeline'ının parçası DEĞİL, geliştirme sürecinde kullanıldı. | — |

### `external_validation\`

ClinVar+gnomAD'dan bağımsız dış test seti üretimi. Ayrıntı: `external_validation\README.md`.

| Dosya | Ne işe yarar | Çalıştırma |
|---|---|---|
| `01_filtre.py` | ClinVar verisini indirip panel-gen eşlemesine göre filtreler. | `01_filtre.py` |
| `02_gnomad_benign.py` | PAH/CFTR için gnomAD'dan sık görülen (muhtemelen benign) varyantları çeker. | `02_gnomad_benign.py` |
| `03_stres_testi_kur.py` | Panel başına, resmi şartnameye uygun oranlı final dış test setini kurar. | `03_stres_testi_kur.py` |
