# Genvision - TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması

**Takım Adı:** NOVA  
**Takım ID:** 918091  
**Yarışma Seviyesi:** Üniversite ve Üzeri  


## Proje Hakkında
TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması için geliştirilen **Genvision** projesinin kaynak kodlarını ve çevrimdışı (offline) çalışma mimarisini barındırmaktadır. Proje, genetik varyantların patisite sınıflandırmasını (missense variant pathogenicity classification) yüksek doğruluk, kararlılık ve açıklanabilirlik ilkeleriyle gerçekleştiren endüstriyel standartta bir Stacking Ensemble pipeline'ıdır.

---

## Klasör Yapısı
Proje, tamamen modüler, taşınabilir ve internet erişimi olmayan (offline) ortam koşullarına uygun olarak tasarlanmıştır:

```text
Genvision_Yarisma/
│
├── veri/
│   ├── egitim/          # Eğitim panelleri (MASTER, KANSER, PAH, CFTR CSV dosyaları)
│   └── test/            # Test verilerinin konumlandırıldığı dizin (USB'den gelen veriler buraya atılır)
│
├── src/
│   ├── __init__.py
│   ├── external_mapper.py # Bayesian Bio-Signature Mapping dış veri modülü
│   └── model_trainer.py   # Stacking Ensemble model eğitim ve değerlendirme mantığı
│
├── sonuclar/            # Üretilen final JSON çıktılarının ve grafiklerin kaydedildiği dizin
│
├── check_missing_data.py  # Kolon ve satır bazlı eksik veri analiz scripti
├── generate_dummy_test.py # Çevrimdışı test provası için sahte veri üretici
├── predict_test.py        # Resmi TEKNOFEST JSON çıktısını üreten ana inference scripti
├── main.py                # Model eğitim, çapraz doğrulama ve feature importance scripti
└── requirements.txt       # Gerekli Python kütüphaneleri ve bağımlılık listesi
```
## Kullanılan Algoritma ve Mimari

Projede, genetik verilerin gürültülü, yüksek boyutlu ve seyrek (sparse) yapısıyla başa çıkabilmek amacıyla tek bir modele dayalı yaklaşım yerine hibrit bir makine öğrenmesi mimarisi tercih edilmiştir.

### 1. Stacking Ensemble — XGBoost + LightGBM

Karar mekanizmasında iki güçlü gradient boosting algoritması birlikte kullanılmıştır:

XGBoost ve  LightGBM

Modeller optimize edilmiş hiperparametrelerle eğitilmiş ve nihai tahmin aşamasında ağırlıklı bir ensemble yaklaşımı uygulanmıştır.

Model ağırlıkları: *XGBoost  → %60* *LightGBM → %40*

Bu yaklaşım, modellerin farklı karar sınırlarından yararlanarak tahmin kararlılığını artırmayı amaçlamaktadır.
### 2. Bayesian Bio-Signature Mapping

Dış veri modülü kullanılarak genetik imzaların (bio-signatures) hastalıklarla ilişkisi haritalandırılmıştır.

Elde edilen bilgiler modele **Bayesgil önsel (prior probability)** katkısı sağlayacak şekilde karar mekanizmasına dahil edilmiştir.

Bu sayede model yalnızca eğitim verisindeki örüntülere değil, aynı zamanda dış biyolojik bilgi kaynaklarından elde edilen olasılıksal bilgilere de dayanmaktadır.



### 3. LabelEncoder & Sparsity-Aware Pipeline

Genetik veri setinde bazı kolonlarda **%90'ın üzerinde eksik veri** bulunabilmektedir. Bu nedenle eksik verilerin doğrudan satır veya kolon silme yöntemiyle ortadan kaldırılması yerine, veri kaybını minimize eden bir yaklaşım kullanılmıştır.

Eksiklik durumunu temsil etmek amacıyla satır bazlı eksiklik sayaçları oluşturulmuştur:

```text
AL_missing_count
EK_missing_count
```

Kategorik değişkenler ise güvenli dönüşüm ve kodlama işlemlerinden geçirilerek modele uygun hale getirilmiştir.

Bu yapı sayesinde yüksek sparsity oranına sahip genetik verilerin mümkün olduğunca büyük bir bölümü modelleme sürecinde korunmuştur.

---

## Model Performansı ve Sonuç Metrikleri

Modellerin başarısı **Cross-Validation (Çapraz Doğrulama)** süreçleri üzerinden değerlendirilmiştir.

Kullanılan temel performans metrikleri:

* **Macro F1**
* **Matthews Correlation Coefficient (MCC)**
* **PR-AUC**
* **Brier Score**

### Cross-Validation Sonuçları

| Panel      | Macro F1 |    MCC | PR-AUC | Brier Score |
| ---------- | -------: | -----: | -----: | ----------: |
| **MASTER** |   0.9610 | 0.9242 | 0.9973 |      0.0268 |
| **KANSER** |   0.9909 | 0.9819 | 0.9998 |      0.0067 |
| **PAH**    |   0.9858 | 0.9719 | 0.9999 |      0.0072 |
| **CFTR**   |   1.0000 | 1.0000 | 1.0000 |      0.0259 |

> **Not:** Bu metrikler çapraz doğrulama sonuçlarını göstermektedir. Özellikle CFTR panelindeki 1.0000 değerleri, ilgili veri bölümlerinde modelin sınıflandırmayı tamamen doğru gerçekleştirdiğini göstermektedir; bağımsız dış test performansı olarak yorumlanmamalıdır.

---

# Kurulum ve Çalıştırma

Projeyi yerel ortamda çalıştırmak için aşağıdaki adımlar izlenebilir.

## 1. Sanal Ortam Oluşturma

Öncelikle proje dizininde bir Python sanal ortamı oluşturun:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

Sanal ortam başarıyla aktifleştirildiğinde terminalde `(venv)` ibaresi görülmelidir.

---

## 2. Bağımlılıkların Yüklenmesi

Projenin ihtiyaç duyduğu Python paketleri `requirements.txt` dosyasında tanımlanmıştır.

Bağımlılıkları yüklemek için:

```bash
pip install -r requirements.txt
```

> Proje offline çalıştırılacaksa gerekli paketlerin önceden indirilmiş olması ve ilgili Python ortamında erişilebilir durumda bulunması gerekir.

---

## 3. Eksik Veri Analizi

Veri setindeki eksik değer oranlarını ve sparsity yapısını incelemek için:

```bash
python check_missing_data.py
```

Bu adım, özellikle yüksek oranda eksik veri içeren kolonların tespit edilmesi ve veri ön işleme sürecinin kontrol edilmesi için kullanılmaktadır.

---

## 4. Model Eğitimi ve Çapraz Doğrulama

Model eğitimini başlatmak, Cross-Validation sonuçlarını hesaplamak ve öznitelik önem analizlerini üretmek için:

```bash
python main.py
```

Bu komut ile:

* Veri ön işleme
* Özellik mühendisliği
* XGBoost eğitimi
* LightGBM eğitimi
* Ensemble tahminleri
* Cross-Validation
* Performans metriklerinin hesaplanması
* Öznitelik önemlerinin oluşturulması

gerçekleştirilir.

---

## 5. Test Provası — İsteğe Bağlı

Modelin test sürecini yarışma öncesinde simüle etmek için eğitim verilerinden örnek test verisi oluşturulabilir:

```bash
python generate_dummy_test.py
```

Bu adım isteğe bağlıdır ve yarışma öncesinde tahmin pipeline'ının uçtan uca kontrol edilmesi amacıyla kullanılabilir.

---

## 6. Yarışma Günü Test ve JSON Çıktısı

Yarışma sırasında USB bellek üzerinden alınan **şifreli test verileri** aşağıdaki dizine yerleştirilmelidir:

```text
veri/test/
```

Ardından tahmin pipeline'ı aşağıdaki komutla çalıştırılır:

```bash
python predict_test.py
```

İşlem tamamlandığında resmi çıktı:

```text
sonuclar/TEAM_918091_FINAL.json
```

konumunda oluşturulur.

Üretilen JSON dosyası, yarışmanın talep ettiği **TEKNOFEST çıktı formatına uygun** şekilde hazırlanır.

---

## 📁 Temel Proje Akışı

```text
Proje
│
├── requirements.txt
├── check_missing_data.py
├── main.py
├── generate_dummy_test.py
├── predict_test.py
│
├── veri/
│   └── test/
│       └── [şifreli test verileri]
│
└── sonuclar/
    └── TEAM_918091_FINAL.json
```

### Önerilen çalışma sırası

```text
1. Sanal ortamı oluştur
        ↓
2. Bağımlılıkları yükle
        ↓
3. Eksik veri analizini çalıştır
        ↓
4. Modeli eğit ve Cross-Validation yap
        ↓
5. Test provası gerçekleştir
        ↓
6. Yarışma test verilerini veri/test/ içerisine koy
        ↓
7. predict_test.py çalıştır
        ↓
8. TEAM_918091_FINAL.json çıktısını kontrol et
```
