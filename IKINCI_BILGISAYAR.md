# İkinci Bilgisayar — Kurulum ve Güncelleme

Final günü lazım DEĞİL, bunları ÖNCEDEN (internetin olduğu bir zamanda) yapın.

## venv neden kopyalanamıyor

`venv/` klasörü, oluşturulduğu bilgisayardaki bir Python kurulumuna bağımlıdır
(bu venv'in genel bir sınırı, projeye özel değil). Başka bir bilgisayarda
`venv/` klasörünü kopyalamak İŞE YARAMAZ — o bilgisayarda sıfırdan
oluşturulması gerekir. Bu ana bilgisayardaki `venv/`, `C:\Users\Casper\
AppData\Local\Programs\Python\Python313` kurulumuna bağlı (13.09.2026'da
eski, miniconda3'e bağımlı venv'in yerine geçirildi).

## Sıfırdan kurulum (yeni/farklı bir bilgisayarda)

1. **Proje klasörünü o bilgisayara taşıyın** — GitHub'dan klonlayarak (bkz.
   aşağı) YA DA zip/USB ile kopyalayarak. `venv/` klasörünü taşımanıza gerek
   yok, boşuna yer kaplar; aşağıda sıfırdan oluşturulacak.

2. **Python kurulu mu kontrol edin.** PowerShell'de:
   ```
   python --version
   ```
   - Sürüm numarası çıkarsa → 3. adıma geçin.
   - "Tanınmıyor" derse → python.org/downloads'tan indirip kurun
     ("Add python.exe to PATH" kutucuğunu MUTLAKA işaretleyin), PowerShell'i
     kapatıp yeniden açın, tekrar deneyin.

3. **Proje klasörüne girin:**
   ```
   cd C:\yol\Genvision_Yarisma
   ```

4. **venv oluşturun ve paketleri kurun** (internet gerekir, ~5-10 dk):
   ```
   python -m venv venv
   venv\Scripts\python.exe -m pip install -r requirements-inference.txt
   ```

5. **Hâlâ internet varken test edin:**
   ```
   cd src
   ..\venv\Scripts\python.exe predict_final.py --dry-run
   ```
   `Doğrulama OK` ve `Kaydedildi` görürseniz, o bilgisayar da hazır demektir.

Bu tarif 13.09.2026'da test edildi: PyPI'dan gelen paket sürümleri mevcut
`models/deploy/*.joblib` dosyalarıyla birebir uyumlu çıktı, sonuçlar ana
bilgisayarla birebir eşleşti.

## GitHub'dan indirme: klon mu, zip mi?

Proje `github.com/Beyzanurekerr/Genvision_Yarisma` adresinde. İki yöntem var,
**farkı ileride güncelleme alıp almayacağınız:**

**A) `git clone` ile (ÖNERİLEN — güncelleme almak istiyorsanız):**
```
git clone https://github.com/Beyzanurekerr/Genvision_Yarisma.git
```
Bu, `.git` klasörünü de indirir. İleride kod güncellenirse, o bilgisayarda
proje klasörüne girip şunu çalıştırmanız yeterli:
```
cd Genvision_Yarisma
git pull
```
(Kendi yerel değişiklikleriniz varsa `git pull` çakışma verebilir — böyle bir
durumda mesajı bana yapıştırın.)

**B) "Code > Download ZIP" ile (Emre'nin denediği yöntem):**
Bu bir zip dosyası indirir, `.git` klasörü İÇERMEZ. Yani o kopyada `git pull`
ÇALIŞMAZ ("not a git repository" hatası verir) — güncelleme almak için ya
yeniden zip indirmeniz, ya da o klasörde `git init` + `git remote add origin
...` + `git pull` ile sonradan git'e bağlamanız gerekir. Tek seferlik bir
kopya için sorun değil, ama güncel kalmak istiyorsanız A yöntemini kullanın.

Zip'i açarken dikkat: Windows bazen zip'in adıyla aynı isimde bir üst klasör
daha oluşturup içine açar (iç içe `Genvision_Yarisma\Genvision_Yarisma\...`
gibi). Bu koda zarar vermez, sadece `cd` ile doğru derinliğe (src/data/models
klasörlerini gördüğünüz yere) inmeniz gerekir.

## Klasör adını değiştirirsem ne olur

Hiçbir Python dosyasında proje klasörünün adı sabit (hardcoded) yazılı değil
— `Genvision_Yarisma_v2` gibi farklı bir isimle de sorunsuz çalışır. Sadece
`cd` komutunda gerçek klasör adını yazmanız yeterli.
