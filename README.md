# GenVision (Takım NOVA, 918091) — Final Günü Rehberi

## İçindekiler
1. [Final günü — yapılacaklar listesi](#1-final-günü--yapılacaklar-listesi)
2. [Bir şey ters giderse](#2-bir-şey-ters-giderse)
3. [Önceden prova etmek istersen](#3-önceden-prova-etmek-istersen)
4. [Başka bir bilgisayarda sıfırdan kurulum](#4-başka-bir-bilgisayarda-sıfırdan-kurulum)
5. [GitHub'dan indirme ve güncelleme](#5-githubdan-indirme-ve-güncelleme)
6. [Diğer bilgiler](#6-diğer-bilgiler)

---

## 1. Final günü — yapılacaklar listesi

Sırayla, atlamadan:

**☐ 1.** `data\final_test\` klasörünü aç, İÇİNDEKİ HER ŞEYİ SİL (eski deneme
dosyaları kalmasın — kalırsa gerçek dosyalarla karışıp hata verir).

**☐ 2.** USB'deki şifreyi organizasyon açıklayınca, test dosyalarını
kopyalayıp `data\final_test\` klasörüne yapıştır.

**☐ 3.** Dosya Gezgini'nde `Genvision_Yarisma` klasörünü aç (nereye
kopyaladıysan/kurduysan orada — bilgisayara göre değişir, sabit bir yol
YOKTUR). Adres çubuğuna tıkla, `powershell` yaz, Enter'a bas — PowerShell
doğrudan o klasörde açılır (hangi bilgisayar/kullanıcı adı olursa olsun
çalışır). VS Code kullanıyorsan `` Ctrl+` `` ile terminal aç — sağ üstteki
▷ "Run" butonuna BASMA.

**☐ 4.** Şunu yapıştır, Enter:
```
cd src
```

**☐ 5.** Şunu yapıştır, Enter:
```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json
```

**☐ 6.** Ekranda çıkan `MASTER -> ...`, `KANSER -> ...`, `PAH -> ...`,
`CFTR -> ...` satırlarını oku — her panelin karşısında GERÇEKTEN o panele ait
bir dosya adı yazıyor mu kontrol et. Yanlış/eksik görünüyorsa bkz. [Bölüm 2](#2-bir-şey-ters-giderse).

**☐ 7.** En altta şu iki satırı gördün mü kontrol et:
```
Doğrulama OK: ... tahmin, ... benzersiz (panel,id).
Kaydedildi: ...TEAM_918091_FINAL.json ...
```
Görmediysen, ekrandaki hatayı olduğu gibi kopyala, Claude'a yapıştır.

**☐ 8.** `TEAM_918091_FINAL.json` dosyasını (proje kök klasöründe) organizasyonun
web sistemine yükle. **Bu, teslimin kendisi.**

---

## 2. Bir şey ters giderse

**Otomatik eşleşme yanlış/eksik çıktıysa** → dosyaları elle belirt (dosya
adının gerçekte ne olduğu önemli değil, hangisini hangi panele verdiğin önemli):
```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --master <yol> --kanser <yol> --pah <yol> --cftr <yol>
```

**4 ayrı dosya değil, TEK dosyada "Panel" sütunu varsa:**
```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --input <tek dosya yolu>
```

**`uret.py`'yi ÇALIŞTIRMA** — final günü işe yaramaz, sahte dosyalarla sınırlı.

**`ModuleNotFoundError`** → yanlış Python kullanıyorsun, mutlaka `venv\Scripts\python.exe` (sistem Python'ı değil).

**`InconsistentVersionWarning`** → zararsız, göz ardı et.

**Sütun eksik hatası** → hangi sütunların eksik olduğu mesajda yazar; doğru panel/dosya olduğunu kontrol et.

**Ayraç/ondalık virgül/farklı ID sütun adı** → script otomatik algılayıp düzeltir, konsola ne yaptığını yazar; ekstra bir şey yapmana gerek yok.

---

## 3. Önceden prova etmek istersen

Bu, `--dry-run` DEĞİL — final günü kullanacağın GERÇEK komutun aynısını,
sahte dosyalarla dener. Yukarıdaki 1-8 adımların bire bir provası:

**1.** `data\test\` klasöründeki sahte dosyaları `data\final_test\`'e kopyala:
```
copy data\test\YARISMA_TEST_*.csv data\final_test\
```

**2.** Final günü kullanacağın GERÇEK komutu (hiç değiştirmeden) çalıştır:
```
cd src
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json
```

**3.** Ekranda `MASTER -> YARISMA_TEST_MASTER.csv` gibi 4 satır ve en altta
`Doğrulama OK` + `Kaydedildi` görmelisin — final günü tam olarak bunu göreceksin,
sadece dosya adları farklı olacak.

**4. ÖNEMLİ:** Provadan sonra `data\final_test\` klasörünü boşalt (Bölüm 1,
adım 1) — yoksa final günü bu sahte dosyalar gerçeklerle karışır.

*(Sadece format/JSON sağlık kontrolü istiyorsan, gerçek dosya kopyalamadan
`..\venv\Scripts\python.exe predict_final.py --dry-run` da çalışır — ama bu
gerçek final komutunu DENEMEZ, sadece kendi eğitim verimizi kullanır.)*

---

## 4. Başka bir bilgisayarda sıfırdan kurulum

**Ne zaman:** final günü DEĞİL, bunu ÖNCEDEN (internetin olduğu bir zamanda) yapın.

**Neden gerekli:** `venv/` klasörü, oluşturulduğu bilgisayardaki bir Python
kurulumuna bağımlıdır (venv'in genel bir sınırı, projeye özel değil).
`venv/` klasörünü başka bir bilgisayara kopyalamak İŞE YARAMAZ — o
bilgisayarda sıfırdan oluşturulması gerekir.

**Adımlar:**

1. **Proje klasörünü o bilgisayara taşıyın** — GitHub'dan klonlayarak (bkz.
   [Bölüm 5](#5-githubdan-indirme-ve-güncelleme)) ya da zip/USB ile
   kopyalayarak. `venv/` klasörünü taşımanıza gerek yok, boşuna yer kaplar;
   burada sıfırdan oluşturulacak.

2. **O bilgisayarda Python kurulu mu kontrol edin.** PowerShell'de:
   ```
   python --version
   ```
   - Sürüm numarası çıkarsa → 3. adıma geçin.
   - "Tanınmıyor" derse → python.org/downloads'tan indirip kurun
     ("Add python.exe to PATH" kutucuğunu MUTLAKA işaretleyin), PowerShell'i
     kapatıp yeniden açın, tekrar deneyin.

3. **Proje klasörüne girin** (Dosya Gezgini'nde klasöre gidip adres çubuğuna
   `powershell` yazarsanız otomatik oraya açılır, yol ezberlemenize gerek yok).

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

6. Bu adımları o bilgisayarda BİR KEZ, önceden yapmanız yeterli — final günü
   sadece [Bölüm 1](#1-final-günü--yapılacaklar-listesi)'deki komutu
   çalıştıracaksınız (internet gerekmez, kurulum tekrarlanmaz).

Bu tarif 13.09.2026'da test edildi: PyPI'dan gelen paket sürümleri mevcut
`models/deploy/*.joblib` dosyalarıyla birebir uyumlu çıktı, sonuçlar ana
bilgisayarla birebir eşleşti.

---

## 5. GitHub'dan indirme ve güncelleme

Proje `github.com/Beyzanurekerr/Genvision_Yarisma` adresinde. İki yöntem var,
**farkı ileride güncelleme alıp almayacağınız:**

**A) `git clone` ile (ÖNERİLEN — güncelleme almak istiyorsanız):**
```
git clone https://github.com/Beyzanurekerr/Genvision_Yarisma.git
```
Bu, `.git` klasörünü de indirir. İleride kod güncellenirse, proje klasörüne
girip şunu çalıştırmanız yeterli:
```
git pull
```

**B) "Code > Download ZIP" ile:**
Bir zip dosyası indirir, `.git` klasörü İÇERMEZ — yani `git pull` ÇALIŞMAZ
("not a git repository" hatası verir). Güncelleme almak isterseniz, o
klasörün içinde şunu çalıştırın (mevcut yerel değişiklikleri GitHub'daki
hâliyle değiştirir, dikkatli kullanın):
```
git init
git remote add origin https://github.com/Beyzanurekerr/Genvision_Yarisma.git
git fetch origin
git reset --hard origin/main
```
Bundan sonra artık gerçek bir git deposu olduğu için sadece `git pull` yeterli.

Zip'i açarken dikkat: Windows bazen zip'in adıyla aynı isimde bir üst klasör
daha oluşturup içine açar (iç içe `Genvision_Yarisma\Genvision_Yarisma\...`
gibi). Bu koda zarar vermez, sadece `cd` ile doğru derinliğe (src/data/models
klasörlerini gördüğünüz yere) inmeniz gerekir.

**Klasör adını değiştirirsem ne olur:** Hiçbir Python dosyasında proje
klasörünün adı sabit (hardcoded) yazılı değil — farklı bir isimle de
sorunsuz çalışır.

---

## 6. Diğer bilgiler

- **Modellerin nasıl seçildiği/kalibre edildiği:** `models/threshold_results.json`, `src/guard_scan.py`, `models/guard_scan_results.json`
- **Bağımsız dış veri doğrulaması** (ClinVar+gnomAD, modele sokulmadı — sebebi belgede): `external_validation/README.md`
- **Kalibrasyon kayması güvenlik ağı:** `predict_final.py`, sabit eşiğin ürettiği pozitif oran beklenenden çok saparsa (1.8x üstü / 0.55x altı) otomatik olarak oran-tabanlı bir eşiğe döner; bu normal ve beklenen bir davranıştır, konsola hangi modun kullanıldığını yazdırır.
