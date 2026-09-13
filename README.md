# GenVision (Takım NOVA, 918091) — Final Günü Çalıştırma Talimatı

Bu dosya, TEKNOFEST 2026 Sağlıkta Yapay Zeka Yarışması Final Uygulama ve Sonuç
Teslim Kılavuzu'nun Bölüm 2 madde 6'sında istenen README'dir: çalıştırma
komutu, giriş/çıkış dosya yolları ve sonuç JSON'unun üretildiği konum.

## Ortam

Tüm bağımlılıklar bu proje klasörünün içindeki `venv/` klasöründe ZATEN
KURULU — final günü **internet gerekmez, `pip install` çalıştırmaya gerek
yoktur**. `requirements.txt` / `requirements-inference.txt` yalnızca
referans/dokümantasyon amaçlıdır, kurulum için değildir.

`venv/`, `C:\Users\Casper\AppData\Local\Programs\Python\Python313` altındaki
STANDART (conda değil) bir Python kurulumundan oluşturuldu (13.09.2026'da
miniconda3'e bağımlı eski venv'in yerine geçirildi -- bkz. `venv/pyvenv.cfg`).
Bu bilgisayarda çalışır durumda kalması için o Python kurulumunun silinmemesi
yeterli.

**BAŞKA BİR BİLGİSAYAR kullanmanız gerekirse:** `venv` klasörünü kopyalamak
İŞE YARAMAZ (her venv, oluşturulduğu bilgisayardaki Python kurulumuna bağımlıdır
-- bu, conda'ya özel bir sorun değil, venv'in genel doğası). O bilgisayarda,
internet varken (final günü değil, ÖNCEDEN), proje klasörünü kopyalayıp şunu
çalıştırın:
```
cd Genvision_Yarisma
python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements-inference.txt
```
Bu tarif 13.09.2026'da test edildi: PyPI'dan gelen sürümler mevcut
`models/deploy/*.joblib` dosyalarıyla birebir uyumlu çıktı (aynı requirements
pinleri, `--dry-run` ile doğrulandı).

**Final günü öncesi bu makinede, Wi-Fi kapalıyken, aşağıdaki komutun hâlâ
çalıştığını mutlaka test edin.**

## Başka bir bilgisayarda SIFIRDAN kurulum (ÖNCEDEN, internet varken yapılmalı)

Final günü değil, ŞİMDİ (evde, internetin olduğu bir zamanda) yapın:

1. **Proje klasörünü o bilgisayara kopyalayın** — tüm `Genvision_Yarisma`
   klasörü (USB, harici disk veya bulut ile). `venv/` klasörünü de
   kopyalayabilirsiniz ama işe yaramayacak, aşağıdaki adımlarda yeniden
   kurulacak — isterseniz kopyalamadan da atlayabilirsiniz, yer kaplamaz.

2. **O bilgisayarda Python kurulu mu kontrol edin.** PowerShell açıp:
   ```
   python --version
   ```
   yazın.
   - Eğer bir sürüm numarası görürseniz (ör. `Python 3.13.5`) → 3. adıma geçin.
   - Eğer "python tanınmıyor/bulunamadı" derse → Python kurulu değil, şuradan
     indirip kurun: python.org/downloads (Windows installer, "Add python.exe
     to PATH" kutucuğunu MUTLAKA işaretleyin). Kurulumdan sonra PowerShell'i
     kapatıp yeniden açın, `python --version`'ı tekrar deneyin.

3. **Proje klasörüne girin** (kopyaladığınız yere göre yol değişir):
   ```
   cd C:\yol\Genvision_Yarisma
   ```

4. **Yeni bir venv oluşturun ve paketleri kurun** (internet gerekir, ~5-10 dk sürebilir):
   ```
   python -m venv venv
   venv\Scripts\python.exe -m pip install --upgrade pip
   venv\Scripts\python.exe -m pip install -r requirements-inference.txt
   ```

5. **Hâlâ internet varken, format doğrulamasını çalıştırıp test edin:**
   ```
   cd src
   ..\venv\Scripts\python.exe predict_final.py --dry-run
   ```
   Sonunda `Doğrulama OK` ve `Kaydedildi` görürseniz, o bilgisayar da final
   günü kullanılmaya hazır demektir.

6. Bu adımları o bilgisayarda BİR KEZ, önceden yapmanız yeterli — final günü
   sadece `README.md`'nin başındaki normal komutu çalıştıracaksınız (internet
   gerekmez, kurulum tekrarlanmaz).

## Final günü test verisi nereye konulacak

USB'den gelen (şifre çözülmüş) test dosyalarını doğrudan USB üzerinden
çalıştırmaya kalkmayın — kopyalayıp şu klasöre yapıştırın:

```
C:\Users\Casper\Desktop\Genvision_Yarisma\data\final_test\
```

Bu klasör önceden oluşturuldu, şu an boş. Dosya adları organizasyonun
verdiği gerçek adlar olacak (script dosya adına bakmaz, siz hangi dosyayı
hangi `--panel` bayrağına verirseniz o kullanılır).

## Final günü çalıştırma komutu

```
cd src
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --master ..\data\final_test\<gerçek dosya adı>.csv --kanser ..\data\final_test\<gerçek dosya adı>.csv --pah ..\data\final_test\<gerçek dosya adı>.csv --cftr ..\data\final_test\<gerçek dosya adı>.csv
```

Her `--panel <yol>` argümanı organizasyonun USB'den verdiği ilgili panelin
gerçek test CSV dosyasının tam yoludur — **dosyanın adı ÖNEMLİ DEĞİL**, hangi
dosyayı hangi panele verdiğiniz önemli. Bir panel için dosya yoksa o `--panel`
argümanı atlanabilir (script diğer panelleri yine üretir, uyarı basar).

**Kılavuz test verisinin panel başına ayrı dosya mı, yoksa TEK bir dosyada
"Panel" sütunuyla mı geleceğini belirtmiyor** — final günü verilen örnek
veriyle (Bölüm 2 madde 2) bu netleşecek. İkinci durumdaysanız (tek dosya,
içinde bir "Panel"/"panel" sütunu var) yukarıdaki komut yerine:

```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --input <tek CSV yolu>
```

`--input` ile `--master/--kanser/--pah/--cftr` AYNI ANDA kullanılamaz.

**Çıktı:** `TEAM_918091_FINAL.json` (proje kök dizininde, yukarıdaki `--out`
ile belirtilen yol) — bu dosya web sistemine yüklenecek TEK dosyadır.

## Format ön kontrolü (gerçek test verisi gelmeden önce, örnek veriyle)

```
cd src
..\venv\Scripts\python.exe predict_final.py --dry-run
```

Kendi eğitim verimizi kullanarak uçtan uca JSON üretimini ve doğrulamasını
test eder (`reports\DRY_RUN_submission.json`). "Doğrulama OK" ve "Kaydedildi"
satırlarını görmelisiniz. Bu, kılavuzun Bölüm 2 madde 2'sindeki resmi format
ön kontrolünün YERİNE GEÇMEZ, sadece kendi ek sağlık kontrolümüzdür.

## Model/deploy paketleri

`models/deploy/{MASTER,KANSER,PAH,CFTR}_deploy.joblib` — önceden eğitilmiş,
diskte hazır. Bu proje klasörünün kendi `venv/`si ile eğitilip paketlendi
(13.09.2026), aynı `venv/` ile yüklenmeli. `models/threshold_results.json`
her panelin karar eşiğini ve seçilen modelini gösterir.

`predict_final.py`, sabit eşiğin ürettiği pozitif oran PDR'nin varsaydığı
orandan çok saparsa (1.8x üstü / 0.55x altı) otomatik olarak oran-tabanlı bir
eşiğe döner (kalibrasyon kayması güvenlik ağı, bkz. `src/guard_scan.py` ve
`models/guard_scan_results.json`). Bu normal ve beklenen bir davranıştır,
konsola hangi modun kullanıldığını yazdırır.

## Dosya biçimi sağlamlığı

`predict_final.py`, test CSV'lerini şu sürprizlere karşı otomatik düzeltir
(hiçbir şey yapmanız gerekmez, konsola ne yaptığını yazar):
- Ayraç virgül değil noktalı virgül/tab ise
- Ondalık ayracı nokta değil virgül ise (TR Excel'den kaydedilmiş dosyalar)
- ID sütunu tam olarak "Variant_ID" değil de büyük/küçük harf veya
  boşluk/alt çizgi farklıysa (örn. "variant id", "VariantID")

ID sütunu HİÇ bulunamazsa script açıklayıcı bir hatayla durur (ID'siz
gönderim üretilemez, sıra numarası uydurmak yanlış eşleşmeye yol açar).

## VS Code'da çalıştırma

VS Code'a özel bir kurulum gerekmiyor -- yukarıdaki komutları VS Code'un
İÇİNDEKİ TERMİNALDEN çalıştırın, ayrı bir PowerShell penceresi açmanıza
gerek yok:

1. VS Code'da projeyi açın (`File > Open Folder` -> `Genvision_Yarisma`).
2. Üst menüden `Terminal > New Terminal` (veya `` Ctrl+` ``).
3. Açılan terminal panelinde bu README'deki komutları AYNEN yazın.

**DİKKAT -- VS Code'un sağ üstteki ▷ ("Run Python File") butonunu veya
"Run > Start Debugging"i final günü KULLANMAYIN:**
- O buton dosyayı hiçbir `--master/--kanser/--pah/--cftr` argümanı
  VERMEDEN çalıştırır -- script "en az bir panel için test dosyası verin"
  hatasıyla durur.
- VS Code'un o an seçili Python yorumlayıcısı `venv\Scripts\python.exe`
  OLMAYABİLİR (sol alttaki Python sürüm göstergesine bakın) -- yanlış
  yorumlayıcıyla paket eksikliği/sürüm uyumsuzluğu hatası alırsınız.

Terminalden komut satırıyla çalıştırmak bu iki sorunu da baştan ortadan
kaldırır -- bu yüzden README'deki komutlar hep terminal (`..\venv\Scripts\
python.exe ...`) biçiminde yazıldı.

## `uret.py` HAKKINDA UYARI

Proje kökündeki `uret.py`, adına rağmen final günü kullanılacak script
DEĞİLDİR — `data/test/YARISMA_TEST_*.csv` sabit dosya adlarını kullanır,
gerçek final dosyalarını bulamaz. Sadece yerel hızlı deneme içindir. Final
günü her zaman yukarıdaki `predict_final.py` komutunu kullanın.

## Sorun giderme

- `ModuleNotFoundError` / import hatası: yanlış Python ile çalıştırıyorsunuz
  demektir — mutlaka `venv\Scripts\python.exe` kullanın, sistem Python'ı değil.
- `models/deploy/*.joblib` yüklenirken `InconsistentVersionWarning` normaldir
  (uyarı, hata değil); sonuç bozulmaz. Tam çökme (exception) alırsanız,
  `venv/`in bozulmadığından ve doğru dizinde (`src/`) çalıştırdığınızdan emin
  olun.
- Bir panelin test dosyası formatı bozuksa (`GenVisionFeaturizer._check_schema`
  hata verir), hangi sütunların eksik olduğunu isim isim listeler.
