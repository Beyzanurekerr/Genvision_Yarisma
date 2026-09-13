# Final Günü — Yapılacaklar Listesi

Sırayla, atlamadan:

**☐ 1.** `data\final_test\` klasörünü aç, İÇİNDEKİ HER ŞEYİ SİL (eski deneme
dosyaları kalmasın — kalırsa gerçek dosyalarla karışıp hata verir).

**☐ 2.** USB'deki şifreyi organizasyon açıklayınca, test dosyalarını
kopyalayıp `data\final_test\` klasörüne yapıştır.

**☐ 3.** PowerShell aç (ya da VS Code'da `` Ctrl+` `` ile terminal aç —
sağ üstteki ▷ "Run" butonuna BASMA).

**☐ 4.** Şunu yapıştır, Enter:
```
cd "C:\Users\Casper\Desktop\Genvision_Yarisma\src"
```

**☐ 5.** Şunu yapıştır, Enter:
```
..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json
```

**☐ 6.** Ekranda çıkan `MASTER -> ...`, `KANSER -> ...`, `PAH -> ...`,
`CFTR -> ...` satırlarını oku — her panelin karşısında GERÇEKTEN o panele ait
bir dosya adı yazıyor mu kontrol et. Yanlış/eksik görünüyorsa bkz. **"Otomatik
eşleşme yanlış çıktıysa"** aşağıda.

**☐ 7.** En altta şu iki satırı gördün mü kontrol et:
```
Doğrulama OK: ... tahmin, ... benzersiz (panel,id).
Kaydedildi: ...TEAM_918091_FINAL.json ...
```
Görmediysen, ekrandaki hatayı olduğu gibi kopyala, buraya yapıştır.

**☐ 8.** `TEAM_918091_FINAL.json` dosyasını (proje kök klasöründe) organizasyonun
web sistemine yükle. **Bu, teslimin kendisi.**

---

## Bir şey ters giderse

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

---

## Önceden (finalden önce, evde) test etmek istersen

```
cd src
..\venv\Scripts\python.exe predict_final.py --dry-run
```
Bu, kendi eğitim verimizle sahte bir uçtan-uca deneme yapar — `Doğrulama OK`
ve `Kaydedildi` görürsen sistem çalışıyor demektir. **Bunu denedikten sonra
`data\final_test\` klasörünü tekrar boşaltmayı unutma (1. adım).**

## Daha fazla bilgi

- İkinci bilgisayarda sıfırdan kurulum, `venv` detayları, GitHub güncelleme: `IKINCI_BILGISAYAR.md`
- Modellerin nasıl seçildiği/kalibre edildiği: `models/threshold_results.json`, `src/guard_scan.py`
- Bağımsız dış veri doğrulaması: `external_validation/README.md`
