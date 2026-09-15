# Final Günü — Yapılacaklar Listesi

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

## Önceden (finalden önce, evde) PROVA etmek istersen

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

**4. ÖNEMLİ:** Provadan sonra `data\final_test\` klasörünü boşalt (1. adım) —
yoksa final günü bu sahte dosyalar gerçeklerle karışır.

*(Sadece format/JSON sağlık kontrolü istiyorsan, gerçek dosya kopyalamadan
`..\venv\Scripts\python.exe predict_final.py --dry-run` da çalışır — ama bu
gerçek final komutunu DENEMEZ, sadece kendi eğitim verimizi kullanır.)*

## Daha fazla bilgi

- Modellerin nasıl seçildiği/kalibre edildiği: `models/threshold_results.json`, `src/guard_scan.py`
- Bağımsız dış veri doğrulaması: `external_validation/README.md`
