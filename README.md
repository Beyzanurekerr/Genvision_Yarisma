# GenVision (Takım NOVA, 918091) — Final Günü Talimatı

## Final günü, tek komut

1. USB'den gelen test dosyalarını şu klasöre kopyala:
   ```
   data\final_test\
   ```
2. PowerShell'de (ya da VS Code'un TERMİNAL panelinden -- ▷ "Run" butonuyla değil):
   ```
   cd src
   ..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json
   ```
3. Sonunda `Doğrulama OK` ve `Kaydedildi` gördüysen, üretilen
   `TEAM_918091_FINAL.json` dosyasını organizasyonun web sistemine yükle. Bu teslim.

Dosya adlarında MASTER/KANSER/PAH/CFTR geçtiği sürece (büyük/küçük harf
önemsiz) bu kadarı yeterli — script hangi dosyayı hangi panele eşlediğini
konsola yazar, göndermeden önce o satırları oku ve kontrol et. İnternet
gerekmez.

## Format ön kontrolü (gerçek veri gelmeden önce denemek için)

```
cd src
..\venv\Scripts\python.exe predict_final.py --dry-run
```

## Bir şey ters giderse

- **Otomatik eşleşme yanlış/eksik çıktıysa** → dosyaları elle belirt:
  ```
  ..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --master <yol> --kanser <yol> --pah <yol> --cftr <yol>
  ```
  (Dosya adı önemli değil, hangi dosyayı hangi bayrağa verdiğin önemli.)
- **Tek dosyada, içinde "Panel" sütunu varsa** (4 ayrı dosya değilse) →
  ```
  ..\venv\Scripts\python.exe predict_final.py --out ..\TEAM_918091_FINAL.json --input <tek dosya yolu>
  ```
- **`uret.py`'yi ÇALIŞTIRMA** — adına rağmen final günü işe yaramaz (sahte dosyalarla sınırlı).
- **`ModuleNotFoundError`** → yanlış Python kullanıyorsun, mutlaka `venv\Scripts\python.exe`.
- **`InconsistentVersionWarning`** → zararsız bir uyarı, göz ardı et.
- **Sütun eksik hatası** → hangi sütunların eksik olduğu mesajda yazar; doğru panel/dosya olduğunu kontrol et.

## Daha fazla bilgi

- İkinci bilgisayarda sıfırdan kurulum, `venv` detayları, GitHub'dan güncelleme: `IKINCI_BILGISAYAR.md`
- Modellerin nasıl seçildiği/kalibre edildiği: `models/threshold_results.json`, `src/guard_scan.py`, `models/guard_scan_results.json`
