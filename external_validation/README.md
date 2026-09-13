# Dış (ClinVar) Doğrulama Veri Seti

Yarışma verisine hiç dokunmadan, tamamen bağımsız kaynaklardan (ClinVar + gnomAD)
kurulmuş, etiketli, panel-uygun bir dış test seti. Resmi şartnamenin (13.09.2026'da
bulunan `2026-_Sağlıkta_Yapay_Zeka_Türkçe_Şartname_v2...pdf`, Bölüm 3.2) kendi
tarif ettiği veri kurma yöntemiyle BİREBİR aynı mantık kullanıldı.

## Yöntem

1. **Patojenik sınıf:** NCBI ClinVar'dan (GRCh38, tek nükleotid varyant,
   missense-benzeri), **3-4 yıldız güvenilirlik** (Expert Panel / Practice
   Guideline) düzeyinde Pathogenic/Likely pathogenic varyantlar.
2. **Benign sınıf:** ClinVar Benign/Likely benign + (PAH ve CFTR panellerinde,
   ClinVar'da yeterli örnek olmadığı için) gnomAD'dan o genlerde sık görülen
   (AF≥0.0001) missense varyantlar -- şartnamenin kendi "gnomAD eklemesi"
   adımıyla aynı gerekçe.
3. **Panel-gen eşlemesi** (resmi şartname + PDR'den doğrulandı):
   - MASTER = genel havuz (CFTR/PAH/KANSER genleri hariç, geniş örneklem)
   - KANSER = kalıtsal kanser genleri (BRCA1/2, TP53, Lynch genleri vb.)
   - PAH = **Fenilketonüri** geni (`PAH`) -- pulmoner hipertansiyon DEĞİL
     (bu karışıklık `generate_report.py`'de bulunup düzeltildi)
   - CFTR = `CFTR` geni
4. **Klinik stres testi oranı:** Resmi test kompozisyonuna (Bölüm 3.2)
   mümkün olduğunca uyularak örneklendi.

## Sonuç

| Panel | Havuzda bulunan | Seçilen | Not |
|---|---|---|---|
| MASTER | 2426 pat / 142655 ben | 500 / 3000 | Resmi hedefe tam ulaşıldı |
| KANSER | 489 pat / 2604 ben | 100 / 500 | Resmi hedefe tam ulaşıldı |
| CFTR | 92 pat / 107 ben | 20 / 100 | Resmi hedefe tam ulaşıldı |
| PAH | 355 pat / 36 ben | 14 / 36 | **Gerçek veri kısıtı** -- oran korunarak küçültüldü (ayrıntı aşağıda) |

**PAH bulgusu:** ClinVar+gnomAD'da PAH geni için güvenilir "benign" missense
varyant sayısı çok az (toplam 36) -- resmi şartnamenin 250 benign hedefine
ulaşılamadı. Bu, organizasyonun kendi verisinde neden gnomAD eklemesi yapmak
zorunda kaldığını (Bölüm 3.2) doğrudan doğruluyor; rastgele bir eksiklik değil,
gerçek dünyada da gözlenen bir veri kıtlığı.

## Neden modelden GEÇİRİLMEDİ (Seviye 2 yapılmadı)

Gerçek eğitim verisinin şeması incelendiğinde, model her varyant için **334
ayrı `AL_` sütunu, 9 ayrı `EK_` sütunu** bekliyor (sadece 2 `AA_` sütunu var).
Bizim üretebildiğimiz gerçek sinyal sayısı (1 gnomAD frekansı, 1 UCSC phyloP
skoru, 1 Grantham mesafesi) bu boyutun çok altında kaldığı için, bu veriyi
doğrudan modele vermek 333/334 `AL_` ve 8/9 `EK_` sütununu "gözlenmemiş"
(sentinel) yapardı -- ki `features.py` eksikliği başlı başına bir sinyal
olarak kullanıyor. Sonuç, modelin gerçek performansını değil, "%99 eksik
veriyle ne yaptığını" ölçerdi -- yanıltıcı olurdu. Bu yüzden bilinçli olarak
durduruldu; veri seti sadece Seviye 1 (etiketli, bağımsız kanıt) olarak
bırakıldı.

## Dosyalar

- `EXTERNAL_TEST_{PANEL}.csv`: panel başına final veri seti (GeneSymbol, Name
  [ClinVar HGVS], Label [1=Patojenik/0=Benign], Variant_ID).
- `01_filtre.py` / `02_gnomad_benign.py` / `03_stres_testi_kur.py`: üretim
  scriptleri (ham ClinVar `variant_summary.txt.gz`, ~440MB, tekrar üretmek
  isteyen NCBI FTP'den indirmeli -- repoya dahil edilmedi).
