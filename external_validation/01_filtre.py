"""
GenVision - Dış (ClinVar) Doğrulama Seti - Adım 1: Filtreleme + Panel Ayrımı
================================================================================
Resmi şartnameye (Üniversite ve Üzeri, Bölüm 3.2) birebir uyan metodoloji:
  Patojenik: ClinVar Expert Panel / Practice Guideline (3-4 yıldız), missense
  Benign: ClinVar Benign/Likely Benign (gnomAD eklemesi bu adımda YOK -- kaynak
          kısıtı, ClinVar'ın kendi Benign/Likely Benign havuzu kullanılıyor)

Panel-gen eşlemesi (resmi şartname + PDR'den):
  CFTR   -> CFTR geni
  PAH    -> PAH geni (Fenilketonüri)
  KANSER -> kalıtsal kanser genleri (BRCA1/2, TP53, Lynch genleri vb.)
  MASTER -> "genel varyant havuzu" -- gen kısıtı yok, diğer 3 panelin genleri
            hariç tutularak genel bir örneklem
"""
import gzip
import pandas as pd

RAW = "/tmp/clinvar_external/variant_summary.txt.gz"
OUT_DIR = "/tmp/claude-1000/-home-beyza/3044984c-e46a-4edd-84f8-c6f9355ddcb2/scratchpad/clinvar_external"

KANSER_GENES = {
    "BRCA1", "BRCA2", "TP53", "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM",
    "APC", "PTEN", "STK11", "CDH1", "ATM", "CHEK2", "PALB2",
    "RAD51C", "RAD51D", "BRIP1", "MUTYH",
}
PAH_GENES = {"PAH"}
CFTR_GENES = {"CFTR"}

GUVENILIR_STATU = {
    "practice guideline",
    "reviewed by expert panel",
}

USECOLS = ["Type", "Name", "GeneSymbol", "ClinicalSignificance", "Assembly",
           "Chromosome", "Start", "ReferenceAlleleVCF", "AlternateAlleleVCF",
           "ReviewStatus", "VariationID"]

print("ClinVar dosyasi parca parca okunuyor (bellek icin chunk'lanmis)...")
parcalar = []
toplam_satir = 0
for chunk in pd.read_csv(RAW, sep="\t", usecols=USECOLS, dtype=str,
                          compression="gzip", low_memory=False, chunksize=200_000):
    toplam_satir += len(chunk)
    chunk = chunk[(chunk["Assembly"] == "GRCh38") &
                  (chunk["Type"] == "single nucleotide variant")]
    if len(chunk):
        parcalar.append(chunk)
df = pd.concat(parcalar, ignore_index=True)
del parcalar
print(f"Toplam okunan satir: {toplam_satir:,}")
print(f"GRCh38+SNV sonrasi: {len(df):,}")

# Missense yaklaşık tespiti: protein değişimi var (p.XxxNNNYyy), ama
# nonsense/frameshift/Ter/del/ins/dup DEĞİL -- HGVS parse etmeden basit
# metin filtresi, %100 kesin değil ama makul bir yaklaşım.
def missense_gibi(name):
    if not isinstance(name, str) or "p." not in name:
        return False
    protein = name.split("p.")[-1]
    if any(k in protein for k in ["Ter", "fs", "del", "ins", "dup", "*", "="]):
        return False
    return True

df["missense_yaklasik"] = df["Name"].apply(missense_gibi)
df_ms = df[df["missense_yaklasik"]]
print(f"Missense-benzeri sonrasi: {len(df_ms):,}")

review_lower = df_ms["ReviewStatus"].str.lower()
pat_mask = (df_ms["ClinicalSignificance"].isin(["Pathogenic", "Likely pathogenic",
                                                  "Pathogenic/Likely pathogenic"])
            & review_lower.isin(GUVENILIR_STATU))
ben_mask = df_ms["ClinicalSignificance"].isin(
    ["Benign", "Likely benign", "Benign/Likely benign"])

pat = df_ms[pat_mask].copy()
ben = df_ms[ben_mask].copy()
print(f"Patojenik (3-4 yildiz, missense-benzeri): {len(pat):,}")
print(f"Benign (tum yildizlar, missense-benzeri): {len(ben):,}")

pat["Label"] = 1
ben["Label"] = 0
tum = pd.concat([pat, ben], ignore_index=True)
tum = tum.drop_duplicates(subset=["VariationID"])

panel_genler = {"CFTR": CFTR_GENES, "PAH": PAH_GENES, "KANSER": KANSER_GENES}
tum_ozel_genler = CFTR_GENES | PAH_GENES | KANSER_GENES

for panel, genler in panel_genler.items():
    alt = tum[tum["GeneSymbol"].isin(genler)]
    yol = f"{OUT_DIR}/clinvar_{panel}.csv"
    alt.to_csv(yol, index=False)
    print(f"{panel:8s}: {len(alt):5d} varyant (patojenik={int((alt.Label==1).sum())}, "
          f"benign={int((alt.Label==0).sum())}) -> {yol}")

# MASTER: ozel panel genleri HARIC, genel havuz
master = tum[~tum["GeneSymbol"].isin(tum_ozel_genler)]
yol = f"{OUT_DIR}/clinvar_MASTER.csv"
master.to_csv(yol, index=False)
print(f"MASTER  : {len(master):5d} varyant (patojenik={int((master.Label==1).sum())}, "
      f"benign={int((master.Label==0).sum())}) -> {yol}")
