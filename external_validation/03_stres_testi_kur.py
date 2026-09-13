"""
Adım 3: ClinVar + gnomAD havuzlarını birleştirip, resmi şartnamedeki TEST
kompozisyonuna (klinik stres testi: benign-baskın) mümkün olan en yakın
oranda örnekleyip panel başına final dış test setini kurar.

Resmi hedef (test seti, şartname Bölüm 3.2):
  MASTER  500 pat / 3000 ben   KANSER 100 pat / 500 ben
  PAH     100 pat / 250 ben    CFTR    20 pat / 100 ben

Gerçek veri kısıtı: PAH ve CFTR'de ClinVar+gnomAD'da yeterli 'benign'
bulunamadı (özellikle PAH -- tam olarak şartnamenin gnomAD eklemesini
gerekçelendirdiği durum). Bu panellerde mevcut havuz kadar, ama ORANI
koruyarak örnekleniyor.
"""
import numpy as np
import pandas as pd

D = "/tmp/claude-1000/-home-beyza/3044984c-e46a-4edd-84f8-c6f9355ddcb2/scratchpad/clinvar_external"
SEED = 42
rng = np.random.default_rng(SEED)

HEDEF = {
    "MASTER": {"pat": 500, "ben": 3000},
    "KANSER": {"pat": 100, "ben": 500},
    "PAH":    {"pat": 100, "ben": 250},
    "CFTR":   {"pat": 20,  "ben": 100},
}


def havuz_yukle(panel):
    clinvar = pd.read_csv(f"{D}/clinvar_{panel}.csv")
    pat = clinvar[clinvar.Label == 1][["GeneSymbol", "Name", "Label"]].copy()
    ben = clinvar[clinvar.Label == 0][["GeneSymbol", "Name", "Label"]].copy()
    try:
        gnomad = pd.read_csv(f"{D}/gnomad_benign_{panel}.csv")
        gnomad_ben = gnomad[["GeneSymbol", "Name", "Label"]].copy()
        ben = pd.concat([ben, gnomad_ben], ignore_index=True)
    except FileNotFoundError:
        pass
    return pat.drop_duplicates(), ben.drop_duplicates()


ozet = []
for panel, hedef in HEDEF.items():
    pat, ben = havuz_yukle(panel)
    n_pat_havuz, n_ben_havuz = len(pat), len(ben)

    if n_pat_havuz >= hedef["pat"] and n_ben_havuz >= hedef["ben"]:
        n_pat, n_ben = hedef["pat"], hedef["ben"]
        not_ekle = "resmi hedefe tam ulasildi"
    else:
        # Havuz yetersiz -- oranı koru, mevcut havuza göre ölçekle.
        oran = hedef["pat"] / hedef["ben"]
        if n_ben_havuz * oran <= n_pat_havuz:
            n_ben = n_ben_havuz
            n_pat = max(1, round(n_ben * oran))
        else:
            n_pat = n_pat_havuz
            n_ben = max(1, round(n_pat / oran))
        not_ekle = f"HAVUZ YETERSIZ -- oran korunarak kucultuldu (hedef {hedef['pat']}/{hedef['ben']})"

    secilen_pat = pat.sample(n=min(n_pat, n_pat_havuz), random_state=SEED)
    secilen_ben = ben.sample(n=min(n_ben, n_ben_havuz), random_state=SEED)
    final = pd.concat([secilen_pat, secilen_ben], ignore_index=True)
    final = final.sample(frac=1, random_state=SEED).reset_index(drop=True)  # karistir
    final["Variant_ID"] = [f"EXT_{panel}_{i:05d}" for i in range(len(final))]

    yol = f"{D}/EXTERNAL_TEST_{panel}.csv"
    final.to_csv(yol, index=False)
    ozet.append((panel, n_pat_havuz, n_ben_havuz, len(secilen_pat), len(secilen_ben), not_ekle))
    print(f"{panel:8s} havuz(pat={n_pat_havuz:5d}, ben={n_ben_havuz:6d}) -> "
          f"secilen(pat={len(secilen_pat):4d}, ben={len(secilen_ben):4d})  [{not_ekle}]  -> {yol}")

pd.DataFrame(ozet, columns=["panel", "havuz_pat", "havuz_ben", "secilen_pat", "secilen_ben", "not"]).to_csv(
    f"{D}/OZET_stres_testi.csv", index=False)
