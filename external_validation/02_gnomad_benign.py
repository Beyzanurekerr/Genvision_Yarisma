"""
Adım 2: gnomAD'dan "sık görülen = muhtemelen sağlıklı" missense varyantlar.
Şartnamenin kendi metodolojisi: "ClinVar (1381 varyant) + gnomAD'dan ClinVar
datasındaki genlerin sık görülen sağlıklı popülasyon varyantları (2153 varyant)".
PAH panelinde ClinVar'da yeterli 'Benign' (n=1) yok -- bu adım o boşluğu kapatır.
"""
import json
import subprocess
import pandas as pd

OUT_DIR = "/tmp/claude-1000/-home-beyza/3044984c-e46a-4edd-84f8-c6f9355ddcb2/scratchpad/clinvar_external"

# AF esigi: ACMG BS1 kriteri ruhunda -- populasyonda hastalik prevalansindan
# COK daha sik gorulen bir missense varyant, patojenik olmasi beklenmez.
AF_ESIK = 0.0001


def gnomad_missense_cek(gen_sembol):
    query = {
        "query": (
            '{ gene(gene_symbol: "%s", reference_genome: GRCh38) { '
            'variants(dataset: gnomad_r4) { pos ref alt consequence hgvsp '
            'genome { af } exome { af } } } }' % gen_sembol
        )
    }
    sonuc = subprocess.run(
        ["curl", "-s", "--max-time", "60", "-X", "POST", "https://gnomad.broadinstitute.org/api",
         "-H", "Content-Type: application/json", "-d", json.dumps(query)],
        capture_output=True, text=True
    )
    data = json.loads(sonuc.stdout)
    varyantlar = data["data"]["gene"]["variants"]
    satirlar = []
    for v in varyantlar:
        if v["consequence"] != "missense_variant" or not v["hgvsp"]:
            continue
        af_genome = (v.get("genome") or {}).get("af") or 0
        af_exome = (v.get("exome") or {}).get("af") or 0
        af = max(af_genome, af_exome)
        if af >= AF_ESIK:
            satirlar.append({
                "GeneSymbol": gen_sembol, "Name": v["hgvsp"], "pos": v["pos"],
                "ref": v["ref"], "alt": v["alt"], "af": af, "Label": 0,
                "kaynak": "gnomAD_sik_gorulen",
            })
    return pd.DataFrame(satirlar)


for gen in ["PAH", "CFTR"]:
    df = gnomad_missense_cek(gen)
    yol = f"{OUT_DIR}/gnomad_benign_{gen}.csv"
    df.to_csv(yol, index=False)
    print(f"{gen:6s}: {len(df)} sik-gorulen missense varyant (AF>={AF_ESIK}) -> {yol}")
