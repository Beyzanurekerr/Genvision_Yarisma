# -*- coding: utf-8 -*-
"""
GenVision - Dış Veri Denemesi: ClinVar Grantham->Patojenite İlişkisi
========================================================================
Organizatör açıkça onaylı ("dış kaynaklı veri seti kullanabilirsiniz... buna
karışmıyoruz"), ama kolon isimleri anonim ve genomik adres yok -- bu yüzden
SATIR BAZLI eşleştirme (bizim varyantımızı ClinVar'daki karşılığıyla
birebir bulma) yapılamıyor. Bunun yerine DOLAYLI bir yaklaşım: ClinVar'dan
GERÇEK, etiketli, gen-bağımsız bir missense varyant örneklemi çekilip
(protein_change alanından) Grantham mesafesi hesaplanıyor, ve
"Grantham mesafesi ne kadar patojeniteyle ilişkili" sorusu GERÇEK dünya
verisinden ampirik olarak öğreniliyor. Bu ilişki (basit bir lojistik
regresyon eğrisi), bizim kendi AA_grantham özelliğimize uygulanarak yeni
bir "dış-veri-bilgili" özellik (EXT_grantham_prior) üretiliyor -- satır
eşleştirme gerekmediği için anonimlik kısıtını ihlal etmiyor.

ÖNEMLİ dürüstlük notu: Bu özellik yalnızca CV ile katkısı doğrulanırsa
features.py'ye kalıcı olarak eklenir (SMOTE ablasyonundaki disiplinle aynı).

Çalıştırma:
    cd src && python fetch_clinvar_grantham.py
Çıktı:
    models/clinvar_grantham_model.joblib (fit edilmiş basit lojistik model)
    models/clinvar_grantham_sample.json  (ham örneklem özeti, şeffaflık için)
"""
import json
import re
import time
from pathlib import Path

import joblib
import numpy as np
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from features import grantham_distance, GRANTHAM_PROPS

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
N_PER_CLASS = 2500  # esearch retmax hedefi (sınıf başına)
BATCH = 200          # esummary batch boyutu
SLEEP = 0.4          # NCBI rate-limit nezaketi (anahtarsız ~3 istek/sn sınırı)

AA3to1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V",
}
PROT_CHANGE_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")  # esummary "protein_change" alanı, örn "D2555G"


def esearch(term, retmax):
    r = requests.get(f"{EUTILS}/esearch.fcgi", params={
        "db": "clinvar", "term": term, "retmax": retmax, "retmode": "json",
    }, timeout=30)
    r.raise_for_status()
    return r.json()["esearchresult"]["idlist"]


def esummary_batch(ids):
    r = requests.get(f"{EUTILS}/esummary.fcgi", params={
        "db": "clinvar", "id": ",".join(ids), "retmode": "json",
    }, timeout=30)
    r.raise_for_status()
    data = r.json()["result"]
    return {uid: data[uid] for uid in data.get("uids", [])}


def fetch_class(term, label, n_target):
    ids = esearch(term, n_target)
    print(f"  esearch: {len(ids)} id bulundu (etiket={label})")
    rows = []
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        try:
            recs = esummary_batch(chunk)
        except Exception as e:
            print(f"    UYARI: batch {i} başarısız ({e}), atlanıyor")
            time.sleep(SLEEP)
            continue
        for uid, rec in recs.items():
            pc = rec.get("protein_change", "")
            if not pc:
                continue
            first = pc.split(",")[0].strip()
            m = PROT_CHANGE_RE.match(first)
            if not m:
                continue
            aa1, pos, aa2 = m.group(1), m.group(2), m.group(3)
            if aa1 not in GRANTHAM_PROPS or aa2 not in GRANTHAM_PROPS or aa1 == aa2:
                continue
            gd = grantham_distance(aa1, aa2)
            rows.append({"uid": uid, "aa1": aa1, "aa2": aa2, "grantham": gd, "label": label})
        time.sleep(SLEEP)
        if (i // BATCH) % 5 == 0:
            print(f"    {min(i+BATCH, len(ids))}/{len(ids)} işlendi, şu ana kadar {len(rows)} geçerli satır")
    return rows


def main():
    print("=== ClinVar'dan Patojenik missense örneklemi ===")
    path_rows = fetch_class(
        'missense_variant[molecular consequence] AND (pathogenic[clinsig] OR likely_pathogenic[clinsig])',
        1, N_PER_CLASS,
    )
    print("\n=== ClinVar'dan Benign missense örneklemi ===")
    benign_rows = fetch_class(
        'missense_variant[molecular consequence] AND (benign[clinsig] OR likely_benign[clinsig])',
        0, N_PER_CLASS,
    )

    all_rows = path_rows + benign_rows
    print(f"\nToplam geçerli satır: {len(all_rows)} (Patojenik={len(path_rows)}, Benign={len(benign_rows)})")

    with open(MODELS_DIR / "clinvar_grantham_sample.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False)

    X = np.array([[r["grantham"]] for r in all_rows])
    y = np.array([r["label"] for r in all_rows])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    cv_auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    print(f"\nDış model (yalnızca Grantham mesafesi -> ClinVar patojenite): "
          f"5-kat CV ROC-AUC = {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")

    clf.fit(X, y)
    joblib.dump(clf, MODELS_DIR / "clinvar_grantham_model.joblib")
    print("Kaydedildi:", MODELS_DIR / "clinvar_grantham_model.joblib")

    # Hızlı okunabilirlik: birkaç Grantham değeri için tahmini olasılık
    for g in [0, 20, 50, 100, 150, 200]:
        p = clf.predict_proba([[g]])[0, 1]
        print(f"  Grantham={g:4d} -> ClinVar-bazlı P(patojenik)={p:.3f}")


if __name__ == "__main__":
    main()
