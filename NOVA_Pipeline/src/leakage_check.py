"""
GenVision - Sızıntı Denetimi
Her panelde her öznitelik TEK BAŞINA etiketle karşılaştırılır (train fold üzerinde,
5-kat CV ile out-of-fold AUC). Hiçbir öznitelik tek başına anormal derecede yüksek
(örn. >0.90) ayırt etme gücüne sahip olmamalı; aksi halde muhtemel sızıntı sinyalidir.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from features import GenVisionFeaturizer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

PANELS = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
SUSPICIOUS_AUC = 0.90


def univariate_auc(df, y, seed=42, n_splits=5):
    """Panelin tüm ham öznitelikleri (AL_, EK_) için tekil OOF AUC hesaplar.
    NaN'lar median ile (yalnızca train fold'undan) dolduruluyor -- tek değişkenli
    test için yeterli, tam featurizer'a gerek yok."""
    feat_cols = [c for c in df.columns if c.startswith("AL_") or c.startswith("EK_")]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_scores = {c: np.full(len(df), np.nan) for c in feat_cols}

    for tr_idx, va_idx in skf.split(df, y):
        for c in feat_cols:
            col_tr = df[c].values[tr_idx].astype(float)
            med = np.nanmedian(col_tr) if not np.all(np.isnan(col_tr)) else 0.0
            col_va = df[c].values[va_idx].astype(float)
            col_va = np.where(np.isnan(col_va), med, col_va)
            oof_scores[c][va_idx] = col_va

    results = {}
    for c in feat_cols:
        vals = oof_scores[c]
        try:
            auc = roc_auc_score(y, vals)
            auc = max(auc, 1 - auc)  # yön farketmez, ayırt etme gücü
        except ValueError:
            auc = np.nan
        results[c] = auc
    return results


def main():
    summary = {}
    for panel, fname in PANELS.items():
        df = pd.read_csv(DATA_DIR / fname)
        y = df["Label"].astype(int).values
        aucs = univariate_auc(df, y)
        aucs_clean = {k: v for k, v in aucs.items() if not np.isnan(v)}
        max_feat = max(aucs_clean, key=aucs_clean.get)
        max_auc = aucs_clean[max_feat]
        top5 = sorted(aucs_clean.items(), key=lambda x: -x[1])[:5]
        summary[panel] = {
            "max_single_feature_auc": max_auc,
            "max_feature_name": max_feat,
            "suspicious": bool(max_auc > SUSPICIOUS_AUC),
            "top5": top5,
        }
        flag = "⚠️  ŞÜPHELİ" if max_auc > SUSPICIOUS_AUC else "temiz"
        print(f"{panel:8s} en yüksek tekil AUC = {max_auc:.3f} ({max_feat})  [{flag}]")
        for name, auc in top5:
            print(f"           {name:12s} AUC={auc:.3f}")

    with open(OUT_DIR / "leakage_check.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nKaydedildi:", OUT_DIR / "leakage_check.json")


if __name__ == "__main__":
    main()
