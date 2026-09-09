# -*- coding: utf-8 -*-
"""
GenVision - Seed Kararlılığı (Danışman Talebi)
==================================================
train_final.py'de 3 seed'in (42, 123, 456) OOF olasılıkları ORTALANARAK
(bagged) tek bir nihai olasılık üretiliyor -- bu, tek-seed rastlantısallığını
zaten azaltan bir tasarım. Ancak danışman geri bildirimi açıkça "tek seed'in
rastlantısal varyansı gizlediği, en az 3-5 seed ile Ortalama ± Std
raporlanması" istiyor -- yani asıl soru "3 seed'i ortalarsak sonuç ne olur"
değil, "3 ayrı seed'in HER BİRİ tek başına ne verirdi, ne kadar SAÇILIRLAR".

Bu script, panel başına SEÇİLEN modelin (threshold_results.json) PANELE ZATEN
ATANMIŞ SABİT test-eşiğinde, her seed'i TEK BAŞINA (tam 5-katlı CV + varsa
stacking meta-adımıyla) çalıştırıp test-projekte F1/MCC'sini hesaplar, sonra
3 seed arası Ortalama ± Std raporlar. Bagged (3-seed ortalaması) sonuçla
karşılaştırma, ensemble'ın tek-seed varyansını gerçekten azalttığını
göstermek için eklenmiştir.

Çalıştırma:
    cd src && python seed_stability.py
Çıktı:
    models/seed_stability.json
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

from features import GenVisionFeaturizer, row_group_ids
from train_final import build_models, SEEDS, N_FOLDS, PANELS
from threshold import project_to_test

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

with open(OUT_DIR / "threshold_results.json", encoding="utf-8") as f:
    THRESH = json.load(f)


def single_seed_oof(df, y, groups, panel_name, seed):
    """Tek bir seed için tam 5-katlı OOF (4 temel model + stacking meta + blend'ler)."""
    n = len(df)
    oof_xgb = np.zeros(n)
    oof_lgb = np.zeros(n)
    oof_rf = np.zeros(n)
    oof_cat = np.zeros(n)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr = y[tr_idx]

        feat = GenVisionFeaturizer().fit(df_tr)
        X_tr_native, X_va_native = feat.transform_native(df_tr), feat.transform_native(df_va)
        X_tr_imp, X_va_imp = feat.transform_imputed(df_tr), feat.transform_imputed(df_va)

        xgb_clf, lgb_clf, rf_clf, cat_clf = build_models(panel_name)
        xgb_clf.fit(X_tr_native, y_tr)
        lgb_clf.fit(X_tr_native, y_tr)
        rf_clf.fit(X_tr_imp, y_tr)
        cat_clf.fit(X_tr_native, y_tr)

        oof_xgb[va_idx] = xgb_clf.predict_proba(X_va_native)[:, 1]
        oof_lgb[va_idx] = lgb_clf.predict_proba(X_va_native)[:, 1]
        oof_rf[va_idx] = rf_clf.predict_proba(X_va_imp)[:, 1]
        oof_cat[va_idx] = cat_clf.predict_proba(X_va_native)[:, 1]

    meta_X = np.column_stack([oof_xgb, oof_lgb, oof_rf, oof_cat])
    oof_stack = np.zeros(n)
    skf_meta = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf_meta.split(meta_X, y, groups=groups):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(meta_X[tr_idx], y[tr_idx])
        oof_stack[va_idx] = meta.predict_proba(meta_X[va_idx])[:, 1]

    base = {"XGB": oof_xgb, "LGB": oof_lgb, "RF": oof_rf, "CAT": oof_cat}
    letter_map = {"X": "XGB", "L": "LGB", "R": "RF", "C": "CAT"}
    out = {
        "XGBoost": oof_xgb, "LightGBM": oof_lgb, "RandomForest": oof_rf, "CatBoost": oof_cat,
        "Stacking": oof_stack,
        "Blend_All3": (oof_xgb + oof_lgb + oof_rf) / 3,
        "Blend_All4": (oof_xgb + oof_lgb + oof_rf + oof_cat) / 4,
    }
    for a, b in [("XGB", "LGB"), ("XGB", "RF"), ("XGB", "CAT"), ("LGB", "RF"), ("LGB", "CAT"), ("RF", "CAT")]:
        out[f"Blend_{a}_{b}"] = (base[a] + base[b]) / 2
    for pair in ["XL", "XR", "XC", "LR", "LC", "RC"]:
        pa, pb = base[letter_map[pair[0]]], base[letter_map[pair[1]]]
        for w in [0.3, 0.4, 0.6, 0.7]:
            out[f"WBlend_{pair}_{w:.1f}"] = w * pa + (1 - w) * pb
    return out


def main():
    summary = {}
    print(f"{'Panel':8s} {'Model':13s} {'Seed':6s} {'Test F1(proj.)':15s} {'Test MCC(proj.)':15s}")
    for panel, cfg in PANELS.items():
        df = pd.read_csv(DATA_DIR / cfg["file"])
        y = df["Label"].astype(int).values
        groups = row_group_ids(df)

        model_name = THRESH[panel]["selected_model"]
        fixed_threshold = THRESH[panel]["test_threshold"]
        test_pat, test_ben = cfg["test_pat"], cfg["test_ben"]

        seed_f1, seed_mcc = [], []
        for seed in SEEDS:
            oofs = single_seed_oof(df, y, groups, panel, seed)
            p = oofs[model_name]
            proj = project_to_test(y, p, fixed_threshold, test_pat, test_ben)
            seed_f1.append(proj["f1"])
            seed_mcc.append(proj["mcc"])
            print(f"{panel:8s} {model_name:13s} {seed:<6d} {proj['f1']:<15.3f} {proj['mcc']:<15.3f}")

        seed_f1, seed_mcc = np.array(seed_f1), np.array(seed_mcc)
        summary[panel] = {
            "model": model_name,
            "fixed_test_threshold": fixed_threshold,
            "seeds": SEEDS,
            "per_seed_f1": seed_f1.tolist(),
            "per_seed_mcc": seed_mcc.tolist(),
            "f1_mean": float(seed_f1.mean()), "f1_std": float(seed_f1.std(ddof=1)),
            "mcc_mean": float(seed_mcc.mean()), "mcc_std": float(seed_mcc.std(ddof=1)),
            "bagged_f1_reference": THRESH[panel]["projected_test_f1"],  # 3-seed OOF-ortalaması (raporda kullanılan)
            "bagged_mcc_reference": THRESH[panel]["projected_mcc"],
        }
        print(f"  -> {panel} bagged (rapordaki) F1={THRESH[panel]['projected_test_f1']:.3f} "
              f"| tek-seed ort={seed_f1.mean():.3f}±{seed_f1.std(ddof=1):.3f}\n")

    with open(OUT_DIR / "seed_stability.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("Kaydedildi:", OUT_DIR / "seed_stability.json")


if __name__ == "__main__":
    main()
