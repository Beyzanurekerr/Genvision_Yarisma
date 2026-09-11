"""
GenVision - Final Eğitim (Optuna ile ayarlanmış hiperparametrelerle)
train.py ile aynı CV/OOF iskeleti, ama panel+model başına best_hparams.json'dan
gelen ayarlanmış hiperparametreler kullanılır.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from features import GenVisionFeaturizer, row_group_ids

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

SEEDS = [42, 123, 456]
N_FOLDS = 5

PANELS = {
    "MASTER": {"file": "YARISMA_TRAIN_MASTER.csv", "test_pat": 500, "test_ben": 3000},
    "KANSER": {"file": "YARISMA_TRAIN_KANSER.csv", "test_pat": 100, "test_ben": 500},
    "PAH":    {"file": "YARISMA_TRAIN_PAH.csv",    "test_pat": 100, "test_ben": 250},
    "CFTR":   {"file": "YARISMA_TRAIN_CFTR.csv",   "test_pat": 20,  "test_ben": 100},
}

with open(OUT_DIR / "best_hparams.json") as f:
    BEST_HPARAMS = json.load(f)


def build_models(panel):
    hp = BEST_HPARAMS[panel]
    xgb_clf = xgb.XGBClassifier(**hp["xgb"], tree_method="hist", eval_metric="logloss",
                                 n_jobs=-1, random_state=42)
    lgb_clf = lgb.LGBMClassifier(**hp["lgb"], class_weight="balanced",
                                  n_jobs=-1, random_state=42, verbosity=-1)
    rf_clf = RandomForestClassifier(**hp["rf"], class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    cat_clf = CatBoostClassifier(**hp["cat"], auto_class_weights="Balanced",
                                  random_seed=42, verbose=False, thread_count=-1)
    return xgb_clf, lgb_clf, rf_clf, cat_clf


def run_panel(panel_name, cfg):
    df = pd.read_csv(DATA_DIR / cfg["file"])
    y = df["Label"].astype(int).values
    n = len(df)
    groups = row_group_ids(df)  # tekrarlı satırlar hep aynı fold'a düşsün (hoca uyarısı)

    oof_xgb = np.zeros((n, len(SEEDS)))
    oof_lgb = np.zeros((n, len(SEEDS)))
    oof_rf = np.zeros((n, len(SEEDS)))
    oof_cat = np.zeros((n, len(SEEDS)))

    for si, seed in enumerate(SEEDS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
            df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
            y_tr = y[tr_idx]

            feat = GenVisionFeaturizer().fit(df_tr)
            X_tr_native = feat.transform_native(df_tr)
            X_va_native = feat.transform_native(df_va)
            X_tr_imp = feat.transform_imputed(df_tr)
            X_va_imp = feat.transform_imputed(df_va)

            xgb_clf, lgb_clf, rf_clf, cat_clf = build_models(panel_name)

            xgb_clf.fit(X_tr_native, y_tr)
            lgb_clf.fit(X_tr_native, y_tr)
            rf_clf.fit(X_tr_imp, y_tr)
            cat_clf.fit(X_tr_native, y_tr)  # CatBoost NaN'ı native destekliyor

            oof_xgb[va_idx, si] = xgb_clf.predict_proba(X_va_native)[:, 1]
            oof_lgb[va_idx, si] = lgb_clf.predict_proba(X_va_native)[:, 1]
            oof_rf[va_idx, si] = rf_clf.predict_proba(X_va_imp)[:, 1]
            oof_cat[va_idx, si] = cat_clf.predict_proba(X_va_native)[:, 1]

    p_xgb, p_lgb, p_rf, p_cat = (oof_xgb.mean(axis=1), oof_lgb.mean(axis=1),
                                  oof_rf.mean(axis=1), oof_cat.mean(axis=1))

    meta_X = np.column_stack([p_xgb, p_lgb, p_rf, p_cat])
    p_stack = np.zeros(n)
    skf_meta = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf_meta.split(meta_X, y, groups=groups):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(meta_X[tr_idx], y[tr_idx])
        p_stack[va_idx] = meta.predict_proba(meta_X[va_idx])[:, 1]

    # Basit ortalama blend'ler -- mevcut adaylara EK olarak eklenir (threshold.py'de aynı
    # test-bilinçli seçim mantığıyla yarışırlar; performans göstermezlerse seçilmezler).
    base = {"xgb": p_xgb, "lgb": p_lgb, "rf": p_rf, "cat": p_cat}
    blends = {}
    for a, b in [("xgb", "lgb"), ("xgb", "rf"), ("xgb", "cat"),
                 ("lgb", "rf"), ("lgb", "cat"), ("rf", "cat")]:
        blends[f"Blend_{a.upper()}_{b.upper()}"] = (base[a] + base[b]) / 2
    blends["Blend_All3"] = (p_xgb + p_lgb + p_rf) / 3
    blends["Blend_All4"] = (p_xgb + p_lgb + p_rf + p_cat) / 4

    results = {}
    all_named = [("XGBoost", p_xgb), ("LightGBM", p_lgb), ("RandomForest", p_rf),
                 ("CatBoost", p_cat), ("Stacking", p_stack)] + list(blends.items())
    for name, p in all_named:
        pred05 = (p >= 0.5).astype(int)
        results[name] = {
            "cv_f1_at_0.5": float(f1_score(y, pred05)),
            "cv_mcc_at_0.5": float(matthews_corrcoef(y, pred05)),
            "cv_balanced_acc_at_0.5": float(balanced_accuracy_score(y, pred05)),
            "roc_auc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
        }

    savez_kwargs = {f"p_blend_{k[len('Blend_'):].lower()}": v for k, v in blends.items()}
    np.savez(OUT_DIR / f"{panel_name}_oof.npz", y=y, p_xgb=p_xgb, p_lgb=p_lgb, p_rf=p_rf,
             p_cat=p_cat, p_stack=p_stack, **savez_kwargs)
    return results


def main():
    import sys
    only_panel = sys.argv[1] if len(sys.argv) > 1 else None
    results_path = OUT_DIR / "final_cv_results.json"
    all_results = json.load(open(results_path)) if results_path.exists() else {}

    for panel_name, cfg in PANELS.items():
        if only_panel and panel_name != only_panel:
            continue
        print(f"\n=== {panel_name} (final, tuned) ===")
        res = run_panel(panel_name, cfg)
        all_results[panel_name] = res
        for model_name, m in res.items():
            print(f"{model_name:14s} F1={m['cv_f1_at_0.5']:.3f}  MCC={m['cv_mcc_at_0.5']:.3f}  "
                  f"ROC-AUC={m['roc_auc']:.3f}  AUPRC={m['auprc']:.3f}")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\nKaydedildi:", results_path)


if __name__ == "__main__":
    main()
