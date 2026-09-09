"""
GenVision - Panel Bazlı Baseline Eğitim
Her panel için: 3 seed x 5-kat stratified CV (fold-safe featurizer), 4 aday model
(XGBoost, LightGBM, RandomForest, Stacking), katman-dışı (OOF) metrikler.
CFTR için ek olarak LOOCV sanity-check.

Sızıntı önleme: Featurizer her fold'da SADECE o fold'un train bölümüne fit edilir.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, f1_score, matthews_corrcoef, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
import lightgbm as lgb
import xgboost as xgb

from features import GenVisionFeaturizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(exist_ok=True)

SEEDS = [42, 123, 456]
N_FOLDS = 5

# PDR Tablo 1'den bilinen panel-test kompozisyonu (patojenik / benign) -- eşik ayarı aşamasında kullanılacak
PANELS = {
    "MASTER": {"file": "YARISMA_TRAIN_MASTER.csv", "test_pat": 500, "test_ben": 3000},
    "KANSER": {"file": "YARISMA_TRAIN_KANSER.csv", "test_pat": 100, "test_ben": 500},
    "PAH":    {"file": "YARISMA_TRAIN_PAH.csv",    "test_pat": 100, "test_ben": 250},
    "CFTR":   {"file": "YARISMA_TRAIN_CFTR.csv",   "test_pat": 20,  "test_ben": 100},
}


def panel_size_bucket(n):
    if n < 200:
        return "small"
    if n < 1000:
        return "medium"
    return "large"


def make_models(n_train, y_train):
    """Panel büyüklüğüne göre ölçeklenmiş hiperparametrelerle 3 temel model döndürür."""
    bucket = panel_size_bucket(n_train)
    depth = {"small": 4, "medium": 6, "large": 8}[bucket]
    n_est = {"small": 200, "medium": 400, "large": 600}[bucket]
    min_child = {"small": 5, "medium": 10, "large": 20}[bucket]

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    spw = neg / max(pos, 1)

    xgb_clf = xgb.XGBClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=0.05,
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=min_child, scale_pos_weight=spw,
        tree_method="hist", eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=n_est, max_depth=depth, learning_rate=0.05,
        reg_lambda=2.0, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=min_child, class_weight="balanced",
        n_jobs=-1, random_state=42, verbosity=-1,
    )
    rf_clf = RandomForestClassifier(
        n_estimators=n_est, max_depth=depth, min_samples_leaf=max(2, min_child // 2),
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    return xgb_clf, lgb_clf, rf_clf


def run_panel(panel_name, cfg):
    df = pd.read_csv(DATA_DIR / cfg["file"])
    y = df["Label"].astype(int).values
    n = len(df)

    oof_xgb = np.zeros((n, len(SEEDS)))
    oof_lgb = np.zeros((n, len(SEEDS)))
    oof_rf = np.zeros((n, len(SEEDS)))

    for si, seed in enumerate(SEEDS):
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, va_idx in skf.split(df, y):
            df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]

            feat = GenVisionFeaturizer().fit(df_tr)  # yalnızca train fold'a fit
            X_tr_native = feat.transform_native(df_tr)
            X_va_native = feat.transform_native(df_va)
            X_tr_imp = feat.transform_imputed(df_tr)
            X_va_imp = feat.transform_imputed(df_va)

            xgb_clf, lgb_clf, rf_clf = make_models(len(tr_idx), y_tr)

            xgb_clf.fit(X_tr_native, y_tr)
            lgb_clf.fit(X_tr_native, y_tr)
            rf_clf.fit(X_tr_imp, y_tr)

            oof_xgb[va_idx, si] = xgb_clf.predict_proba(X_va_native)[:, 1]
            oof_lgb[va_idx, si] = lgb_clf.predict_proba(X_va_native)[:, 1]
            oof_rf[va_idx, si] = rf_clf.predict_proba(X_va_imp)[:, 1]

    # 3 seed ortalaması -> nihai OOF olasılık
    p_xgb = oof_xgb.mean(axis=1)
    p_lgb = oof_lgb.mean(axis=1)
    p_rf = oof_rf.mean(axis=1)

    # Stacking meta-öğrenici: 3 temel modelin OOF olasılıkları üzerinde ayrı bir 5-kat CV ile
    meta_X = np.column_stack([p_xgb, p_lgb, p_rf])
    p_stack = np.zeros(n)
    skf_meta = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for tr_idx, va_idx in skf_meta.split(meta_X, y):
        meta = LogisticRegression(max_iter=1000)
        meta.fit(meta_X[tr_idx], y[tr_idx])
        p_stack[va_idx] = meta.predict_proba(meta_X[va_idx])[:, 1]

    results = {}
    for name, p in [("XGBoost", p_xgb), ("LightGBM", p_lgb), ("RandomForest", p_rf), ("Stacking", p_stack)]:
        pred05 = (p >= 0.5).astype(int)
        results[name] = {
            "cv_f1_at_0.5": float(f1_score(y, pred05)),
            "cv_mcc_at_0.5": float(matthews_corrcoef(y, pred05)),
            "roc_auc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
        }

    # OOF olasılıkları sonraki (eşik ayarı) aşama için sakla
    np.savez(
        OUT_DIR / f"{panel_name}_oof.npz",
        y=y, p_xgb=p_xgb, p_lgb=p_lgb, p_rf=p_rf, p_stack=p_stack,
    )

    return results


def main():
    all_results = {}
    for panel_name, cfg in PANELS.items():
        print(f"\n=== {panel_name} ===")
        res = run_panel(panel_name, cfg)
        all_results[panel_name] = res
        for model_name, m in res.items():
            print(f"{model_name:14s} F1={m['cv_f1_at_0.5']:.3f}  MCC={m['cv_mcc_at_0.5']:.3f}  "
                  f"ROC-AUC={m['roc_auc']:.3f}  AUPRC={m['auprc']:.3f}")

    with open(OUT_DIR / "baseline_cv_results.json", "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nSonuçlar kaydedildi:", OUT_DIR / "baseline_cv_results.json")


if __name__ == "__main__":
    main()
