# -*- coding: utf-8 -*-
"""
GenVision - Final (Deploy) Model Eğitimi
============================================
KRİTİK: 16 Eylül 2026 final günü İNTERNET YOK ve süre kısıtlı -- model o gün
eğitilmeyecek, önceden eğitilmiş ve diskte hazır olacak. train_final.py sadece
CV/OOF tahmini üretiyordu (performans ölçümü için); bu script panel başına
SEÇİLEN modeli (threshold_results.json) TÜM eğitim verisiyle yeniden eğitip
diske kaydeder (joblib) -- gerçek test verisi geldiğinde tek yapılacak şey
predict_final.py ile bu dosyaları yükleyip tahmin üretmek.

Stacking meta-öğrenicisi: temel modeller %100 veriyle yeniden eğitiliyor, ama
meta-öğrenici SIZINTISIZ kalması için train_final.py'nin zaten ürettüğü
(5-katlı OOF'tan gelen, dolayısıyla sızıntısız) p_xgb/p_lgb/p_rf OOF havuzu
üzerinde yeniden fit ediliyor -- meta-öğrenici asla temel modellerin kendi
eğitim verisini "görmüş" tahminleriyle eğitilmiyor.

Çalıştırma:
    cd src && python train_deploy.py
Çıktı:
    models/deploy/{PANEL}_deploy.joblib
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from features import GenVisionFeaturizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
DEPLOY_DIR = MODELS_DIR / "deploy"
DEPLOY_DIR.mkdir(exist_ok=True)

# test_pat/test_ben: threshold.py / train_final.py ile AYNI (PDR'deki varsayılan
# test kompozisyonu) -- predict_final.py'deki kalibrasyon kayması güvenlik ağının
# "beklenen pozitif oran" referansı buradan (expected_pos_rate) bundle'a gömülür.
PANELS = {
    "MASTER": {"file": "YARISMA_TRAIN_MASTER.csv", "test_pat": 500, "test_ben": 3000},
    "KANSER": {"file": "YARISMA_TRAIN_KANSER.csv", "test_pat": 100, "test_ben": 500},
    "PAH":    {"file": "YARISMA_TRAIN_PAH.csv",    "test_pat": 100, "test_ben": 250},
    "CFTR":   {"file": "YARISMA_TRAIN_CFTR.csv",   "test_pat": 20,  "test_ben": 100},
}

with open(MODELS_DIR / "best_hparams.json", encoding="utf-8") as f:
    HP = json.load(f)
with open(MODELS_DIR / "threshold_results.json", encoding="utf-8") as f:
    THRESH = json.load(f)


def build_models(panel):
    hp = HP[panel]
    xgb_clf = xgb.XGBClassifier(**hp["xgb"], tree_method="hist", eval_metric="logloss",
                                 n_jobs=-1, random_state=42)
    lgb_clf = lgb.LGBMClassifier(**hp["lgb"], class_weight="balanced",
                                  n_jobs=-1, random_state=42, verbosity=-1)
    rf_clf = RandomForestClassifier(**hp["rf"], class_weight="balanced",
                                     n_jobs=-1, random_state=42)
    cat_clf = CatBoostClassifier(**hp["cat"], auto_class_weights="Balanced",
                                  random_seed=42, verbose=False, thread_count=-1)
    return xgb_clf, lgb_clf, rf_clf, cat_clf


def deploy_panel(panel_name, cfg):
    df = pd.read_csv(DATA_DIR / cfg["file"])
    y = df["Label"].astype(int).values

    feat = GenVisionFeaturizer().fit(df)  # TÜM veriyle fit -- deploy için doğru davranış
    X_native = feat.transform_native(df)
    X_imp = feat.transform_imputed(df)

    xgb_clf, lgb_clf, rf_clf, cat_clf = build_models(panel_name)
    xgb_clf.fit(X_native, y)
    lgb_clf.fit(X_native, y)
    rf_clf.fit(X_imp, y)
    cat_clf.fit(X_native, y)

    # Meta-öğrenici: train_final.py'nin ürettiği sızıntısız OOF havuzunu kullanarak yeniden fit
    oof = np.load(MODELS_DIR / f"{panel_name}_oof.npz")
    meta_X = np.column_stack([oof["p_xgb"], oof["p_lgb"], oof["p_rf"], oof["p_cat"]])
    meta = LogisticRegression(max_iter=1000)
    meta.fit(meta_X, oof["y"])

    bundle = {
        "panel": panel_name,
        "featurizer": feat,
        "xgb_model": xgb_clf,
        "lgb_model": lgb_clf,
        "rf_model": rf_clf,
        "cat_model": cat_clf,
        "meta_model": meta,
        "selected_model": THRESH[panel_name]["selected_model"],
        "threshold": THRESH[panel_name]["test_threshold"],
        "expected_pos_rate": cfg["test_pat"] / (cfg["test_pat"] + cfg["test_ben"]),
        "n_train_rows": len(df),
    }
    out_path = DEPLOY_DIR / f"{panel_name}_deploy.joblib"
    joblib.dump(bundle, out_path)
    print(f"{panel_name:8s} -> {out_path.name}  (model: {bundle['selected_model']}, "
          f"eşik: {bundle['threshold']:.2f}, n_train={bundle['n_train_rows']})")


def main():
    for panel_name, cfg in PANELS.items():
        deploy_panel(panel_name, cfg)
    print("\nTüm panellerin deploy paketleri hazır:", DEPLOY_DIR)


if __name__ == "__main__":
    main()
