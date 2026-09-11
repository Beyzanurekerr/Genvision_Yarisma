# -*- coding: utf-8 -*-
"""
GenVision - CatBoost Tarama Denemesi (Model Çeşitliliği)
============================================================
MASTER'da XGBoost/LightGBM/RandomForest hiperparametre araması tavana ulaştı
(120 deneme, 60 denemeyle birebir aynı optimum). Yeni bir model ailesi
(CatBoost) gerçek çeşitlilik katabilir mi diye HIZLI bir tarama (varsayılan/
makul parametrelerle, tam Optuna değil) yapılır -- yalnızca işe yararsa
tam entegrasyon (tune+train_final+blend) yapılacak.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from features import GenVisionFeaturizer, row_group_ids

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

PANELS = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv", "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv", "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
N_FOLDS = 5
SEED = 101


def run_catboost(df, y, groups):
    n = len(df)
    oof = np.zeros(n)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr = y[tr_idx]

        feat = GenVisionFeaturizer().fit(df_tr)
        X_tr = feat.transform_native(df_tr)  # CatBoost NaN'ı native destekliyor
        X_va = feat.transform_native(df_va)

        clf = CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.03, l2_leaf_reg=3.0,
            auto_class_weights="Balanced", random_seed=SEED, verbose=False,
            thread_count=-1,
        )
        clf.fit(X_tr, y_tr)
        oof[va_idx] = clf.predict_proba(X_va)[:, 1]

    pred05 = (oof >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred05)), "mcc": float(matthews_corrcoef(y, pred05)),
        "roc_auc": float(roc_auc_score(y, oof)),
    }, oof


def main():
    print(f"{'Panel':8s} {'F1@0.5':8s} {'MCC@0.5':8s} {'ROC-AUC':8s}")
    for panel, fname in PANELS.items():
        df = pd.read_csv(DATA_DIR / fname)
        y = df["Label"].astype(int).values
        groups = row_group_ids(df)
        res, oof = run_catboost(df, y, groups)
        print(f"{panel:8s} {res['f1']:<8.3f} {res['mcc']:<8.3f} {res['roc_auc']:<8.3f}")
        np.savez(ROOT / "models" / f"_catboost_screen_{panel}.npz", y=y, p=oof)


if __name__ == "__main__":
    main()
