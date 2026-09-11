# -*- coding: utf-8 -*-
"""
GenVision - Dış Veri (ClinVar Grantham-Prior) Ablasyon Testi
================================================================
fetch_clinvar_grantham.py'nin ürettiği dış-veri-bilgili modeli (yalnızca
Grantham mesafesi -> ClinVar patojenite olasılığı) EXT_grantham_prior adında
yeni bir özellik olarak ekleyip, panel başına RandomForest ile aynı
group-aware CV altında BASELINE (özellik yok) vs +EXT_grantham_prior
karşılaştırması yapar. SMOTE ablasyonuyla aynı disiplin: yalnızca CV'de
gerçekten katkı gösterirse features.py'ye kalıcı olarak eklenir.
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from features import GenVisionFeaturizer, row_group_ids

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

PANELS = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv", "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv", "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
N_FOLDS = 5
SEED = 101

ext_model = joblib.load(MODELS_DIR / "clinvar_grantham_model.joblib")


def run_variant(df, y, groups, use_ext):
    n = len(df)
    oof = np.zeros(n)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr = y[tr_idx]

        feat = GenVisionFeaturizer().fit(df_tr)
        X_tr = feat.transform_imputed(df_tr)
        X_va = feat.transform_imputed(df_va)

        if use_ext:
            X_tr = X_tr.copy(); X_va = X_va.copy()
            X_tr["EXT_grantham_prior"] = ext_model.predict_proba(X_tr[["AA_grantham"]].values)[:, 1]
            X_va["EXT_grantham_prior"] = ext_model.predict_proba(X_va[["AA_grantham"]].values)[:, 1]

        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                      class_weight="balanced", n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)
        oof[va_idx] = clf.predict_proba(X_va)[:, 1]

    pred05 = (oof >= 0.5).astype(int)
    return {"f1": float(f1_score(y, pred05)), "mcc": float(matthews_corrcoef(y, pred05)),
            "roc_auc": float(roc_auc_score(y, oof))}


def main():
    print(f"{'Panel':8s} {'Varyant':20s} {'F1':7s} {'MCC':7s} {'ROC-AUC':8s}")
    for panel, fname in PANELS.items():
        df = pd.read_csv(DATA_DIR / fname)
        y = df["Label"].astype(int).values
        groups = row_group_ids(df)

        baseline = run_variant(df, y, groups, use_ext=False)
        with_ext = run_variant(df, y, groups, use_ext=True)

        print(f"{panel:8s} {'baseline':20s} {baseline['f1']:.3f}  {baseline['mcc']:.3f}  {baseline['roc_auc']:.3f}")
        print(f"{panel:8s} {'+EXT_grantham_prior':20s} {with_ext['f1']:.3f}  {with_ext['mcc']:.3f}  {with_ext['roc_auc']:.3f}")
        d_f1 = with_ext["f1"] - baseline["f1"]
        d_mcc = with_ext["mcc"] - baseline["mcc"]
        print(f"{'':8s} {'delta':20s} F1={d_f1:+.3f}  MCC={d_mcc:+.3f}\n")


if __name__ == "__main__":
    main()
