"""
GenVision - SMOTE Deneyi (Dış/Sentetik Veri Sayfası için)
============================================================
Organizatör Q&A'da açıkça onaylanan sentetik veri çoğaltmanın (SMOTE) panel
başına CV performansına etkisini kontrollü şekilde test eder.

ÖNEMLİ metodolojik nokta: SMOTE SADECE train fold'una uygulanır (val fold'una
asla) -- aksi halde val'de sentetik komşuları olan gerçek örnekler sızıntıya
yol açar. Ayrıca SMOTE, imputed (NaN'siz) öznitelik uzayında çalışmak zorunda
(k-NN mesafesi NaN ile hesaplanamaz) -- bu yüzden yalnızca RandomForest/imputed
görünüm üzerinde test edilir; XGBoost/LightGBM'in native-missing avantajını
bozmamak için SMOTE onlara uygulanmaz.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from features import GenVisionFeaturizer, row_group_ids

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

PANELS = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}
N_FOLDS = 5
SEED = 101  # tune ile aynı (final-eval seed'lerinden ayrı, tutarlılık için)


def run_variant(df, y, groups, use_smote, k_neighbors=5):
    n = len(df)
    oof = np.zeros(n)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr = y[tr_idx]

        feat = GenVisionFeaturizer().fit(df_tr)
        X_tr = feat.transform_imputed(df_tr).values
        X_va = feat.transform_imputed(df_va).values

        if use_smote:
            pos = int(y_tr.sum())
            k = min(k_neighbors, max(1, pos - 1))
            if pos >= 2:  # SMOTE en az 2 azınlık örneği ister
                sm = SMOTE(random_state=SEED, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

        clf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                                      class_weight=None if use_smote else "balanced",
                                      n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)
        oof[va_idx] = clf.predict_proba(X_va)[:, 1]

    pred05 = (oof >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, pred05)),
        "mcc": float(matthews_corrcoef(y, pred05)),
        "roc_auc": float(roc_auc_score(y, oof)),
    }


def main():
    print(f"{'Panel':8s} {'Varyant':16s} {'F1':7s} {'MCC':7s} {'ROC-AUC':8s}")
    results = {}
    for panel, fname in PANELS.items():
        df = pd.read_csv(DATA_DIR / fname)
        y = df["Label"].astype(int).values
        groups = row_group_ids(df)

        baseline = run_variant(df, y, groups, use_smote=False)
        smote = run_variant(df, y, groups, use_smote=True)

        results[panel] = {"baseline_class_weight": baseline, "smote": smote}
        print(f"{panel:8s} {'class_weight=balanced':16s} {baseline['f1']:.3f}  {baseline['mcc']:.3f}  {baseline['roc_auc']:.3f}")
        print(f"{panel:8s} {'SMOTE':16s} {smote['f1']:.3f}  {smote['mcc']:.3f}  {smote['roc_auc']:.3f}")
        delta_f1 = smote['f1'] - baseline['f1']
        print(f"{'':8s} {'Δ F1':16s} {delta_f1:+.3f}\n")

    import json
    with open(OUT_DIR / "smote_experiment.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Kaydedildi:", OUT_DIR / "smote_experiment.json")


if __name__ == "__main__":
    main()
