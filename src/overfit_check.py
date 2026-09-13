# -*- coding: utf-8 -*-
"""
GenVision - Asiri Ogrenme (Overfitting) Denetimi
==================================================
train_final.py ile AYNI CV iskeletini (StratifiedGroupKFold, best_hparams.json)
kullanir, ama iki ek olcum yapar:

1) TRAIN-vs-OOF BOSLUGU: her fold'da modelin KENDI train foldundaki skoru ile
   gormedigi val (OOF) foldundaki skoru karsilastirilir. Buyuk bosluk (train
   yuksek, OOF dusuk) = asiri ogrenme sinyali.
2) ETIKET KARISTIRMA (permutation) TESTI: y rastgele karistirilir, ayni CV
   pipeline'i tekrar calistirilir. Gercek sinyal yoksa OOF AUC ~0.50 civarinda
   kalmali; belirgin sekilde yuksekse (ornegin >0.60) bu sizinti ya da
   modelin gurultuyu ezberledigine isaret eder.

Kullanim:
    cd src && python overfit_check.py            # tum paneller
    cd src && python overfit_check.py CFTR        # tek panel
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from features import GenVisionFeaturizer, row_group_ids

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

SEED = 42
N_FOLDS = 5

PANELS = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}

with open(OUT_DIR / "best_hparams.json") as f:
    BEST_HPARAMS = json.load(f)


def build_models(panel):
    hp = BEST_HPARAMS[panel]
    xgb_clf = xgb.XGBClassifier(**hp["xgb"], tree_method="hist", eval_metric="logloss",
                                 n_jobs=-1, random_state=SEED)
    lgb_clf = lgb.LGBMClassifier(**hp["lgb"], class_weight="balanced",
                                  n_jobs=-1, random_state=SEED, verbosity=-1)
    rf_clf = RandomForestClassifier(**hp["rf"], class_weight="balanced",
                                     n_jobs=-1, random_state=SEED)
    cat_clf = CatBoostClassifier(**hp["cat"], auto_class_weights="Balanced",
                                  random_seed=SEED, verbose=False, thread_count=-1)
    return {"XGBoost": xgb_clf, "LightGBM": lgb_clf, "RandomForest": rf_clf, "CatBoost": cat_clf}


def _scores(y_true, p):
    pred05 = (p >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y_true, p)
    except ValueError:
        auc = np.nan
    return {
        "auc": auc,
        "f1": f1_score(y_true, pred05, zero_division=0),
        "mcc": matthews_corrcoef(y_true, pred05) if len(set(pred05)) > 1 else 0.0,
    }


def run_cv(df, y, groups, model_builders_fn, panel_name):
    """Tek seed, N_FOLDS StratifiedGroupKFold. Her fold icin train-fold VE
    val-fold (OOF) tahminlerini toplar. Doner: {model_ad: {"train": [...], "oof_y":.., "oof_p":..}}"""
    n = len(df)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    model_names = list(model_builders_fn(panel_name).keys())
    oof_p = {m: np.zeros(n) for m in model_names}
    train_fold_scores = {m: [] for m in model_names}

    for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        feat = GenVisionFeaturizer().fit(df_tr)
        X_tr_native = feat.transform_native(df_tr)
        X_va_native = feat.transform_native(df_va)
        X_tr_imp = feat.transform_imputed(df_tr)
        X_va_imp = feat.transform_imputed(df_va)

        models = model_builders_fn(panel_name)
        for name, clf in models.items():
            X_tr = X_tr_imp if name == "RandomForest" else X_tr_native
            X_va = X_va_imp if name == "RandomForest" else X_va_native
            clf.fit(X_tr, y_tr)
            p_tr = clf.predict_proba(X_tr)[:, 1]
            p_va = clf.predict_proba(X_va)[:, 1]
            train_fold_scores[name].append(_scores(y_tr, p_tr))
            oof_p[name][va_idx] = p_va

    out = {}
    for name in model_names:
        tr_scores = train_fold_scores[name]
        tr_mean = {k: float(np.mean([s[k] for s in tr_scores])) for k in ("auc", "f1", "mcc")}
        oof_scores = _scores(y, oof_p[name])
        out[name] = {"train": tr_mean, "oof": oof_scores}
    return out


def overfit_gap_report(panel_name, fname):
    df = pd.read_csv(DATA_DIR / fname)
    y = df["Label"].astype(int).values
    groups = row_group_ids(df)

    print(f"\n=== {panel_name} (n={len(df)}, pozitif={int(y.sum())}) ===")
    res = run_cv(df, y, groups, build_models, panel_name)
    for name, r in res.items():
        gap_auc = r["train"]["auc"] - r["oof"]["auc"]
        gap_mcc = r["train"]["mcc"] - r["oof"]["mcc"]
        flag = "  <-- BUYUK BOSLUK (olasi asiri ogrenme)" if gap_auc > 0.15 else ""
        print(f"  {name:14s} train AUC={r['train']['auc']:.3f} MCC={r['train']['mcc']:.3f}  |  "
              f"OOF AUC={r['oof']['auc']:.3f} MCC={r['oof']['mcc']:.3f}  |  "
              f"bosluk(AUC)={gap_auc:+.3f} bosluk(MCC)={gap_mcc:+.3f}{flag}")
    return res


def permutation_test(panel_name, fname, rng_seed=42):
    """Etiketleri karistirip ayni CV'yi tekrar calistirir. Gercek sinyal yoksa
    OOF AUC ~0.50 olmali."""
    df = pd.read_csv(DATA_DIR / fname)
    y = df["Label"].astype(int).values
    groups = row_group_ids(df)

    rng = np.random.RandomState(rng_seed)
    y_shuffled = rng.permutation(y)

    res = run_cv(df, y_shuffled, groups, build_models, panel_name)
    print(f"  [KARISTIRMA TESTI] {panel_name}:")
    for name, r in res.items():
        auc = r["oof"]["auc"]
        flag = "  <-- SUPHELI (gurultuyu ogreniyor / sizinti)" if auc > 0.60 else ("  (beklenen araliktan biraz yuksek)" if auc > 0.55 else "  (temiz -- sans seviyesinde)")
        print(f"    {name:14s} karistirilmis-OOF AUC={auc:.3f}{flag}")
    return res


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    gap_summary = {}
    perm_summary = {}
    for panel_name, fname in PANELS.items():
        if only and panel_name != only:
            continue
        gap_summary[panel_name] = overfit_gap_report(panel_name, fname)

    print("\n" + "=" * 70)
    print("ETIKET KARISTIRMA (permutation) TESTI -- her panel icin ayri CV")
    print("=" * 70)
    for panel_name, fname in PANELS.items():
        if only and panel_name != only:
            continue
        perm_summary[panel_name] = permutation_test(panel_name, fname)

    def jsonable(d):
        return {p: {m: {"train": r["train"], "oof": r["oof"]} for m, r in models.items()}
                for p, models in d.items()}

    with open(OUT_DIR / "overfit_check.json", "w") as f:
        json.dump({"gap": jsonable(gap_summary), "permutation": jsonable(perm_summary)},
                   f, indent=2, ensure_ascii=False)
    print("\nKaydedildi:", OUT_DIR / "overfit_check.json")


if __name__ == "__main__":
    main()
