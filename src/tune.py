"""
GenVision - Optuna Hiperparametre Araması
Panel başına, model başına (XGBoost, LightGBM, RandomForest, CatBoost) ayrı arama.
İç döngü: tek 5-katlı stratified CV (hız için), fold-safe featurizer.
Amaç metriği: OOF ROC-AUC + OOF F1(@0.5) ortalaması.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb

from features import GenVisionFeaturizer, row_group_ids

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "models"

PANELS = {
    "MASTER": {"file": "YARISMA_TRAIN_MASTER.csv", "n_trials": 120, "timeout": 480},
    "KANSER": {"file": "YARISMA_TRAIN_KANSER.csv", "n_trials": 60, "timeout": 150},
    "PAH":    {"file": "YARISMA_TRAIN_PAH.csv",    "n_trials": 60, "timeout": 150},
    "CFTR":   {"file": "YARISMA_TRAIN_CFTR.csv",   "n_trials": 80, "timeout": 100},
}
N_FOLDS = 5
TUNE_SEED = 101


def cv_score(df, y, build_model_fn, use_native):
    groups = row_group_ids(df)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=TUNE_SEED)
    oof = np.zeros(len(df))
    for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
        y_tr = y[tr_idx]
        feat = GenVisionFeaturizer().fit(df_tr)
        if use_native:
            X_tr, X_va = feat.transform_native(df_tr), feat.transform_native(df_va)
        else:
            X_tr, X_va = feat.transform_imputed(df_tr), feat.transform_imputed(df_va)
        model = build_model_fn(y_tr)
        model.fit(X_tr, y_tr)
        oof[va_idx] = model.predict_proba(X_va)[:, 1]
    auc = roc_auc_score(y, oof)
    f1 = f1_score(y, (oof >= 0.5).astype(int))
    return 0.5 * auc + 0.5 * f1, oof


def tune_xgb(df, y, n_trials, timeout=None):
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    natural_spw = neg / max(pos, 1)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            scale_pos_weight=natural_spw * trial.suggest_float("spw_mult", 0.2, 2.0),
        )

        def build(y_tr):
            return xgb.XGBClassifier(**params, tree_method="hist", device="cuda", eval_metric="logloss",
                                     n_jobs=-1, random_state=TUNE_SEED)
        score, _ = cv_score(df, y, build, use_native=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best = dict(study.best_params)
    best["scale_pos_weight"] = natural_spw * best.pop("spw_mult")
    return best, study.best_value


def tune_lgb(df, y, n_trials, timeout=None):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 2, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 2, 30),
        )

        def build(y_tr):
            return lgb.LGBMClassifier(**params, class_weight="balanced", device_type="gpu",
                                        n_jobs=-1, random_state=TUNE_SEED, verbosity=-1)
        score, _ = cv_score(df, y, build, use_native=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return dict(study.best_params), study.best_value


def tune_rf(df, y, n_trials, timeout=None):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", 2, 12),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 15),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
        )

        def build(y_tr):
            return RandomForestClassifier(**params, class_weight="balanced",
                                          n_jobs=-1, random_state=TUNE_SEED)
        score, _ = cv_score(df, y, build, use_native=False)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return dict(study.best_params), study.best_value


def tune_catboost(df, y, n_trials, timeout=None):
    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 150, 500),
            depth=trial.suggest_int("depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        )

        def build(y_tr):
            return CatBoostClassifier(**params, auto_class_weights="Balanced", task_type="GPU",
                                        random_seed=TUNE_SEED, verbose=False, thread_count=-1)
        score, _ = cv_score(df, y, build, use_native=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return dict(study.best_params), study.best_value


def main():
    import sys
    args = sys.argv[1:]
    cat_only = "--cat-only" in args
    args = [a for a in args if a != "--cat-only"]
    only_panel = args[0] if args else None
    hparams_path = OUT_DIR / "best_hparams.json"
    best_params = json.load(open(hparams_path)) if hparams_path.exists() else {}

    for panel, cfg in PANELS.items():
        if only_panel and panel != only_panel:
            continue
        print(f"\n=== {panel} (n_trials={cfg['n_trials']}{', yalnızca CatBoost' if cat_only else ''}) ===")
        df = pd.read_csv(DATA_DIR / cfg["file"])
        y = df["Label"].astype(int).values

        if cat_only and panel in best_params:
            xgb_params, lgb_params, rf_params = (best_params[panel]["xgb"],
                                                 best_params[panel]["lgb"], best_params[panel]["rf"])
            print("XGBoost/LightGBM/RandomForest: mevcut hiperparametreler korunuyor")
        else:
            xgb_params, xgb_score = tune_xgb(df, y, cfg["n_trials"], cfg["timeout"])
            print(f"XGBoost    best combined-score={xgb_score:.3f}  spw={xgb_params['scale_pos_weight']:.2f}")

            lgb_params, lgb_score = tune_lgb(df, y, cfg["n_trials"], cfg["timeout"])
            print(f"LightGBM    best combined-score={lgb_score:.3f}")

            rf_params, rf_score = tune_rf(df, y, cfg["n_trials"], cfg["timeout"])
            print(f"RandomForest best combined-score={rf_score:.3f}")

        cat_params, cat_score = tune_catboost(df, y, cfg["n_trials"], cfg["timeout"])
        print(f"CatBoost    best combined-score={cat_score:.3f}")

        best_params[panel] = {"xgb": xgb_params, "lgb": lgb_params, "rf": rf_params, "cat": cat_params}
        with open(hparams_path, "w") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

    print("\nKaydedildi:", hparams_path)


if __name__ == "__main__":
    main()