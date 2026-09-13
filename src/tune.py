"""
GenVision - Optuna Hiperparametre Aramasi
Panel basina, model basina (XGBoost, LightGBM, RandomForest, CatBoost) ayri arama.
Ic donguy: TEK sabit split yerine birden fazla seed'in CV skorunun ortalamasi
(hiz icin panel buyuklugune gore seed sayisi), fold-safe featurizer.
Amac metrigi: OOF ROC-AUC + OOF F1(@0.5) ortalamasi.

Asiri-ogrenme onlemleri (overfit_check.py bulgularina karsi):
  - n_estimators/iterations artik dogrudan aranmiyor -- buyuk bir tavan + erken
    durdurma (early stopping, val fold'u eval_set olarak kullanir) ile modelin
    kac agac kullanacagina veri karar veriyor (sabit yuksek sayida agaca
    zorlanmiyor).
  - Arama uzayi (max_depth, min_child_weight/min_samples_leaf, reg_lambda/
    l2_leaf_reg) panel buyuklugune gore olcekleniyor: kucuk panellerde
    (CFTR, PAH, KANSER) daha sigi/daha guclu regularizasyonlu aralik.
  - CV skoru artik TEK sabit random_state yerine birden fazla seed'in
    ortalamasi -- arama, tek bir foldun gurultusune kilitlenmesin.
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
    "KANSER": {"file": "YARISMA_TRAIN_KANSER.csv", "n_trials": 50, "timeout": 150},
    "PAH":    {"file": "YARISMA_TRAIN_PAH.csv",    "n_trials": 50, "timeout": 150},
    "CFTR":   {"file": "YARISMA_TRAIN_CFTR.csv",   "n_trials": 40, "timeout": 100},
}
N_FOLDS = 5
TUNE_SEEDS_BY_BUCKET = {"small": [101, 202], "medium": [101, 202], "large": [101]}
EARLY_STOP = {"small": 25, "medium": 35, "large": 50}
CAP_ESTIMATORS = {"small": 600, "medium": 800, "large": 1000}


def bucket(n):
    if n < 200:
        return "small"
    if n < 1000:
        return "medium"
    return "large"


def _n_rounds_used(model):
    """Erken durdurmadan sonra modelin gercekten kac agac/iterasyon kullandigini
    okur (kutuphaneye gore farkli attribute). Bulunamazsa None doner (ornegin RF)."""
    v = getattr(model, "best_iteration", None)  # XGBoost: 0-indexli
    if v is not None:
        return int(v) + 1
    v = getattr(model, "best_iteration_", None)  # LightGBM: 0-indexli
    if v is not None:
        return int(v) + 1
    v = getattr(model, "tree_count_", None)  # CatBoost: dogrudan sayim
    if v is not None:
        return int(v)
    return None


def cv_score(df, y, build_model_fn, use_native, fit_fn, b, collect_n_rounds=False):
    """Birden fazla seed uzerinde StratifiedGroupKFold OOF skoru ortalamasi.
    fit_fn(model, X_tr, y_tr, X_va, y_va) -- erken durdurma icin val fold'u
    eval_set olarak modele veren, model-tipine ozel fit cagrisi.
    collect_n_rounds=True ise (score, ortalama_kullanilan_agac_sayisi) doner --
    erken durdurmanin gercekte kac agacta durdugunu deploy'a tasimak icin."""
    groups = row_group_ids(df)
    seeds = TUNE_SEEDS_BY_BUCKET[b]
    fold_scores = []
    n_rounds_list = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        oof = np.zeros(len(df))
        degenerate = False
        for tr_idx, va_idx in sgkf.split(df, y, groups=groups):
            df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            feat = GenVisionFeaturizer().fit(df_tr)
            if use_native:
                X_tr, X_va = feat.transform_native(df_tr), feat.transform_native(df_va)
            else:
                X_tr, X_va = feat.transform_imputed(df_tr), feat.transform_imputed(df_va)
            model = build_model_fn(y_tr)
            fit_fn(model, X_tr, y_tr, X_va, y_va)
            # Egitim-fold TAHMINININ varyansina bak (val fold kucukse val
            # varyansi 0 olabilir sans eseri; ama gercekten hic dallanmamis bir
            # agac topluluğu train'de de sabit cikar -- bu daha guvenilir sinyal).
            p_tr = model.predict_proba(X_tr)[:, 1]
            if np.std(p_tr) < 1e-6:
                degenerate = True
            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            if collect_n_rounds:
                nr = _n_rounds_used(model)
                if nr is not None:
                    n_rounds_list.append(nr)
        if degenerate:
            # Cökmüs/dejenere model: agaclar hic dallanmamis, sabit olasilik
            # uretiyor (ornegin dusuk scale_pos_weight + yuksek min_child_weight
            # birlesimi kucuk panellerde butun dugumleri esikaltinda birakabilir).
            # Boyle bir kombinasyon, cogunluk sinifini sabit tahmin ettigi icin
            # F1'i yanlislikla yuksek gosterebilir -- Optuna bunu secmesin diye
            # sert cezalandiriyoruz.
            fold_scores.append(-1.0)
            continue
        auc = roc_auc_score(y, oof)
        f1 = f1_score(y, (oof >= 0.5).astype(int))
        fold_scores.append(0.5 * auc + 0.5 * f1)
    score = float(np.mean(fold_scores))
    if collect_n_rounds:
        mean_n = float(np.mean(n_rounds_list)) if n_rounds_list else None
        return score, mean_n
    return score


def _fit_xgb(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)


def _make_fit_lgb(rounds):
    def fit_lgb(model, X_tr, y_tr, X_va, y_va):
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(rounds, verbose=False), lgb.log_evaluation(0)])
    return fit_lgb


def _make_fit_cat(rounds):
    def fit_cat(model, X_tr, y_tr, X_va, y_va):
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=rounds, verbose=False)
    return fit_cat


def _fit_rf(model, X_tr, y_tr, X_va, y_va):
    model.fit(X_tr, y_tr)  # RF'de agac sayisi boosting gibi ardisik ezberlemiyor -- erken durdurma yok


def tune_xgb(df, y, n_trials, timeout, b):
    pos, neg = int(y.sum()), int(len(y) - y.sum())
    natural_spw = neg / max(pos, 1)
    depth_range = {"small": (2, 4), "medium": (2, 6), "large": (2, 8)}[b]
    # NOT: taban 1'de tutuluyor -- yuksek taban + dusuk scale_pos_weight (cogunluk
    # sinifi patojenik olan CFTR/PAH gibi panellerde natural_spw<1) agirlikli
    # Hessian toplamini esigin altina dusurup TUM agaclarin dallanmamasina yol
    # acabiliyor (olcduk: mcw>=5 + spw=0.2 -> sabit tahmin). Dejenere-koruma
    # cv_score() icinde zaten var, burada sadece arama uzayini guvenli tutuyoruz.
    mcw_range = {"small": (1, 20), "medium": (1, 20), "large": (1, 30)}[b]
    reg_range = {"small": (1.0, 15.0), "medium": (0.5, 12.0), "large": (0.1, 10.0)}[b]

    def objective(trial):
        params = dict(
            n_estimators=CAP_ESTIMATORS[b],
            early_stopping_rounds=EARLY_STOP[b],
            max_depth=trial.suggest_int("max_depth", *depth_range),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", *reg_range, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", *mcw_range),
            scale_pos_weight=natural_spw * trial.suggest_float("spw_mult", 0.2, 2.0),
        )

        def build(y_tr):
            return xgb.XGBClassifier(**params, tree_method="hist", eval_metric="logloss",
                                     n_jobs=-1, random_state=101)
        return cv_score(df, y, build, use_native=True, fit_fn=_fit_xgb, b=b)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=101))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best = dict(study.best_params)
    spw_mult = best.pop("spw_mult")
    best["scale_pos_weight"] = natural_spw * spw_mult

    def build_best(y_tr):
        return xgb.XGBClassifier(**best, n_estimators=CAP_ESTIMATORS[b], early_stopping_rounds=EARLY_STOP[b],
                                  tree_method="hist", eval_metric="logloss",
                                  n_jobs=-1, random_state=101)
    _, mean_n = cv_score(df, y, build_best, use_native=True, fit_fn=_fit_xgb, b=b, collect_n_rounds=True)
    # deploy'da early-stopping yok (eval_set icin ayri bir val parcasi yok) --
    # aramanin gercekte kac agacta durdugunu (+%15 pay) sabit n_estimators olarak tasi
    best["n_estimators"] = int(np.clip(round((mean_n or CAP_ESTIMATORS[b]) * 1.15), 30, CAP_ESTIMATORS[b]))
    return best, study.best_value


def tune_lgb(df, y, n_trials, timeout, b):
    depth_range = {"small": (2, 4), "medium": (2, 6), "large": (2, 8)}[b]
    mcs_range = {"small": (10, 40), "medium": (5, 35), "large": (2, 30)}[b]
    reg_range = {"small": (1.0, 15.0), "medium": (0.5, 12.0), "large": (0.1, 10.0)}[b]

    def objective(trial):
        params = dict(
            n_estimators=CAP_ESTIMATORS[b],
            max_depth=trial.suggest_int("max_depth", *depth_range),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", *reg_range, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", *mcs_range),
        )

        def build(y_tr):
            return lgb.LGBMClassifier(**params, class_weight="balanced", device_type="cpu",
                                       n_jobs=-1, random_state=101, verbosity=-1)
        return cv_score(df, y, build, use_native=True, fit_fn=_make_fit_lgb(EARLY_STOP[b]), b=b)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=101))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best = dict(study.best_params)

    def build_best(y_tr):
        return lgb.LGBMClassifier(**best, n_estimators=CAP_ESTIMATORS[b], class_weight="balanced",
                                   device_type="cpu", n_jobs=-1, random_state=101, verbosity=-1)
    _, mean_n = cv_score(df, y, build_best, use_native=True, fit_fn=_make_fit_lgb(EARLY_STOP[b]), b=b,
                          collect_n_rounds=True)
    best["n_estimators"] = int(np.clip(round((mean_n or CAP_ESTIMATORS[b]) * 1.15), 30, CAP_ESTIMATORS[b]))
    return best, study.best_value


def tune_rf(df, y, n_trials, timeout, b):
    depth_range = {"small": (2, 6), "medium": (2, 9), "large": (2, 12)}[b]
    leaf_range = {"small": (3, 20), "medium": (2, 18), "large": (1, 15)}[b]

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 400),
            max_depth=trial.suggest_int("max_depth", *depth_range),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", *leaf_range),
            max_features=trial.suggest_float("max_features", 0.3, 1.0),
        )

        def build(y_tr):
            return RandomForestClassifier(**params, class_weight="balanced",
                                          n_jobs=-1, random_state=101)
        return cv_score(df, y, build, use_native=False, fit_fn=_fit_rf, b=b)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=101))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    return dict(study.best_params), study.best_value


def tune_catboost(df, y, n_trials, timeout, b):
    depth_range = {"small": (2, 5), "medium": (2, 6), "large": (3, 8)}[b]
    reg_range = {"small": (3.0, 20.0), "medium": (2.0, 15.0), "large": (1.0, 10.0)}[b]

    def objective(trial):
        params = dict(
            iterations=CAP_ESTIMATORS[b],
            depth=trial.suggest_int("depth", *depth_range),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", *reg_range, log=True),
            random_strength=trial.suggest_float("random_strength", 0.0, 2.0),
            bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0),
        )

        def build(y_tr):
            return CatBoostClassifier(**params, auto_class_weights="Balanced",
                                       random_seed=101, verbose=False, thread_count=-1)
        return cv_score(df, y, build, use_native=True, fit_fn=_make_fit_cat(EARLY_STOP[b]), b=b)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=101))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    best = dict(study.best_params)

    def build_best(y_tr):
        return CatBoostClassifier(**best, iterations=CAP_ESTIMATORS[b], auto_class_weights="Balanced",
                                   random_seed=101, verbose=False, thread_count=-1)
    _, mean_n = cv_score(df, y, build_best, use_native=True, fit_fn=_make_fit_cat(EARLY_STOP[b]), b=b,
                          collect_n_rounds=True)
    best["iterations"] = int(np.clip(round((mean_n or CAP_ESTIMATORS[b]) * 1.15), 30, CAP_ESTIMATORS[b]))
    return best, study.best_value


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
        df = pd.read_csv(DATA_DIR / cfg["file"])
        y = df["Label"].astype(int).values
        b = bucket(len(df))
        print(f"\n=== {panel} (n={len(df)}, bucket={b}, n_trials={cfg['n_trials']}, "
              f"seeds={TUNE_SEEDS_BY_BUCKET[b]}{', yalnizca CatBoost' if cat_only else ''}) ===")

        if cat_only and panel in best_params:
            xgb_params, lgb_params, rf_params = (best_params[panel]["xgb"],
                                                 best_params[panel]["lgb"], best_params[panel]["rf"])
            print("XGBoost/LightGBM/RandomForest: mevcut hiperparametreler korunuyor")
        else:
            xgb_params, xgb_score = tune_xgb(df, y, cfg["n_trials"], cfg["timeout"], b)
            print(f"XGBoost    best combined-score={xgb_score:.3f}  spw={xgb_params['scale_pos_weight']:.2f}")

            lgb_params, lgb_score = tune_lgb(df, y, cfg["n_trials"], cfg["timeout"], b)
            print(f"LightGBM    best combined-score={lgb_score:.3f}")

            rf_params, rf_score = tune_rf(df, y, cfg["n_trials"], cfg["timeout"], b)
            print(f"RandomForest best combined-score={rf_score:.3f}")

        cat_params, cat_score = tune_catboost(df, y, cfg["n_trials"], cfg["timeout"], b)
        print(f"CatBoost    best combined-score={cat_score:.3f}")

        best_params[panel] = {"xgb": xgb_params, "lgb": lgb_params, "rf": rf_params, "cat": cat_params}
        with open(hparams_path, "w") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

    print("\nKaydedildi:", hparams_path)


if __name__ == "__main__":
    main()
