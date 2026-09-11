"""
GenVision - SHAP Kategori Bazlı Açıklanabilirlik
Her panelin seçilen (threshold_results.json'daki) modeli TÜM veri üzerinde yeniden
eğitilir (yalnızca açıklama amaçlı -- performans burada raporlanmaz, o CV/OOF'tan
geliyor), SHAP TreeExplainer ile öznitelik katkıları hesaplanır ve kategoriye
(AL_/EK_/CAT_/AA_) göre toplanır.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

from features import GenVisionFeaturizer

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

PANELS = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv",
    "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv",
    "CFTR": "YARISMA_TRAIN_CFTR.csv",
}

with open(MODELS_DIR / "threshold_results.json") as f:
    THRESH = json.load(f)
with open(MODELS_DIR / "best_hparams.json") as f:
    HP = json.load(f)


def category_of(col):
    if col.startswith("INTX_"):
        return "INTX (AL x EK etkileşim)"
    if col.startswith("AL_"):
        return "AL (frekans)"
    if col.startswith("EK_"):
        return "EK (korunmuşluk)"
    if col.startswith("CAT_"):
        return "CAT (kategorik)"
    if col.startswith("AA_"):
        return "AA (amino asit)"
    return "diğer (eksiklik özeti)"


def build_model(panel, model_name, y):
    if model_name == "XGBoost":
        return xgb.XGBClassifier(**HP[panel]["xgb"], tree_method="hist",
                                  eval_metric="logloss", n_jobs=-1, random_state=42), True
    if model_name == "LightGBM":
        return lgb.LGBMClassifier(**HP[panel]["lgb"], class_weight="balanced",
                                   n_jobs=-1, random_state=42, verbosity=-1), True
    if model_name == "RandomForest":
        return RandomForestClassifier(**HP[panel]["rf"], class_weight="balanced",
                                       n_jobs=-1, random_state=42), False
    raise ValueError(f"Stacking için doğrudan SHAP yok, temel modellerden biri seçilmeli: {model_name}")


def run_panel(panel, fname):
    df = pd.read_csv(DATA_DIR / fname)
    y = df["Label"].astype(int).values
    model_name = THRESH[panel]["selected_model"]

    feat = GenVisionFeaturizer().fit(df)
    if model_name == "Stacking":
        # Stacking'in kendisi açıklanamaz; en güçlü bileşenini (genelde XGBoost) kullan
        model_name = "XGBoost"
    X = feat.transform_native(df) if model_name in ("XGBoost", "LightGBM") else feat.transform_imputed(df)

    model, native = build_model(panel, model_name, y)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):  # bazı sürümlerde binary için [neg, pos] listesi döner
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs = np.abs(shap_values).mean(axis=0)
    contrib = pd.Series(mean_abs, index=X.columns)
    cat_contrib = contrib.groupby([category_of(c) for c in X.columns]).sum()
    cat_contrib_pct = (cat_contrib / cat_contrib.sum() * 100).round(1).sort_values(ascending=False)

    top_features = contrib.sort_values(ascending=False).head(10)
    return model_name, cat_contrib_pct.to_dict(), top_features.to_dict()


def main():
    summary = {}
    for panel, fname in PANELS.items():
        print(f"\n=== {panel} (model: ...) ===")
        model_name, cat_pct, top_feats = run_panel(panel, fname)
        summary[panel] = {"model_used": model_name, "category_contribution_pct": cat_pct,
                           "top10_features": top_feats}
        print(f"Kullanılan model: {model_name}")
        for cat, pct in cat_pct.items():
            print(f"  {cat:20s} {pct:5.1f}%")
        print("  En etkili 5 öznitelik:", list(top_feats.keys())[:5])

    with open(MODELS_DIR / "shap_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nKaydedildi:", MODELS_DIR / "shap_summary.json")


if __name__ == "__main__":
    main()
