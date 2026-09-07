import os
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    matthews_corrcoef, average_precision_score, brier_score_loss
)

class GenvisionTrainer:
    def __init__(self, panel_name):
        self.panel_name = panel_name
        self.best_threshold = 0.5
        self.seeds = [42, 123, 2024] 
        os.makedirs('sonuclar', exist_ok=True)

    def feature_engineering(self, df):
        al_cols = [c for c in df.columns if c.startswith('AL_')]
        ek_cols = [c for c in df.columns if c.startswith('EK_')]
        
        if al_cols:
            df['AL_missing_count'] = df[al_cols].isna().sum(axis=1)
        if ek_cols:
            df['EK_missing_count'] = df[ek_cols].isna().sum(axis=1)
            
        return df

    def optimize_threshold(self, y_true, y_prob):
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1, best_thresh = 0, 0.5
        for thresh in thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, average='macro')
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        return best_thresh

    def plot_feature_importance(self, model, feature_names):
        """Modelin en çok önem verdiği ilk 15 özelliği görselleştirip kaydeder."""
        try:
            importances = model.feature_importances_
            indices = np.argsort(importances)[-15:] # En önemli 15 özellik
            
            plt.figure(figsize=(10, 6))
            plt.title(f'GenVision - Feature Importance ({self.panel_name})', fontsize=12, fontweight='bold')
            plt.barh(range(len(indices)), importances[indices], color='skyblue', align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Önem Düzeyi (Importance)')
            plt.tight_layout()
            
            chart_path = os.path.join('sonuclar', f'feature_importance_{self.panel_name}.png')
            plt.savefig(chart_path, dpi=300)
            plt.close()
        except Exception as e:
            print(f"Grafik çizilirken hata oluştu: {e}")

    def train_evaluate(self, df):
        print(f"\n{'='*50}")
        print(f"=== {self.panel_name} PANELİ STACKING + FEATURE IMPORTANCE EĞİTİMİ ===")
        
        df = self.feature_engineering(df)
        
        target_col = 'Label'
        y = df[target_col].values
        X = df.drop(columns=[target_col])

        if 'Variant_ID' in X.columns:
            X = X.drop(columns=['Variant_ID'])

        # Kategorik verileri sayısal koda dönüştür
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype(str)
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

        feature_names = X.columns.tolist()
        num_neg, num_pos = np.sum(y == 0), np.sum(y == 1)
        spw = num_neg / num_pos if num_pos > 0 else 1.0
        
        seed_f1s, seed_mccs, seed_praucs, seed_briers = [], [], [], []
        last_xgb_model = None

        for seed in self.seeds:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_preds = np.zeros(len(y))

            for train_idx, val_idx in skf.split(X, y):
                X_train, y_train = X.iloc[train_idx], y[train_idx]
                X_val, y_val = X.iloc[val_idx], y[val_idx]

                # 1. Model: XGBoost
                xgb_model = xgb.XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    scale_pos_weight=spw,
                    random_state=seed,
                    missing=np.nan,
                    enable_categorical=False,
                    early_stopping_rounds=30,
                    eval_metric="logloss"
                )
                xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
                last_xgb_model = xgb_model # Grafik için son modeli tutuyoruz

                # 2. Model: LightGBM
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=5,
                    scale_pos_weight=spw,
                    random_state=seed,
                    verbose=-1
                )
                lgb_model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
                )
                lgb_probs = lgb_model.predict_proba(X_val)[:, 1]

                oof_preds[val_idx] = (0.6 * xgb_probs) + (0.4 * lgb_probs)

            thresh = self.optimize_threshold(y, oof_preds)
            final_preds = (oof_preds >= thresh).astype(int)

            seed_f1s.append(f1_score(y, final_preds, average='macro'))
            seed_mccs.append(matthews_corrcoef(y, final_preds))
            seed_praucs.append(average_precision_score(y, oof_preds))
            seed_briers.append(brier_score_loss(y, oof_preds))

        # Panel eğitimi bitince son model üzerinden öznitelik önem grafiğini çıkar
        if last_xgb_model is not None:
            self.plot_feature_importance(last_xgb_model, feature_names)

        print(f"\n[STABİLİTE ANALİZİ - GÖRSELLEŞTİRİLDİ] {self.panel_name}")
        print(f">> Macro F1:    {np.mean(seed_f1s):.4f} ± {np.std(seed_f1s):.4f}")
        print(f">> MCC:         {np.mean(seed_mccs):.4f} ± {np.std(seed_mccs):.4f}")
        print(f">> PR-AUC:      {np.mean(seed_praucs):.4f} ± {np.std(seed_praucs):.4f}")
        print(f">> Brier Score: {np.mean(seed_briers):.4f} ± {np.std(seed_briers):.4f}")
        print(f">> [BİLGİ] Öznitelik Önem Grafiği 'sonuclar/' klasörüne kaydedildi.")
        print(f"{'='*50}")

        return oof_preds