import os
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, log_loss
from src.external_mapper import BayesianExternalMapper
import warnings
warnings.filterwarnings('ignore')

def check_model_overfitting():
    train_dir = os.path.join('veri', 'egitim')
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']
    
    print("GENVISION OVERFITTING VE GENELLEME KONTROL RAPORU\n" + "="*50)

    for panel in panels:
        train_path = os.path.join(train_dir, f'YARISMA_TRAIN_{panel}.csv')
        if not os.path.exists(train_path):
            continue

        df_train = pd.read_csv(train_path)
        if 'Variant_ID' in df_train.columns:
            df_train = df_train.drop(columns=['Variant_ID'])

        # Dış Veri Haritalaması
        mapper = BayesianExternalMapper(weight=10)
        df_train_mapped = mapper.fit_transform(df_train, target_col='Label')

        # Eksik Veri Analizi Sütunları
        al_cols = [c for c in df_train_mapped.columns if c.startswith('AL_')]
        ek_cols = [c for c in df_train_mapped.columns if c.startswith('EK_')]
        if al_cols:
            df_train_mapped['AL_missing_count'] = df_train_mapped[al_cols].isna().sum(axis=1)
        if ek_cols:
            df_train_mapped['EK_missing_count'] = df_train_mapped[ek_cols].isna().sum(axis=1)

        y = df_train_mapped['Label'].values
        X = df_train_mapped.drop(columns=['Label'])

        # Kategorik verileri sayısal formata çevirme
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category')

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        train_f1s, val_f1s = [], []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]

            num_neg, num_pos = np.sum(y_tr == 0), np.sum(y_tr == 1)
            spw = num_neg / num_pos if num_pos > 0 else 1.0

            # XGBoost Model Eğitimi
            model = xgb.XGBClassifier(
                n_estimators=150, learning_rate=0.05, max_depth=4,
                scale_pos_weight=spw, random_state=42, enable_categorical=True, eval_metric="logloss"
            )
            model.fit(X_tr, y_tr)

            # Train ve Validation F1 Skorları
            tr_preds = (model.predict_proba(X_tr)[:, 1] >= 0.5).astype(int)
            va_preds = (model.predict_proba(X_va)[:, 1] >= 0.5).astype(int)

            train_f1s.append(f1_score(y_tr, tr_preds, average='macro'))
            val_f1s.append(f1_score(y_va, va_preds, average='macro'))

        avg_train_f1 = np.mean(train_f1s)
        avg_val_f1 = np.mean(val_f1s)
        gap = avg_train_f1 - avg_val_f1

        print(f"\n[{panel} Paneli]")
        print(f"  Ortalama Train Macro F1: {avg_train_f1:.4f}")
        print(f"  Ortalama Valid Macro F1: {avg_val_f1:.4f}")
        print(f"  Performans Farkı (Gap):   {gap:.4f}")
        
        if gap < 0.03:
            print("  [DURUM]: Mükemmel genelleme! Overfitting riski YOK.")
        else:
            print("  [DURUM]: Hafif ezberleme eğilimi var, regularization artırılabilir.")

    print("\n" + "="*50 + "\nAnaliz tamamlandı.")

if __name__ == "__main__":
    check_model_overfitting()