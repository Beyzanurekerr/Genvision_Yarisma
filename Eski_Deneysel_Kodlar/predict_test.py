import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from src.external_mapper import BayesianExternalMapper
import warnings
warnings.filterwarnings('ignore')

def generate_final_json_submission():
    # --- TAKIM BİLGİLERİNİZ ---
    TEAM_NAME = "nova"
    TEAM_ID = "918091"          
    APPLICATION_ID = "918091" 
    # --------------------------

    train_dir = os.path.join('veri', 'egitim')
    test_dir = os.path.join('veri', 'test') 
    output_dir = 'sonuclar'
    os.makedirs(output_dir, exist_ok=True)
    
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']
    seeds = [42, 123, 2024]

    print("GENVISION RESMİ TEKNOFEST JSON ÇIKTI MİMARİSİ BAŞLATILIYOR...\n")

    master_predictions_list = []

    for panel in panels:
        train_path = os.path.join(train_dir, f'YARISMA_TRAIN_{panel}.csv')
        test_path = os.path.join(test_dir, f'YARISMA_TEST_{panel}.csv')

        if not os.path.exists(test_path):
            print(f"Uyarı: {test_path} bulunamadı. Lütfen test CSV'sini 'veri/test/' klasörüne koyun.")
            continue

        print(f"[{panel}] Paneli işleniyor ve tahminler üretiliyor...")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        test_ids = df_test['Variant_ID'] if 'Variant_ID' in df_test.columns else pd.Series(range(len(df_test)))

        if 'Variant_ID' in df_train.columns:
            df_train = df_train.drop(columns=['Variant_ID'])
        if 'Variant_ID' in df_test.columns:
            df_test = df_test.drop(columns=['Variant_ID'])

        # 1. Dış Veri Modülü Entegrasyonu
        mapper = BayesianExternalMapper(weight=10)
        df_train_mapped = mapper.fit_transform(df_train, target_col='Label')
        df_test_mapped = mapper.transform(df_test)

        # Özellik Mühendisliği (Eksik Veri Sayıları)
        for df_curr in [df_train_mapped, df_test_mapped]:
            al_cols = [c for c in df_curr.columns if c.startswith('AL_')]
            ek_cols = [c for c in df_curr.columns if c.startswith('EK_')]
            if al_cols:
                df_curr['AL_missing_count'] = df_curr[al_cols].isna().sum(axis=1)
            if ek_cols:
                df_curr['EK_missing_count'] = df_curr[ek_cols].isna().sum(axis=1)

        target_col = 'Label'
        y_train = df_train_mapped[target_col].values
        X_train = df_train_mapped.drop(columns=[target_col])
        X_test = df_test_mapped.copy()

        # Kategorik verileri LabelEncoder ile güvenli sayısal forma dönüştürme
        for col in X_train.select_dtypes(include=['object']).columns:
            X_train[col] = X_train[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col])
            
            # Test setinde eğitimde olmayan yeni kategori gelirse ilk sınıfa map et
            X_test[col] = X_test[col].map(lambda s: s if s in le.classes_ else le.classes_[0])
            X_test[col] = le.transform(X_test[col])

        num_neg, num_pos = np.sum(y_train == 0), np.sum(y_train == 1)
        spw = num_neg / num_pos if num_pos > 0 else 1.0

        # 2. Stacking Ensemble Tahmini (XGBoost + LightGBM)
        ensemble_probs = np.zeros(len(X_test))

        for seed in seeds:
            # XGBoost
            xgb_model = xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                scale_pos_weight=spw, random_state=seed, missing=np.nan,
                enable_categorical=False, eval_metric="logloss"
            )
            xgb_model.fit(X_train, y_train)
            xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

            # LightGBM
            lgb_model = lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=5,
                scale_pos_weight=spw, random_state=seed, verbose=-1
            )
            lgb_model.fit(X_train, y_train)
            lgb_probs = lgb_model.predict_proba(X_test)[:, 1]

            # Ağırlıklı Harmanlama (%60 XGB + %40 LGBM)
            ensemble_probs += (0.6 * xgb_probs) + (0.4 * lgb_probs)

        final_probs = ensemble_probs / len(seeds)
        final_preds = (final_probs >= 0.5).astype(int)

        # 3. Kılavuza Uygun Prediction Sözlüklerini Oluşturma
        for idx, row_id in enumerate(test_ids):
            pred_item = {
                "id": str(row_id),
                "panel": panel,
                "predicted_class": str(final_preds[idx]),
                "predicted_prob": float(round(final_probs[idx], 4))
            }
            master_predictions_list.append(pred_item)

    # 4. Resmi Teknik Şemaya Göre Tek Bir JSON Dosyası Oluşturma
    final_output_structure = {
        "team_name": TEAM_NAME,
        "team_id": TEAM_ID,
        "application_id": APPLICATION_ID,
        "competition_level": "UNIVERSITE VE UZERI",
        "predictions": master_predictions_list
    }

    json_filename = f"TEAM_{TEAM_ID}_FINAL.json"
    json_path = os.path.join(output_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_output_structure, f, ensure_ascii=False, indent=4)

    print(f"\nBAŞARILI! Resmi TEKNOFEST JSON dosyası oluşturuldu: {json_path}")

if __name__ == "__main__":
    generate_final_json_submission()