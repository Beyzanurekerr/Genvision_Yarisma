import os
import pandas as pd
import numpy as np
import xgboost as xgb
from src.external_mapper import BayesianExternalMapper
import warnings
warnings.filterwarnings('ignore')

def simulate_and_test_external_data():
    print("HARİCİ/HAM VERİ FEATURE ALIGNMENT VE GENELLEME TESTİ BAŞLATILIYOR...\n")
    
    train_dir = os.path.join('veri', 'egitim')
    panel = 'MASTER'
    train_path = os.path.join(train_dir, f'YARISMA_TRAIN_{panel}.csv')
    
    if not os.path.exists(train_path):
        print(f"[HATA]: Eğitim dosyası bulunamadı.")
        return

    df_train = pd.read_csv(train_path)
    
    # 1. Ham Dış Veri Simülasyonu
    raw_external_df = df_train.sample(n=50, random_state=42).copy()

    print(f"-> Simüle edilen ham dış veri boyutu: {raw_external_df.shape}")
    print("-> Adım 1: Dış veri, BayesianExternalMapper ve Feature Engineering boru hattına sokuluyor...")

    # 2. Pipeline Dönüşümü
    mapper = BayesianExternalMapper(weight=10)
    df_mapped = mapper.fit_transform(raw_external_df, target_col='Label')

    if 'Label' in df_mapped.columns:
        df_mapped = df_mapped.drop(columns=['Label'])

    # Satır bazlı eksiklik sayaçları
    al_cols = [c for c in df_mapped.columns if c.startswith('AL_')]
    ek_cols = [c for c in df_mapped.columns if c.startswith('EK_')]
    if al_cols:
        df_mapped['AL_missing_count'] = df_mapped[al_cols].isna().sum(axis=1)
    if ek_cols:
        df_mapped['EK_missing_count'] = df_mapped[ek_cols].isna().sum(axis=1)

    # 3. Model Eğitimi Hazırlığı
    y_train = df_train['Label'].values
    X_train = df_train.drop(columns=['Variant_ID', 'Label'])

    # Eğitim verisindeki kategorik sütunları uygun tipe çevirelim
    for col in X_train.select_dtypes(include=['object']).columns:
        X_train[col] = X_train[col].astype('category')

    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, enable_categorical=True, random_state=42)
    model.fit(X_train, y_train)

    print("-> Adım 2: Model şeması ile harici veri hizalanıyor ve kategoriler güvenceye alınıyor...")

    # Eğitim sütun şemasıyla birebir hizalama ve kategori uyumluluğu
    train_features = [c for c in df_train.columns if c not in ['Variant_ID', 'Label']]
    
    X_external = pd.DataFrame(index=df_mapped.index)
    for col in train_features:
        if col in df_mapped.columns:
            X_external[col] = df_mapped[col]
        else:
            X_external[col] = np.nan

    # Kategorik sütunların kategorilerini eğitim verisiyle eşitleme (XGBoost n_categories çökmesini önler)
    for col in X_train.select_dtypes(include=['category']).columns:
        train_cats = X_train[col].cat.categories
        # Eksik veya NaN değerleri eğitimdeki ilk kategoriyle doldurup kategori tipine çevirelim
        fill_val = train_cats[0] if len(train_cats) > 0 else "Missing"
        X_external[col] = X_external[col].fillna(fill_val).astype(str)
        # Eğitim kategorilerini zorla uygula
        X_external[col] = pd.Categorical(X_external[col], categories=train_cats)

    print("-> Adım 3: Model ile harici veri üzerinden tahmin üretiliyor...")

    preds = model.predict(X_external)
    probs = model.predict_proba(X_external)[:, 1]

    print("="*50)
    print(f"[BAŞARILI]: Harici ham veri başarıyla dönüştürüldü ve hizalandı!")
    print(f"Toplam İncelenen Harici Varyant: {len(preds)}")
    print(f"Ortalama Tahmin Edilen Olasılık: {np.mean(probs):.4f}")
    print("="*50)

if __name__ == "__main__":
    simulate_and_test_external_data()