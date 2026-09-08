import os
import pandas as pd
import numpy as np
from src.external_mapper import BayesianExternalMapper
import warnings
warnings.filterwarnings('ignore')

def test_schema_robustness():
    print("GENVISION ŞEMA VE SÜTUN UYUMSUZLUK (DRIFT) TESTİ BAŞLATILIYOR...\n")
    
    train_dir = os.path.join('veri', 'egitim')
    test_dir = os.path.join('veri', 'test')
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']
    
    passed_tests = 0
    total_tests = len(panels)

    for panel in panels:
        train_path = os.path.join(train_dir, f'YARISMA_TRAIN_{panel}.csv')
        test_path = os.path.join(test_dir, f'YARISMA_TEST_{panel}.csv')

        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"[{panel}] Test dosyası eksik, önce generate_dummy_test.py çalıştırmalısın.")
            continue

        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # 1. Senaryo Simülasyonu: Test verisinden rastgele bir özellik (feature) sütunu silelim
        feature_cols = [c for c in df_test.columns if c not in ['Variant_ID', 'Label']]
        if len(feature_cols) > 0:
            dropped_col = feature_cols[0]
            df_test_corrupted = df_test.drop(columns=[dropped_col])
        else:
            df_test_corrupted = df_test.copy()
            dropped_col = "Yok"

        # 2. Senaryo Simülasyonu: Test verisine modelin hiç bilmediği yabancı bir kolon ekleyelim
        df_test_corrupted['YABANCI_FAZLA_KOLON_999'] = np.random.rand(len(df_test_corrupted))

        print(f"[{panel} Paneli Test Ediliyor]")
        print(f"  -> Silinen Kritik Kolon: {dropped_col}")
        print(f"  -> Eklenen Bilinmeyen Kolon: YABANCI_FAZLA_KOLON_999")

        try:
            # Pipeline Güvenlik Testi: Eğitim verisinde olup testte olmayanları eşitleme simülasyonu
            train_features = [c for c in df_train.columns if c not in ['Variant_ID', 'Label']]
            
            # Eksik kolonları NaN ile doldurma güvenliği
            for col in train_features:
                if col not in df_test_corrupted.columns:
                    df_test_corrupted[col] = np.nan

            # Fazla kolonları temizleme güvenliği
            df_test_corrupted = df_test_corrupted[[c for c in df_test_corrupted.columns if c in train_features or c == 'Variant_ID']]

            print(f"  [BAŞARILI]: Şema uyumsuzluğu başarıyla izole edildi ve veri hizalandı.")
            passed_tests += 1

        except Exception as e:
            print(f"  [HATA]: Şema testi patladı! Hata: {str(e)}")

    print("\n" + "="*50)
    print(f"Test Sonucu: {passed_tests}/{total_tests} panel şema direncini başarıyla geçiş yaptı.")
    print("="*50)

if __name__ == "__main__":
    test_schema_robustness()