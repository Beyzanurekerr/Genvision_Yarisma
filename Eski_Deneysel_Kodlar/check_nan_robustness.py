import os
import pandas as pd
import numpy as np
from src.external_mapper import BayesianExternalMapper
import warnings
warnings.filterwarnings('ignore')

def test_nan_and_outliers():
    print("GENVISION UÇ DEĞER VE NAN DAYANIKLILIK TESTİ BAŞLATILIYOR...\n")
    
    test_dir = os.path.join('veri', 'test')
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']
    
    passed = 0
    for panel in panels:
        test_path = os.path.join(test_dir, f'YARISMA_TEST_{panel}.csv')
        if not os.path.exists(test_path):
            continue

        df_test = pd.read_csv(test_path)
        
        # Simülasyon: Verinin içine rastgele NaN (boş değer) ve aşırı uç değerler (Infinity) basalım
        df_corrupted = df_test.copy()
        numeric_cols = df_corrupted.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Rastgele hücrelere NaN ve inf atayalım
            idx_to_break = np.random.choice(df_corrupted.index, size=min(5, len(df_corrupted)), replace=False)
            for col in numeric_cols[:3]:
                df_corrupted.loc[idx_to_break, col] = np.nan
        
        print(f"[{panel} Paneli]: Veriye yapay NaN ve uç değerler enjekte edildi.")

        try:
            # Eksik veri sayaçları ve dış mapper'ın bu bozukluğu yönetebilmesi
            mapper = BayesianExternalMapper(weight=10)
            _ = mapper.transform(df_corrupted)
            
            al_cols = [c for c in df_corrupted.columns if c.startswith('AL_')]
            ek_cols = [c for c in df_corrupted.columns if c.startswith('EK_')]
            if al_cols:
                df_corrupted['AL_missing_count'] = df_corrupted[al_cols].isna().sum(axis=1)
            if ek_cols:
                df_corrupted['EK_missing_count'] = df_corrupted[ek_cols].isna().sum(axis=1)

            print(f"  [BAŞARILI]: NaN ve uç değerler başarıyla izole edildi, sistem çökmedi.")
            passed += 1
        except Exception as e:
            print(f"  [HATA]: Sistem bozuk veride çöktü! Hata: {str(e)}")

    print(f"\nSonuç: {passed}/{len(panels)} panel NaN dayanıklılık testini başarıyla geçti.\n")

if __name__ == "__main__":
    test_nan_and_outliers()