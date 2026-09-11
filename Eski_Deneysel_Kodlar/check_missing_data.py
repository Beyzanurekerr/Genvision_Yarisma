import os
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt

def analyze_dataset_missing_values():
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']
    train_dir = os.path.join('veri', 'egitim')
    
    for panel in panels:
        filepath = os.path.join(train_dir, f'YARISMA_TRAIN_{panel}.csv')
        if not os.path.exists(filepath):
            print(f"Uyarı: {filepath} bulunamadı.")
            continue
            
        print(f"\n{'='*40}")
        print(f"ANALİZ EDİLEN PANEL: {panel}")
        print(f"{'='*40}")
        
        df = pd.read_csv(filepath)
        
        # 1. Kolon Bazlı Eksik Veri Analizi
        missing_summary = pd.DataFrame({
            'Eksik_Deger_Sayisi': df.isna().sum(),
            'Eksik_Orani_Yuzde': (df.isna().mean() * 100).round(2)
        })
        
        cols_with_missing = missing_summary[missing_summary['Eksik_Deger_Sayisi'] > 0]
        print(f"Toplam Kolon Sayısı: {df.shape[1]}")
        print(f"Eksik Değeri Olan Kolon Sayısı: {len(cols_with_missing)}")
        if not cols_with_missing.empty:
            print("\nİlk 5 eksik kolon detayı:")
            print(cols_with_missing.head())
            
        # 2. Ürettiğimiz Eksiklik Sayaçlarının Dağılımı
        al_cols = [c for c in df.columns if c.startswith('AL_')]
        ek_cols = [c for c in df.columns if c.startswith('EK_')]
        
        if al_cols:
            missing_al = df[al_cols].isna().sum(axis=1)
            print(f"\n--- {panel} AL_missing_count Dağılımı ---")
            print(missing_al.value_counts().sort_index())
            
        if ek_cols:
            missing_ek = df[ek_cols].isna().sum(axis=1)
            print(f"\n--- {panel} EK_missing_count Dağılımı ---")
            print(missing_ek.value_counts().sort_index())

        # 3. İsteğe Bağlı Görselleştirme (Matris Çıkarma)
        # Her panel için ilk 50 sütunun haritasını sonuclar/ klasörüne kaydeder
        os.makedirs('sonuclar', exist_ok=True)
        plt.figure(figsize=(12, 6))
        msno.matrix(df.iloc[:, :min(50, df.shape[1])])
        plt.title(f'Eksik Veri Haritası - {panel}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join('sonuclar', f'missing_matrix_{panel}.png'), dpi=300)
        plt.close()
        print(f">> Eksik veri matrisi görseli 'sonuclar/missing_matrix_{panel}.png' olarak kaydedildi.")

if __name__ == "__main__":
    analyze_dataset_missing_values()