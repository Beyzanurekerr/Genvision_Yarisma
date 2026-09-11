import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.external_mapper import BayesianExternalMapper
import warnings
warnings.filterwarnings('ignore')

def inspect_bayesian_mapper():
    print("BAYESIAN EXTERNAL MAPPER GÖRSELLEŞTİRİLİYOR...\n")
    
    train_path = os.path.join('veri', 'egitim', 'YARISMA_TRAIN_MASTER.csv')
    if not os.path.exists(train_path):
        print("[HATA]: Eğitim verisi bulunamadı.")
        return

    df_train = pd.read_csv(train_path)
    
    # Mapper nesnesini çalıştıralım
    mapper = BayesianExternalMapper(weight=10)
    mapper.fit_transform(df_train, target_col='Label')
    
    if hasattr(mapper, 'mapping_dict') and mapper.mapping_dict:
        values = list(mapper.mapping_dict.values())
        print(f"Toplam Öğrenilen Biyolojik İmza Sayısı: {len(values)}")
        print(f"Ortalama Bayesian Ağırlığı: {sum(values)/len(values):.4f}")
        
        # Grafik tasarımı (Yarışma sunumları için yüksek kaliteli PNG)
        plt.figure(figsize=(9, 5), dpi=300)
        sns.histplot(values, kde=True, color='indigo', bins=30)
        plt.title("Bayesian Mapping Ağırlık Dağılımı (Bio_Signature)", fontsize=12, fontweight='bold')
        plt.xlabel("Hesaplanan Bayesian Olasılık Ağırlığı", fontsize=10)
        plt.ylabel("İmza Frekansı", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        os.makedirs("sonuclar", exist_ok=True)
        output_path = "sonuclar/mapper_distribution.png"
        plt.savefig(output_path)
        plt.close()
        
        print(f"\n[BAŞARILI]: Terminal metin kalabalığı temizlendi ve grafik PNG olarak kaydedildi!")
        print(f"Dosya Yolu: {os.path.abspath(output_path)}")
    else:
        print("[BİLGİ]: 'mapping_dict' niteliği bulunamadı.")

if __name__ == "__main__":
    inspect_bayesian_mapper()