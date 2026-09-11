import os
import pandas as pd
from src.external_mapper import BayesianExternalMapper
from src.model_trainer import GenvisionTrainer
import warnings
warnings.filterwarnings('ignore')

def main():
    # Eğitim verilerinizin olduğu klasör yolu
    data_dir = os.path.join('veri', 'egitim')
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']

    print("GENVISION YARIŞMA MİMARİSİ BAŞLATILIYOR...")

    for panel in panels:
        filepath = os.path.join(data_dir, f'YARISMA_TRAIN_{panel}.csv')
        
        if not os.path.exists(filepath):
            print(f"Uyarı: {filepath} bulunamadı. Lütfen CSV'yi veri/egitim klasörüne koyun.")
            continue

        # 1. Veri Okuma
        df = pd.read_csv(filepath)

        # 2. Dış Veri (Bayesian Mapping) Uygulaması
        print(f"\n[DIŞ VERİ MODÜLÜ] {panel} paneli için 'Bayesian Bio-Signature Mapping' uygulanıyor...")
        mapper = BayesianExternalMapper(weight=10)
        target_col = 'Label' # KESİN ÇÖZÜM: Son kolon yerine doğrudan 'Label' adını veriyoruz
        df = mapper.fit_transform(df, target_col=target_col)

        # 3. Model Eğitimi ve Validasyon
        trainer = GenvisionTrainer(panel_name=panel)
        trainer.train_evaluate(df)

if __name__ == "__main__":
    main()