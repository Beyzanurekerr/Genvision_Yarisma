import pandas as pd
import numpy as np

class BayesianExternalMapper:
    def __init__(self, weight=10):
        self.weight = weight
        self.global_mean = 0
        self.mapping_dict = {}
        self.aa_risk_matrix = {}

    def _build_alphamissense_proxy_matrix(self):
        """
        AlphaMissense veritabanının genel biyokimyasal özelliklerini simüle eden 
        Amino Asit Değişim Risk Matrisi (Substitution Risk Matrix).
        Hidrofobik ve yük değişimi büyük olan mutasyonlara yüksek risk (patojenik eğilim) atar.
        """
        # Örnek biyokimyasal risk skorları (In-silico proxy)
        charged = set(['R', 'K', 'D', 'E'])
        polar = set(['S', 'T', 'N', 'Q'])
        hydrophobic = set(['A', 'V', 'L', 'I', 'M', 'F', 'W', 'P'])
        
        # Tüm olası kombinasyonlar için varsayılan risk matrisi
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        for ref in amino_acids:
            for alt in amino_acids:
                if ref == alt:
                    self.aa_risk_matrix[(ref, alt)] = 0.1 # Aynı amino asit benign eğilimli
                elif (ref in charged and alt in hydrophobic) or (ref in hydrophobic and alt in charged):
                    self.aa_risk_matrix[(ref, alt)] = 0.85 # Zıt kimyasal özellik değişimi -> Yüksek Patojenik Risk
                else:
                    self.aa_risk_matrix[(ref, alt)] = 0.45 # Orta düzey risk

    def fit_transform(self, df, target_col):
        """Eğitim verisinden Biyolojik İmzalara ve AlphaMissense Proxy matrisine göre Prior hesaplar."""
        self.global_mean = df[target_col].mean()
        self._build_alphamissense_proxy_matrix()

        aa_cols = [c for c in df.columns if c.startswith('AA_')]
        al_cols = [c for c in df.columns if c.startswith('AL_')]
        sig_cols = aa_cols + al_cols

        if not sig_cols:
            df['Bayesian_Prior'] = self.global_mean
            return df

        # 1. Aşama: AlphaMissense Amino Asit Matris Skoru Türetme
        if 'AA_1' in df.columns and 'AA_2' in df.columns:
            df['AlphaMissense_Proxy'] = df.apply(
                lambda row: self.aa_risk_matrix.get((str(row['AA_1']), str(row['AA_2'])), 0.5), axis=1
            )
        else:
            df['AlphaMissense_Proxy'] = 0.5

        # 2. Aşama: Bio-Signature Bayesian Target Encoding
        df['Bio_Signature'] = df[sig_cols].apply(lambda row: '-'.join([str(val) for val in row]), axis=1)

        agg = df.groupby('Bio_Signature')[target_col].agg(['count', 'mean'])
        counts = agg['count']
        means = agg['mean']

        smoothed_means = (counts * means + self.weight * self.global_mean) / (counts + self.weight)
        self.mapping_dict = smoothed_means.to_dict()

        df['Bayesian_Prior'] = df['Bio_Signature'].map(self.mapping_dict).fillna(self.global_mean)
        
        # Bayesian Prior ile AlphaMissense Proxy skorunu birleştirerek zenginleştirilmiş dış veri sinyali üretiyoruz
        df['Enhanced_External_Score'] = (df['Bayesian_Prior'] * 0.7) + (df['AlphaMissense_Proxy'] * 0.3)
        
        return df.drop(columns=['Bio_Signature'])

    def transform(self, df):
        """Test verisine hesaplanan dış veri ve AlphaMissense matris skorlarını uygular."""
        aa_cols = [c for c in df.columns if c.startswith('AA_')]
        al_cols = [c for c in df.columns if c.startswith('AL_')]
        sig_cols = aa_cols + al_cols

        if not sig_cols:
            df['Bayesian_Prior'] = self.global_mean
            df['AlphaMissense_Proxy'] = 0.5
            df['Enhanced_External_Score'] = self.global_mean
            return df

        if 'AA_1' in df.columns and 'AA_2' in df.columns:
            df['AlphaMissense_Proxy'] = df.apply(
                lambda row: self.aa_risk_matrix.get((str(row['AA_1']), str(row['AA_2'])), 0.5), axis=1
            )
        else:
            df['AlphaMissense_Proxy'] = 0.5

        df['Bio_Signature'] = df[sig_cols].apply(lambda row: '-'.join([str(val) for val in row]), axis=1)
        
        df['Bayesian_Prior'] = df['Bio_Signature'].map(self.mapping_dict).fillna(self.global_mean)
        df['Enhanced_External_Score'] = (df['Bayesian_Prior'] * 0.7) + (df['AlphaMissense_Proxy'] * 0.3)
        
        return df.drop(columns=['Bio_Signature'])