"""
GenVision - Öznitelik Mühendisliği
Grantham mesafesi, eksiklik-sinyali öznitelikleri, kategorik kodlama.
Tüm istatistikler (medyan/sentinel/kategori haritaları) SADECE fit() sırasında
train fold'undan öğrenilir -> sızıntı yok. transform() öğrenilen istatistikleri uygular.
"""
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Grantham (1974) amino asit fizikokimyasal özellikleri: (kompozisyon, polarite, hacim)
GRANTHAM_PROPS = {
    "A": (0.00, 8.1, 31.0),  "R": (0.65, 10.5, 124.0), "N": (1.33, 11.6, 56.0),
    "D": (1.38, 13.0, 54.0), "C": (2.75, 5.5, 55.0),   "Q": (0.89, 10.5, 85.0),
    "E": (0.92, 12.3, 83.0), "G": (0.74, 9.0, 3.0),    "H": (0.58, 10.4, 96.0),
    "I": (0.00, 5.2, 111.0), "L": (0.00, 4.9, 111.0),  "K": (0.33, 11.3, 119.0),
    "M": (0.00, 5.7, 105.0), "F": (0.00, 5.2, 132.0),  "P": (0.39, 8.0, 32.5),
    "S": (1.42, 9.2, 32.0),  "T": (0.71, 8.6, 61.0),   "W": (0.13, 5.4, 170.0),
    "Y": (0.20, 6.2, 136.0), "V": (0.00, 5.9, 84.0),
}
_ALPHA, _BETA, _GAMMA = 1.833, 0.1018, 0.000399


def grantham_distance(aa1: str, aa2: str) -> float:
    """Tek harfli amino asit kodları arasında Grantham mesafesi.
    Bilinmeyen/çoklu-harfli kod (indel, delins vb.) -> NaN döner (ayrı flag'le yakalanır)."""
    if not isinstance(aa1, str) or not isinstance(aa2, str):
        return np.nan
    if aa1 not in GRANTHAM_PROPS or aa2 not in GRANTHAM_PROPS:
        return np.nan
    c1, p1, v1 = GRANTHAM_PROPS[aa1]
    c2, p2, v2 = GRANTHAM_PROPS[aa2]
    return float(np.sqrt(_ALPHA * (c1 - c2) ** 2 + _BETA * (p1 - p2) ** 2 + _GAMMA * (v1 - v2) ** 2))


AL_SENTINEL = -1.0     # frekans kolonları için "gözlenmemiş/nadir" sentinel (tüm gerçek değerler >= 0)
EK_SENTINEL = -999.0   # korunmuşluk skorları için sentinel (bazı EK kolonlarında negatif gerçek değer var, çok daha küçük bir değer seçildi)


class GenVisionFeaturizer:
    """Panel-agnostik, fold-safe öznitelik dönüştürücü.
    fit(df_train) -> yalnızca train fold'undan istatistik öğrenir.
    transform(df) -> öğrenilen istatistiklerle her iki (RF-safe imputed / tree-native NaN) görünümü üretir.
    """

    def __init__(self):
        self.al_cols = None
        self.ek_cols = None
        self.cat_cols = None
        self.aa_cols = None
        self.cat_maps = {}          # kategori -> {value: code}, bilinmeyen değerler -1
        self.ek_medians = {}        # RF-safe görünüm için EK kolonu medyanları (yalnızca train fold)
        self.fitted = False

    def _identify_columns(self, df: pd.DataFrame):
        cols = [c for c in df.columns if c not in ("Variant_ID", "Label")]
        self.al_cols = [c for c in cols if c.startswith("AL_")]
        self.ek_cols = [c for c in cols if c.startswith("EK_")]
        self.cat_cols = [c for c in cols if c.startswith("CAT_")]
        self.aa_cols = [c for c in cols if c.startswith("AA_")]

    def fit(self, df: pd.DataFrame):
        self._identify_columns(df)
        # Birebir aynı (redundant) CAT_ kolonlarını tespit et ve at (örn. CAT_5==CAT_3
        # tüm panellerde doğrulandı) -- yalnızca train fold'undan öğrenilir.
        redundant = set()
        for i, c1 in enumerate(self.cat_cols):
            if c1 in redundant:
                continue
            for c2 in self.cat_cols[i + 1:]:
                if c2 in redundant:
                    continue
                if df[c1].equals(df[c2]):
                    redundant.add(c2)
        self.cat_cols = [c for c in self.cat_cols if c not in redundant]
        self.dropped_redundant_cat = sorted(redundant)

        for c in self.cat_cols:
            uniq = df[c].dropna().unique().tolist()
            self.cat_maps[c] = {v: i for i, v in enumerate(sorted(map(str, uniq)))}
        for c in self.ek_cols:
            self.ek_medians[c] = df[c].median()  # yalnızca train fold

        # EK_ öbeği tek ölçekte değil (bkz. rapor Bölüm 2.3): [0,1] sınırlı (olasılıksal/
        # risk-skoru benzeri) kolonlarla sınırsız-sürekli (korunmuşluk-skoru benzeri)
        # kolonları AYRI toplayıp, AL_ (nadirlik) ile etkileşim özelliği üretmek için
        # ayrıştırıyoruz -- PSR'de vaat edilen "EK x AL bileşik etkileşim özelliği".
        self.ek_bounded_cols, self.ek_continuous_cols = [], []
        self.ek_cont_mean, self.ek_cont_std = {}, {}
        for c in self.ek_cols:
            vals = df[c].dropna()
            if len(vals) and vals.min() >= -1e-6 and vals.max() <= 1 + 1e-6:
                self.ek_bounded_cols.append(c)
            else:
                self.ek_continuous_cols.append(c)
                self.ek_cont_mean[c] = vals.mean() if len(vals) else 0.0
                self.ek_cont_std[c] = vals.std() if len(vals) and vals.std() > 1e-9 else 1.0

        self.fitted = True
        return self

    def _base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # --- AL (frekans) blok: eksiklik = nadirlik sinyali, ham NaN korunur (tree-native görünüm ayrı) ---
        al_df = df[self.al_cols].astype(float)
        out["AL_missing_ratio"] = al_df.isna().mean(axis=1)
        out["AL_missing_count"] = al_df.isna().sum(axis=1)
        for c in self.al_cols:
            out[c] = al_df[c]  # NaN korunur (XGB/LGBM native)

        # --- EK (korunmuşluk) blok ---
        ek_df = df[self.ek_cols].astype(float)
        out["EK_missing_ratio"] = ek_df.isna().mean(axis=1)
        for c in self.ek_cols:
            out[c] = ek_df[c]

        # --- CAT (kategorik meta-veri) blok: eksiklik ayrı seviye (-1) ---
        for c in self.cat_cols:
            mapping = self.cat_maps.get(c, {})
            out[c + "_code"] = df[c].astype(str).map(mapping).fillna(-1).astype(int)
        out["CAT_missing_ratio"] = df[self.cat_cols].isna().mean(axis=1)

        # --- AA (amino asit) blok: Grantham mesafesi + bilinmeyen-kod flag ---
        aa1 = df[self.aa_cols[0]] if len(self.aa_cols) > 0 else pd.Series(index=df.index, dtype=object)
        aa2 = df[self.aa_cols[1]] if len(self.aa_cols) > 1 else pd.Series(index=df.index, dtype=object)
        out["AA_grantham"] = [grantham_distance(a, b) for a, b in zip(aa1, aa2)]
        out["AA_unknown_code"] = out["AA_grantham"].isna().astype(int)
        out["AA_same"] = (aa1.astype(str) == aa2.astype(str)).astype(int)

        # --- Genel toplam eksiklik oranı ---
        all_feat_cols = self.al_cols + self.ek_cols
        out["TOTAL_missing_ratio"] = df[all_feat_cols].isna().mean(axis=1)

        # --- INTX (AL x EK bileşik etkileşim) blok: PSR'de vaat edilen özellik ---
        if self.ek_bounded_cols:
            out["INTX_EK_bounded_mean"] = df[self.ek_bounded_cols].astype(float).mean(axis=1, skipna=True)
        else:
            out["INTX_EK_bounded_mean"] = np.nan
        if self.ek_continuous_cols:
            z_cols = []
            for c in self.ek_continuous_cols:
                z = (df[c].astype(float) - self.ek_cont_mean[c]) / self.ek_cont_std[c]
                z_cols.append(z)
            out["INTX_EK_continuous_mean_z"] = pd.concat(z_cols, axis=1).mean(axis=1, skipna=True)
        else:
            out["INTX_EK_continuous_mean_z"] = np.nan
        # rarity (AL_missing_ratio, zaten doğrulanmış bir nadirlik vekili) x korunmuşluk sinyali
        out["INTX_rarity_x_bounded"] = out["AL_missing_ratio"] * out["INTX_EK_bounded_mean"]
        out["INTX_rarity_x_continuous"] = out["AL_missing_ratio"] * out["INTX_EK_continuous_mean_z"]

        return out

    def transform_native(self, df: pd.DataFrame) -> pd.DataFrame:
        """XGBoost / LightGBM için: NaN olduğu gibi korunur (native missing-value desteği)."""
        assert self.fitted, "Önce fit() çağrılmalı"
        return self._base_features(df)

    def transform_imputed(self, df: pd.DataFrame) -> pd.DataFrame:
        """RandomForest / stacking meta-öğrenici için: NaN kabul etmeyen modeller için dolduruldu.
        AL_ -> rarity sentinel, EK_ -> train-fold medyanı, AA_grantham -> train-fold medyanı."""
        out = self._base_features(df).copy()
        for c in self.al_cols:
            out[c] = out[c].fillna(AL_SENTINEL)
        for c in self.ek_cols:
            out[c] = out[c].fillna(self.ek_medians.get(c, EK_SENTINEL))
        out["AA_grantham"] = out["AA_grantham"].fillna(out["AA_grantham"].median() if out["AA_grantham"].notna().any() else 0.0)
        for c in ["INTX_EK_bounded_mean", "INTX_EK_continuous_mean_z", "INTX_rarity_x_bounded", "INTX_rarity_x_continuous"]:
            out[c] = out[c].fillna(out[c].median() if out[c].notna().any() else 0.0)
        return out

    @property
    def feature_names(self):
        base = ["AL_missing_ratio", "AL_missing_count"] + self.al_cols
        base += ["EK_missing_ratio"] + self.ek_cols
        base += [c + "_code" for c in self.cat_cols] + ["CAT_missing_ratio"]
        base += ["AA_grantham", "AA_unknown_code", "AA_same", "TOTAL_missing_ratio"]
        base += ["INTX_EK_bounded_mean", "INTX_EK_continuous_mean_z", "INTX_rarity_x_bounded", "INTX_rarity_x_continuous"]
        return base


def row_group_ids(df: pd.DataFrame) -> np.ndarray:
    """Panel içinde tam öznitelik-profili aynı olan satırlara AYNI grup kimliği verir
    (organizatörün doğruladığı gibi bu tekrarlar kasıtlı -- silinmiyor, ama
    StratifiedKFold'un aynı satırı hem train hem val'e bölmesini önlemek için
    grup-farkında CV kullanılır -- hoca'nın olası varyant örtüşmesi uyarısına yanıt).
    Variant_ID ve Label hariç tüm kolonların hash'i alınır."""
    cols = [c for c in df.columns if c not in ("Variant_ID", "Label")]
    arr = df[cols].to_numpy(dtype=object)
    row_strs = ["|".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in row) for row in arr]
    codes, _ = pd.factorize(np.array(row_strs))
    return codes
