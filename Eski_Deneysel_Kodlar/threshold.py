"""
GenVision - Test-Bilinçli Eşik Ayarı + Bootstrap Doğrulama
=============================================================
Fikir: Eğitim OOF olasılıklarından TPR(t) ve TNR(t) fonksiyonlarını (eşiğe göre)
tahmin ediyoruz. Bu oranların, modelin sınıf-koşullu skor dağılımı aynı popülasyondan
geldiği sürece (yalnızca önsel/prevalence değiştiği sürece) test dağılımına da
taşınabileceğini varsayıyoruz -- yarışmanın kurgusu tam olarak bu (aynı panel,
sadece Pat/Ben oranı ters çevrilmiş).

Panel başına, TEST kümesinin BİLİNEN patojenik/benign sayısına göre projekte
edilmiş (F1+MCC)/2'yi maksimize eden eşik seçilir -- final şartnamesinin (16 Eylül
2026 kılavuzu) resmi skor formülüyle birebir aynı ("panel bazında F1 ve MCC
metriklerinin aritmetik ortalaması"). Bu, "naif" (OOF üzerinde F1-maksimize eden)
eşikten sistematik olarak farklıdır.

Doğrulama: OOF havuzundan (patojenik ve benign ayrı) test büyüklüğüne göre 2000 kez
bootstrap örneklem alınır, gerçek F1/MCC hesaplanır -> ortalama + %90 güven aralığı.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

PANELS = {
    "MASTER": {"test_pat": 500, "test_ben": 3000},
    "KANSER": {"test_pat": 100, "test_ben": 500},
    "PAH":    {"test_pat": 100, "test_ben": 250},
    "CFTR":   {"test_pat": 20,  "test_ben": 100},
}
MODEL_KEYS = {
    "XGBoost": "p_xgb", "LightGBM": "p_lgb", "RandomForest": "p_rf", "CatBoost": "p_cat",
    "Stacking": "p_stack",
    "Blend_XGB_LGB": "p_blend_xgb_lgb", "Blend_XGB_RF": "p_blend_xgb_rf",
    "Blend_XGB_CAT": "p_blend_xgb_cat", "Blend_LGB_RF": "p_blend_lgb_rf",
    "Blend_LGB_CAT": "p_blend_lgb_cat", "Blend_RF_CAT": "p_blend_rf_cat",
    "Blend_All3": "p_blend_all3", "Blend_All4": "p_blend_all4",
}

THRESH_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
N_BOOTSTRAP = 2000
CI = 0.90  # %90 güven aralığı (5. - 95. yüzdelik)


def naive_threshold(y, p):
    """OOF (eğitim dağılımı) üzerinde F1'i maksimize eden eşik -- yalnızca eşiği seçmek için."""
    best_t, best_f1 = 0.5, -1
    for t in THRESH_GRID:
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


def project_to_test(y, p, threshold, test_pat, test_ben):
    """Verilen (eğitimde seçilmiş) bir eşiğin, bilinen test Pat/Ben sayısına projekte
    edildiğinde ürettiği F1/MCC/precision/recall/specificity -- eşik seçim stratejilerini
    AYNI test-projeksiyonu üzerinden karşılaştırmak için (PDR Tablo 3 ile birebir aynı mantık)."""
    pos_scores = p[y == 1]
    neg_scores = p[y == 0]
    tpr = float((pos_scores >= threshold).mean())
    tnr = float((neg_scores < threshold).mean())
    tp, fn = tpr * test_pat, (1 - tpr) * test_pat
    tn, fp = tnr * test_ben, (1 - tnr) * test_ben
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    num = tp * tn - fp * fn
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = num / den if den > 0 else 0.0
    return {"f1": f1, "mcc": mcc, "precision": precision, "recall": recall, "specificity": tnr}


def test_aware_threshold(y, p, test_pat, test_ben):
    """TPR(t)/TNR(t) OOF'tan tahmin edilir, bilinen test Pat/Ben sayısına göre
    projekte edilmiş (F1+MCC)/2'yi maksimize eden eşik seçilir.
    ÖNEMLİ: final şartnamesi (16 Eylül 2026 kılavuzu) resmi skoru "panel bazında F1 ve
    MCC metriklerinin aritmetik ortalaması" olarak tanımlıyor -- bu yüzden yalnızca F1
    değil, F1 ile MCC'nin ortalaması maksimize edilir (önceki sürüm yalnızca F1
    maksimize ediyordu)."""
    pos_scores = p[y == 1]
    neg_scores = p[y == 0]
    best_t, best_combined, best_f1, best_stats = 0.5, -1e9, -1, None
    for t in THRESH_GRID:
        tpr = float((pos_scores >= t).mean())
        tnr = float((neg_scores < t).mean())
        tp = tpr * test_pat
        fn = (1 - tpr) * test_pat
        tn = tnr * test_ben
        fp = (1 - tnr) * test_ben
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        # MCC (projekte edilmiş kontenjans tablosundan)
        num = tp * tn - fp * fn
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = num / den if den > 0 else 0.0
        combined = (f1 + mcc) / 2
        if combined > best_combined:
            best_combined, best_t, best_f1 = combined, t, f1
            best_stats = {"precision": precision, "recall": recall, "specificity": tnr, "mcc": mcc}
    return float(best_t), float(best_f1), float(best_combined), best_stats


def bootstrap_validate(y, p, threshold, test_pat, test_ben, n_boot=N_BOOTSTRAP, seed=42):
    """OOF havuzundan test büyüklüğüne göre tekrarlı örneklem al, gerçek F1/MCC hesapla."""
    rng = np.random.default_rng(seed)
    pos_scores = p[y == 1]
    neg_scores = p[y == 0]
    f1s, mccs = [], []
    for _ in range(n_boot):
        samp_pos = rng.choice(pos_scores, size=test_pat, replace=True)
        samp_neg = rng.choice(neg_scores, size=test_ben, replace=True)
        y_samp = np.concatenate([np.ones(test_pat), np.zeros(test_ben)])
        p_samp = np.concatenate([samp_pos, samp_neg])
        pred = (p_samp >= threshold).astype(int)
        f1s.append(f1_score(y_samp, pred, zero_division=0))
        mccs.append(matthews_corrcoef(y_samp, pred) if len(set(pred)) > 1 else 0.0)
    f1s, mccs = np.array(f1s), np.array(mccs)
    lo, hi = (1 - CI) / 2 * 100, (1 + CI) / 2 * 100
    return {
        "f1_mean": float(f1s.mean()), "f1_ci": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "mcc_mean": float(mccs.mean()), "mcc_ci": [float(np.percentile(mccs, lo)), float(np.percentile(mccs, hi))],
    }


WEIGHTED_BLEND_WEIGHTS = [0.3, 0.4, 0.6, 0.7]  # 0.5 zaten Blend_* sabitlerinde var


def weighted_blend_candidates(data):
    """Sabit 50/50 blend'lere ek olarak, ikili kombinasyonlar için ağırlık taraması --
    ek model eğitimi gerektirmez, mevcut p_xgb/p_lgb/p_rf/p_cat'in doğrusal kombinasyonu."""
    cands = {}
    if not all(k in data.files for k in ("p_xgb", "p_lgb", "p_rf", "p_cat")):
        return cands
    pairs = {
        "XL": (data["p_xgb"], data["p_lgb"]), "XR": (data["p_xgb"], data["p_rf"]),
        "XC": (data["p_xgb"], data["p_cat"]), "LR": (data["p_lgb"], data["p_rf"]),
        "LC": (data["p_lgb"], data["p_cat"]), "RC": (data["p_rf"], data["p_cat"]),
    }
    for name, (pa, pb) in pairs.items():
        for w in WEIGHTED_BLEND_WEIGHTS:
            cands[f"WBlend_{name}_{w:.1f}"] = w * pa + (1 - w) * pb
    return cands


def main():
    summary = {}
    print(f"{'Panel':8s} {'Model':16s} {'Naif eşik':10s} {'Naif F1':9s} {'Test eşik':10s} {'Proj.F1':9s} {'Proj.MCC':9s} {'Proj.(F1+MCC)/2':16s} {'Boot F1 (90% CI)':22s}")
    for panel, cfg in PANELS.items():
        data = np.load(MODELS_DIR / f"{panel}_oof.npz")
        y = data["y"]
        best_model, best_combined, best_payload = None, -1e9, None
        wblends = weighted_blend_candidates(data)
        for model_name, key in list(MODEL_KEYS.items()) + [(k, k) for k in wblends]:
            if key in wblends:
                p = wblends[key]
            elif key not in data.files:
                continue  # eski .npz dosyalarında blend kolonları olmayabilir
            else:
                p = data[key]
            nt, _ = naive_threshold(y, p)
            naive_proj = project_to_test(y, p, nt, cfg["test_pat"], cfg["test_ben"])
            tt, tf1, tcombined, stats = test_aware_threshold(y, p, cfg["test_pat"], cfg["test_ben"])
            if tcombined > best_combined:
                best_combined = tcombined
                best_model = model_name
                best_payload = {
                    "naive_threshold": nt, "naive_f1_projected": naive_proj["f1"],
                    "naive_mcc_projected": naive_proj["mcc"],
                    "test_threshold": tt, "projected_test_f1": tf1, "projected_stats": stats,
                    "p": p,
                }
        boot = bootstrap_validate(y, best_payload["p"], best_payload["test_threshold"],
                                   cfg["test_pat"], cfg["test_ben"])
        summary[panel] = {
            "selected_model": best_model,
            "naive_threshold": best_payload["naive_threshold"],
            "naive_f1_projected": best_payload["naive_f1_projected"],
            "naive_mcc_projected": best_payload["naive_mcc_projected"],
            "test_threshold": best_payload["test_threshold"],
            "projected_test_f1": best_payload["projected_test_f1"],
            "projected_mcc": best_payload["projected_stats"]["mcc"],
            "projected_combined_score": (best_payload["projected_test_f1"] + best_payload["projected_stats"]["mcc"]) / 2,
            "projected_precision": best_payload["projected_stats"]["precision"],
            "projected_recall": best_payload["projected_stats"]["recall"],
            "projected_specificity": best_payload["projected_stats"]["specificity"],
            "bootstrap": boot,
        }
        ci_str = f"[{boot['f1_ci'][0]:.3f}, {boot['f1_ci'][1]:.3f}]"
        print(f"{panel:8s} {best_model:16s} {best_payload['naive_threshold']:<10.2f} "
              f"{best_payload['naive_f1_projected']:<9.3f} {best_payload['test_threshold']:<10.2f} "
              f"{best_payload['projected_test_f1']:<9.3f} {best_payload['projected_stats']['mcc']:<9.3f} "
              f"{summary[panel]['projected_combined_score']:<16.3f} {boot['f1_mean']:.3f} {ci_str}")

    final_score = float(np.mean([summary[p]["projected_combined_score"] for p in PANELS]))
    print(f"\n>>> Resmi final formülüne göre TAHMİNİ BİRLEŞİK SKOR (4 panel ortalaması): {final_score:.4f}")

    with open(MODELS_DIR / "threshold_results.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nKaydedildi:", MODELS_DIR / "threshold_results.json")


if __name__ == "__main__":
    main()
