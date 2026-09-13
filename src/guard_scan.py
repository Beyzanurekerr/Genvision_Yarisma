# -*- coding: utf-8 -*-
"""
GenVision - Kalibrasyon Kayması Güvenlik Ağı: GUARD Taraması + Ölçümü
=========================================================================
Sunumda ("Girişim: GenVision Kalibre", madde 03) "popülasyon-özel yeniden
kalibrasyon ✓ÖLÇÜLDÜ" deniyor ama bugüne kadar predict_final.py'de gerçek test
dağılımı beklenenden saparsa devreye giren OTOMATİK bir düzeltme YOKTU -- sadece
tek seferlik, PDR'deki varsayılan kompozisyona göre kalibre edilmiş SABİT bir
eşik vardı (bkz. threshold.py / models/threshold_results.json).

Bu script, kardeş projede (~/kansergen, predict2.py + simulasyon_final.py)
zaten kanıtlanmış "auto sigorta" desenini bu projenin 4 paneline uyarlar:

  1. GUARD_HI/GUARD_LO taraması: sabit eşiğin ürettiği pozitif oran, beklenen
     (PDR) oranın kaç katına çıkarsa oran-tabanlı eşiğe dönülmeli? Kansergen'de
     taranan 3 aday (1.8/0.55, 1.4/0.7, 1.15/0.85) burada da taranır -- o
     projenin verisine özel sonucu KOPYALAMAK yerine, kendi 4 panelimizde
     hangisinin iyi çalıştığı ÖLÇÜLÜR.
  2. Seçilen GUARD değerleriyle "sabit eşik" (mevcut/eski davranış) ile
     "auto sigorta" (yeni davranış) stratejilerinin F1'i, OOF havuzundan
     olasılıklara logit-kayması uygulanarak simüle edilen 7 kayma senaryosunda
     (kalibrasyon stres testi) karşılaştırılır.

Veri kaynağı: SADECE yarışma eğitim verisinin OOF (out-of-fold) tahminleri
(models/{PANEL}_oof.npz) -- gerçek final test verisi elimizde yok (henüz
verilmedi), bu yüzden "gerçek test seti beklenenden farklı çıkarsa ne olur"
sorusunun cevabı ancak simülasyonla, kendi eğitim verimizin OOF'undan
üretilebilir. Model bu satırları eğitimde hiç görmemiştir (OOF), dolayısıyla
bu ölçüm iyimser (train satırlarını yeniden kullanan bir "kuru deneme"nin
aksine) dürüst bir final-günü tahminidir.

Çıktı: models/guard_scan_results.json (seçilen GUARD_HI/LO + ölçülen F1
kazancı) -- predict_final.py'ye gömülecek sabitler buradan gelir.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# threshold.py / train_final.py ile AYNI (PDR'deki varsayılan test kompozisyonu).
PANELS = {
    "MASTER": {"test_pat": 500, "test_ben": 3000},
    "KANSER": {"test_pat": 100, "test_ben": 500},
    "PAH":    {"test_pat": 100, "test_ben": 250},
    "CFTR":   {"test_pat": 20,  "test_ben": 100},
}

# Kansergen'in kendi verisinde taradığı 3 aday (predict2.py'deki gerekçeyle
# aynı mantık) -- burada KOPYALANMIYOR, sadece aday havuzu olarak kullanılıp
# bu projenin panelleri için yeniden ölçülüyor.
GUARD_CANDIDATES = [(1.8, 0.55), (1.4, 0.7), (1.15, 0.85)]

# Kalibrasyon stres testi: OOF olasılıklarına eklenen logit kayması (kansergen
# ile aynı tarama aralığı -- karşılaştırılabilir olması için).
LOGIT_SHIFTS = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

N_SIM = 500
SEED = 42


def load_selected_probability(panel):
    """threshold_results.json'daki SEÇİLEN modelin OOF olasılığını, base
    (p_xgb/p_lgb/p_rf/p_cat) bileşenlerinden -- predict_final.py'deki
    predict_probabilities() ile AYNI adlandırma sözleşmesiyle -- yeniden
    üretir (WBlend_* ağırlıklı blend'ler .npz'de doğrudan saklanmıyor)."""
    with open(MODELS_DIR / "threshold_results.json", encoding="utf-8") as f:
        thresh = json.load(f)[panel]
    data = np.load(MODELS_DIR / f"{panel}_oof.npz")
    y = data["y"]
    name = thresh["selected_model"]
    threshold = thresh["test_threshold"]

    base = {"XGB": data["p_xgb"], "LGB": data["p_lgb"], "RF": data["p_rf"], "CAT": data["p_cat"]}
    letter_map = {"X": "XGB", "L": "LGB", "R": "RF", "C": "CAT"}

    if name == "XGBoost":
        p = data["p_xgb"]
    elif name == "LightGBM":
        p = data["p_lgb"]
    elif name == "RandomForest":
        p = data["p_rf"]
    elif name == "CatBoost":
        p = data["p_cat"]
    elif name == "Stacking":
        p = data["p_stack"]
    elif name == "Blend_All3":
        p = data["p_blend_all3"]
    elif name == "Blend_All4":
        p = data["p_blend_all4"]
    elif name.startswith("Blend_"):
        key = f"p_blend_{name[len('Blend_'):].lower()}"
        p = data[key]
    elif name.startswith("WBlend_"):
        _, pair, w_str = name.split("_")
        w = float(w_str)
        pa, pb = base[letter_map[pair[0]]], base[letter_map[pair[1]]]
        p = w * pa + (1 - w) * pb
    else:
        raise ValueError(f"Bilinmeyen model adı: {name}")

    return y, p, threshold, name


def apply_logit_shift(p, shift):
    if not shift:
        return p
    c = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(c / (1 - c))
    return 1 / (1 + np.exp(-(logit + shift)))


def simulate(y, p, threshold, test_pat, test_ben, strategy, guard_hi, guard_lo,
             n_sim=N_SIM, seed=SEED):
    """PDR test kompozisyonuna göre tekrarlı bootstrap örneklem alır; 'fixed'
    (sabit eşik, mevcut/eski davranış) veya 'auto' (sigorta: oran beklenenin
    dışına çıkarsa oran-tabanlı eşiğe döner) stratejisiyle F1 ölçer."""
    rng = np.random.default_rng(seed)
    ip, ib = np.where(y == 1)[0], np.where(y == 0)[0]
    exp_pos = test_pat / (test_pat + test_ben)

    f1s, sigorta_tetik = [], 0
    for _ in range(n_sim):
        idx = np.concatenate([rng.choice(ip, test_pat, replace=True),
                               rng.choice(ib, test_ben, replace=True)])
        yv, pv = y[idx], p[idx]
        yh = (pv >= threshold).astype(int)
        rate_fixed = float(yh.mean())
        if strategy == "auto" and (rate_fixed > guard_hi * exp_pos or rate_fixed < guard_lo * exp_pos):
            k = max(1, int(round(exp_pos * len(pv))))
            cut = np.sort(pv)[::-1][k - 1]
            yh = (pv >= cut).astype(int)
            sigorta_tetik += 1
        f1s.append(f1_score(yv, yh, pos_label=1, zero_division=0))
    return np.array(f1s), sigorta_tetik / n_sim


def scan_guard_candidates():
    """Her aday (GUARD_HI, GUARD_LO) çifti için 4 panel x 7 kayma senaryosunda
    ortalama F1'i ölçer; kaymasız/en-kötü/ortalama raporlar (kansergen'deki
    seçim mantığıyla aynı üç ölçüt)."""
    per_panel_data = {p: load_selected_probability(p) for p in PANELS}

    print("GUARD taraması (4 panel ortalaması, 7 kayma senaryosu üzerinden):")
    print(f"{'GUARD_HI/LO':14s}{'kaymasız':>10s}{'en kötü':>10s}{'ortalama':>10s}")
    results = {}
    for guard_hi, guard_lo in GUARD_CANDIDATES:
        per_shift_scores = []
        for shift in LOGIT_SHIFTS:
            panel_scores = []
            for panel, cfg in PANELS.items():
                y, p, threshold, _ = per_panel_data[panel]
                p_shifted = apply_logit_shift(p, shift)
                f1s, _ = simulate(y, p_shifted, threshold, cfg["test_pat"], cfg["test_ben"],
                                   "auto", guard_hi, guard_lo)
                panel_scores.append(f1s.mean())
            per_shift_scores.append(float(np.mean(panel_scores)))
        kaymasiz = per_shift_scores[LOGIT_SHIFTS.index(0.0)]
        en_kotu = min(per_shift_scores)
        ortalama = float(np.mean(per_shift_scores))
        results[f"{guard_hi}/{guard_lo}"] = {
            "guard_hi": guard_hi, "guard_lo": guard_lo,
            "kaymasiz_f1": kaymasiz, "en_kotu_f1": en_kotu, "ortalama_f1": ortalama,
            "per_shift_f1": dict(zip(map(str, LOGIT_SHIFTS), per_shift_scores)),
        }
        print(f"{guard_hi}/{guard_lo:<9}{kaymasiz:>10.3f}{en_kotu:>10.3f}{ortalama:>10.3f}")

    best_key = max(results, key=lambda k: results[k]["ortalama_f1"])
    print(f"\nSeçilen: GUARD_HI/LO = {best_key} (en yüksek ortalama F1)")
    return results, results[best_key]["guard_hi"], results[best_key]["guard_lo"], per_panel_data


def measure_fixed_vs_auto(per_panel_data, guard_hi, guard_lo):
    """Seçilen GUARD değerleriyle, panel bazında 'fixed' (eski/sigortasız) ile
    'auto' (yeni/sigortalı) stratejilerini 7 kayma senaryosunda karşılaştırır
    -- sunumdaki iddiayı destekleyecek asıl ölçüm bu tablodur."""
    print("\nÖlçüm: sigortasız (fixed) vs sigortalı (auto) -- panel bazında, kayma taraması:")
    print(f"{'panel':8s}{'kaymasız F1':>13s}{'kaymasız F1':>13s}   |  {'en kötü F1':>11s}{'en kötü F1':>11s}   |  {'ort. F1':>9s}{'ort. F1':>9s}")
    print(f"{'':8s}{'(fixed)':>13s}{'(auto)':>13s}   |  {'(fixed)':>11s}{'(auto)':>11s}   |  {'(fixed)':>9s}{'(auto)':>9s}")
    panel_results = {}
    for panel, cfg in PANELS.items():
        y, p, threshold, model_name = per_panel_data[panel]
        fixed_scores, auto_scores = [], []
        for shift in LOGIT_SHIFTS:
            p_shifted = apply_logit_shift(p, shift)
            f1_fixed, _ = simulate(y, p_shifted, threshold, cfg["test_pat"], cfg["test_ben"],
                                    "fixed", guard_hi, guard_lo)
            f1_auto, sig_rate = simulate(y, p_shifted, threshold, cfg["test_pat"], cfg["test_ben"],
                                          "auto", guard_hi, guard_lo)
            fixed_scores.append(f1_fixed.mean())
            auto_scores.append(f1_auto.mean())
        kaymasiz_ix = LOGIT_SHIFTS.index(0.0)
        panel_results[panel] = {
            "selected_model": model_name,
            "kaymasiz_f1_fixed": fixed_scores[kaymasiz_ix], "kaymasiz_f1_auto": auto_scores[kaymasiz_ix],
            "en_kotu_f1_fixed": min(fixed_scores), "en_kotu_f1_auto": min(auto_scores),
            "ortalama_f1_fixed": float(np.mean(fixed_scores)), "ortalama_f1_auto": float(np.mean(auto_scores)),
        }
        r = panel_results[panel]
        print(f"{panel:8s}{r['kaymasiz_f1_fixed']:>13.3f}{r['kaymasiz_f1_auto']:>13.3f}   |  "
              f"{r['en_kotu_f1_fixed']:>11.3f}{r['en_kotu_f1_auto']:>11.3f}   |  "
              f"{r['ortalama_f1_fixed']:>9.3f}{r['ortalama_f1_auto']:>9.3f}")

    avg_fixed = float(np.mean([r["ortalama_f1_fixed"] for r in panel_results.values()]))
    avg_auto = float(np.mean([r["ortalama_f1_auto"] for r in panel_results.values()]))
    worst_fixed = float(np.mean([r["en_kotu_f1_fixed"] for r in panel_results.values()]))
    worst_auto = float(np.mean([r["en_kotu_f1_auto"] for r in panel_results.values()]))
    print(f"\n4 panel ortalaması -- ortalama F1: fixed={avg_fixed:.3f} auto={avg_auto:.3f} "
          f"(kazanç {avg_auto - avg_fixed:+.3f})")
    print(f"4 panel ortalaması -- en kötü senaryo F1: fixed={worst_fixed:.3f} auto={worst_auto:.3f} "
          f"(kazanç {worst_auto - worst_fixed:+.3f})")
    return panel_results, avg_fixed, avg_auto, worst_fixed, worst_auto


def main():
    scan_results, guard_hi, guard_lo, per_panel_data = scan_guard_candidates()
    panel_results, avg_fixed, avg_auto, worst_fixed, worst_auto = measure_fixed_vs_auto(
        per_panel_data, guard_hi, guard_lo)

    out = {
        "guard_hi": guard_hi, "guard_lo": guard_lo,
        "candidates_scanned": scan_results,
        "fixed_vs_auto_per_panel": panel_results,
        "fixed_vs_auto_summary": {
            "ortalama_f1_fixed": avg_fixed, "ortalama_f1_auto": avg_auto,
            "ortalama_f1_kazanc": avg_auto - avg_fixed,
            "en_kotu_f1_fixed": worst_fixed, "en_kotu_f1_auto": worst_auto,
            "en_kotu_f1_kazanc": worst_auto - worst_fixed,
        },
        "n_sim": N_SIM, "logit_shifts": LOGIT_SHIFTS,
        "not": ("Simülasyon SADECE yarışma eğitim verisinin OOF olasılıklarından "
                "üretildi (gerçek final test verisi henüz elimizde yok); logit "
                "kayması ile kalibrasyon/dağılım kayması senaryoları taklit edildi."),
    }
    out_path = MODELS_DIR / "guard_scan_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nKaydedildi: {out_path}")


if __name__ == "__main__":
    main()
