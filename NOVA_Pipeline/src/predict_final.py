# -*- coding: utf-8 -*-
"""
GenVision - Final Tahmin / JSON Teslim Dosyası Üretici
==========================================================
16 Eylül 2026 final kılavuzundaki ZORUNLU JSON şemasına birebir uyan tahmin
dosyasını üretir. train_deploy.py'nin ürettiği {PANEL}_deploy.joblib
paketlerini yükler, verilen (etiketsiz) test CSV'lerini dönüştürüp tahmin
üretir ve TEK bir JSON dosyasında (tüm paneller birleşik) yazar.

Kullanım (gerçek final günü):
    cd src && python predict_final.py \
        --out ../TEAM_918091_FINAL.json \
        --master path/to/test_master.csv \
        --kanser path/to/test_kanser.csv \
        --pah path/to/test_pah.csv \
        --cftr path/to/test_cftr.csv

Kuru deneme (gerçek test verisi olmadan, kendi eğitim verimizle format+
uçtan-uca doğrulama için):
    cd src && python predict_final.py --dry-run

JSON şeması (final kılavuzu Bölüm 3.2, Üniversite ve Üzeri Seviyesi):
    {
      "team_name": ..., "team_id": ..., "application_id": ..., "competition_level": ...,
      "predictions": [{"id": ..., "panel": ..., "predicted_class": "0"/"1", "predicted_prob": ...}]
    }
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEPLOY_DIR = ROOT / "models" / "deploy"

# TODO: final günü KYS'deki bilgilerle BİREBİR teyit edin (PDR başlığından alındı).
TEAM_INFO = {
    "team_name": "NOVA",
    "team_id": "918091",
    "application_id": "4885814",
    "competition_level": "UNIVERSITE_VE_UZERI",
}

PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
TRAIN_FILES = {
    "MASTER": "YARISMA_TRAIN_MASTER.csv", "KANSER": "YARISMA_TRAIN_KANSER.csv",
    "PAH": "YARISMA_TRAIN_PAH.csv", "CFTR": "YARISMA_TRAIN_CFTR.csv",
}


def load_bundle(panel):
    path = DEPLOY_DIR / f"{panel}_deploy.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} yok -- önce 'python train_deploy.py' çalıştırın.")
    return joblib.load(path)


def predict_probabilities(bundle, df):
    """Panelin seçilen modeline göre (XGB/LGB/RF/CatBoost/Stacking/Blend_*/WBlend_*)
    olasılık üretir. Tek harfli kodlar: X=XGBoost, L=LightGBM, R=RandomForest, C=CatBoost."""
    feat = bundle["featurizer"]
    X_native = feat.transform_native(df)
    X_imp = feat.transform_imputed(df)

    p_xgb = bundle["xgb_model"].predict_proba(X_native)[:, 1]
    p_lgb = bundle["lgb_model"].predict_proba(X_native)[:, 1]
    p_rf = bundle["rf_model"].predict_proba(X_imp)[:, 1]
    p_cat = bundle["cat_model"].predict_proba(X_native)[:, 1]
    base = {"XGB": p_xgb, "LGB": p_lgb, "RF": p_rf, "CAT": p_cat}
    letter_map = {"X": "XGB", "L": "LGB", "R": "RF", "C": "CAT"}

    name = bundle["selected_model"]
    if name == "XGBoost":
        return p_xgb
    if name == "LightGBM":
        return p_lgb
    if name == "RandomForest":
        return p_rf
    if name == "CatBoost":
        return p_cat
    if name == "Stacking":
        meta_X = np.column_stack([p_xgb, p_lgb, p_rf, p_cat])
        return bundle["meta_model"].predict_proba(meta_X)[:, 1]
    if name == "Blend_All3":
        return (p_xgb + p_lgb + p_rf) / 3
    if name == "Blend_All4":
        return (p_xgb + p_lgb + p_rf + p_cat) / 4
    if name.startswith("Blend_"):
        # "Blend_XGB_LGB" gibi -- iki bileşenin sabit 50/50 ortalaması.
        a, b = name[len("Blend_"):].split("_")
        return (base[a] + base[b]) / 2
    if name.startswith("WBlend_"):
        # threshold.py'nin ürettiği dinamik ağırlıklı blend: "WBlend_{XL|XR|XC|LR|LC|RC}_{w:.1f}"
        # w, ilk bileşenin ağırlığı (örn. WBlend_XR_0.6 -> 0.6*p_xgb + 0.4*p_rf).
        _, pair, w_str = name.split("_")
        w = float(w_str)
        pa, pb = base[letter_map[pair[0]]], base[letter_map[pair[1]]]
        return w * pa + (1 - w) * pb
    raise ValueError(f"Bilinmeyen model adı: {name}")


def predict_panel(panel, csv_path):
    bundle = load_bundle(panel)
    df = pd.read_csv(csv_path)
    id_col = "Variant_ID" if "Variant_ID" in df.columns else df.columns[0]
    p = predict_probabilities(bundle, df)
    threshold = bundle["threshold"]
    preds = []
    for vid, prob in zip(df[id_col].astype(str), p):
        preds.append({
            "id": vid,
            "panel": panel,
            "predicted_class": "1" if prob >= threshold else "0",
            "predicted_prob": round(float(prob), 6),
        })
    return preds


def validate_predictions(preds):
    """JSON teslim kurallarına karşı hızlı bir yerel doğrulama (final kılavuzu Bölüm 3)."""
    ids_seen = set()
    for p in preds:
        assert p["predicted_class"] in ("0", "1"), f"Geçersiz sınıf: {p}"
        assert 0.0 <= p["predicted_prob"] <= 1.0, f"Olasılık [0,1] dışında: {p}"
        assert not (isinstance(p["predicted_prob"], float) and (np.isnan(p["predicted_prob"]) or np.isinf(p["predicted_prob"]))), f"NaN/Inf: {p}"
        key = (p["panel"], p["id"])
        assert key not in ids_seen, f"Tekrar eden id: {key}"
        ids_seen.add(key)
    print(f"Doğrulama OK: {len(preds)} tahmin, {len(ids_seen)} benzersiz (panel,id).")


def build_submission(panel_csv_map, out_path):
    all_preds = []
    for panel in PANELS:
        if panel not in panel_csv_map:
            print(f"UYARI: {panel} için test dosyası verilmedi, atlanıyor.")
            continue
        preds = predict_panel(panel, panel_csv_map[panel])
        print(f"{panel:8s}: {len(preds)} tahmin üretildi "
              f"(patojenik oranı: %{100*np.mean([p['predicted_class']=='1' for p in preds]):.1f})")
        all_preds.extend(preds)

    validate_predictions(all_preds)

    submission = dict(TEAM_INFO)
    submission["predictions"] = all_preds
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=None, separators=(",", ":"))
    print(f"\nKaydedildi: {out_path}  ({len(all_preds)} tahmin, {Path(out_path).stat().st_size/1024:.1f} KB)")


def dry_run():
    """Gerçek test verisi olmadan uçtan-uca format doğrulaması: kendi eğitim CSV'lerimizi
    (Label kolonu görmezden gelinerek) 'sahte test verisi' gibi kullanır ve ayrıca
    gerçek etiketlerle performansı da yazdırır (yalnızca kendi kendine sağlık kontrolü)."""
    from sklearn.metrics import f1_score, matthews_corrcoef
    print("=== KURU DENEME (gerçek final verisi değil, kendi eğitim verimiz) ===\n")
    panel_csv_map = {p: DATA_DIR / TRAIN_FILES[p] for p in PANELS}
    out_path = ROOT / "reports" / "DRY_RUN_submission.json"
    build_submission(panel_csv_map, out_path)

    with open(out_path, encoding="utf-8") as f:
        sub = json.load(f)
    by_panel = {}
    for p in sub["predictions"]:
        by_panel.setdefault(p["panel"], []).append(p)

    print("\n--- Kendi kendine sağlık kontrolü (gerçek etiketlerle, yalnızca kuru denemede) ---")
    for panel in PANELS:
        df = pd.read_csv(DATA_DIR / TRAIN_FILES[panel])
        y_true = df["Label"].astype(int).values
        id_col = "Variant_ID" if "Variant_ID" in df.columns else df.columns[0]
        id_to_true = dict(zip(df[id_col].astype(str), y_true))
        preds = by_panel[panel]
        y_pred = np.array([int(p["predicted_class"]) for p in preds])
        y_t = np.array([id_to_true[p["id"]] for p in preds])
        f1 = f1_score(y_t, y_pred)
        mcc = matthews_corrcoef(y_t, y_pred)
        print(f"{panel:8s} train-üzerinde F1={f1:.3f} MCC={mcc:.3f}  "
              f"(NOT: bu eğitim verisi üzerinde, iyimser -- gerçek performans göstergesi değil, "
              f"sadece JSON/format doğrulaması + boru hattı sağlık kontrolü)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "TEAM_918091_FINAL.json"))
    ap.add_argument("--master"); ap.add_argument("--kanser")
    ap.add_argument("--pah"); ap.add_argument("--cftr")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    panel_csv_map = {}
    if args.master: panel_csv_map["MASTER"] = args.master
    if args.kanser: panel_csv_map["KANSER"] = args.kanser
    if args.pah: panel_csv_map["PAH"] = args.pah
    if args.cftr: panel_csv_map["CFTR"] = args.cftr
    if not panel_csv_map:
        raise SystemExit("En az bir panel için test dosyası verin (--master/--kanser/--pah/--cftr) ya da --dry-run kullanın.")
    build_submission(panel_csv_map, args.out)


if __name__ == "__main__":
    main()
