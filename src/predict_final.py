# -*- coding: utf-8 -*-
"""
GenVision - Final Tahmin / JSON Teslim Dosyası Üretici
==========================================================
16 Eylül 2026 final kılavuzundaki ZORUNLU JSON şemasına birebir uyan tahmin
dosyasını üretir. train_deploy.py'nin ürettiği {PANEL}_deploy.joblib
paketlerini yükler, verilen (etiketsiz) test CSV'lerini dönüştürüp tahmin
üretir ve TEK bir JSON dosyasında (tüm paneller birleşik) yazar.

Kullanım (gerçek final günü, panel başına AYRI dosya verilirse):
    cd src && python predict_final.py \
        --out ../TEAM_918091_FINAL.json \
        --master path/to/test_master.csv \
        --kanser path/to/test_kanser.csv \
        --pah path/to/test_pah.csv \
        --cftr path/to/test_cftr.csv

Kullanım (TEK bir dosyada, "Panel" sütunuyla karışık verilirse):
    cd src && python predict_final.py --out ../TEAM_918091_FINAL.json --input path/to/test.csv

Kullanım (dosyaları tek tek belirtmeden, ../data/final_test/ klasörüne koyup
otomatik buldurma -- dosya ADINDA panel adı geçmesi yeterli, ör. "MASTER" gibi):
    cd src && python predict_final.py --out ../TEAM_918091_FINAL.json
(hiç argüman verilmezse, --dry-run/--master/.../--input hiçbiri yoksa,
varsayılan olarak ../data/final_test/ taranır.)

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
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEPLOY_DIR = ROOT / "models" / "deploy"

# --------------------------------------------------------------------------
# Kalibrasyon kayması güvenlik ağı ("auto" sigorta)
# --------------------------------------------------------------------------
# threshold.py'de kalibre edilen sabit eşik PDR'deki VARSAYILAN test
# kompozisyonuna göre seçildi (bkz. expected_pos_rate, bundle'a train_deploy.py
# tarafından gömüldü). Gerçek final test setinin bileşimi bu varsayımdan
# saparsa (yeni popülasyon, farklı sevk zinciri vb.) sabit eşik sessizce kötü
# performans verir. Bu yüzden: sabit eşiğin ürettiği pozitif oran, beklenen
# oranın GUARD_HI/GUARD_LO katları dışına çıkarsa, ORAN-TABANLI (rank-based,
# top-k) bir eşiğe dönülür -- üretilen pozitif oranı beklenen orana zorlar.
#
# GUARD_HI/GUARD_LO = 1.8/0.55: guard_scan.py ile BU PROJENİN 4 paneli için
# ölçüldü (kansergen'deki (~/kansergen/predict2.py) 1.4/0.7 değeri
# KOPYALANMADI -- kendi OOF havuzumuzda 7 logit-kayması senaryosu taranıp en
# yüksek ortalama F1'i veren aday seçildi). Sonuçlar: models/guard_scan_results.json
# (4 panel ortalaması: sigortasız F1=0.469, sigortalı F1=0.571, en kötü
# senaryoda kazanç +0.263). Yeniden tarama: `python guard_scan.py`.
GUARD_HI, GUARD_LO = 1.8, 0.55

# team_id kullanıcı tarafından KYS'den teyit edildi (13.09.2026) -- sunumdaki
# "918081" ve kodun eski değeri "918191" YANLIŞ, doğrusu budur.
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


# --------------------------------------------------------------------------
# Sağlam CSV okuma -- final günü test dosyasının hangi yerelden/programdan
# geleceğini bilmiyoruz (~/kansergen/predict2.py'de aynı problem çözülüp
# 28 senaryolu final_senaryo_testleri.py ile doğrulanmıştı; aynı desen
# buraya taşındı).
# --------------------------------------------------------------------------
ID_COL = "Variant_ID"
ONDALIK_ESIK = 0.80  # bkz. _ondalik_virgul_duzelt


def _ayraci_bul(yol) -> str:
    """Başlık satırındaki aday ayraçları sayar; en çok geçeni seçer."""
    with open(yol, "r", encoding="utf-8-sig", errors="replace") as f:
        baslik = f.readline()
    adaylar = {",": baslik.count(","), ";": baslik.count(";"),
               "\t": baslik.count("\t"), "|": baslik.count("|")}
    en_iyi = max(adaylar, key=adaylar.get)
    return en_iyi if adaylar[en_iyi] > 0 else ","


def _tr_sayiya_cevir(seri: pd.Series) -> pd.Series:
    """TR yerelli sayı metnini sayıya çevirir. '1.234,56'->1234.56, '8,5e-09'->8.5e-09."""
    t = seri.astype(str).str.strip()
    virgullu = t.str.contains(",", na=False)
    t = t.where(~virgullu,
                t.str.replace(".", "", regex=False).str.replace(",", ".", regex=False))
    return pd.to_numeric(t, errors="coerce")


def _ondalik_virgul_duzelt(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """'0,0123' / '8,52e-09' biçimli metin kolonlarını sayısala çevirir -- düzeltilmezse
    bu kolonlar features.py'de sessizce NaN'a düşer (script çökmez, tahminler çöpe döner)."""
    duzeltilen = 0
    for kolon in df.columns:
        if kolon == ID_COL or df[kolon].dtype != object:
            continue
        ham = df[kolon]
        dolu = int(ham.notna().sum())
        if dolu == 0:
            continue
        dogrudan = int(pd.to_numeric(ham, errors="coerce").notna().sum())
        cevrilmis = _tr_sayiya_cevir(ham)
        kazanc = int(cevrilmis.notna().sum())
        if kazanc > dogrudan and kazanc >= ONDALIK_ESIK * dolu:
            df[kolon] = cevrilmis
            duzeltilen += 1
    return df, duzeltilen


def _id_kolonunu_bul(df: pd.DataFrame) -> str:
    """'Variant_ID' kolonunu ada bakmadan (boşluk/alt çizgi/büyük harf toleranslı) bulur."""
    def sadelestir(ad: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(ad).lower())

    hedef = sadelestir(ID_COL)
    for kolon in df.columns:
        if sadelestir(kolon) == hedef:
            return kolon
    raise ValueError(
        f"'{ID_COL}' kolonu bulunamadı. Dosyadaki ilk kolonlar: "
        f"{list(df.columns)[:5]}. Gönderim ID kolonu olmadan üretilemez; "
        "sıra numarası uydurmak yanlış eşleşmeye yol açar."
    )


def guvenli_oku(yol) -> pd.DataFrame:
    """Test CSV'sini yerelden bağımsız okur: ayraç sniff, TR ondalık virgül düzeltme,
    ID kolonunu ada bakmadan bulup 'Variant_ID'ye yeniden adlandırma."""
    ayrac = _ayraci_bul(yol)
    notlar = [] if ayrac == "," else [f"ayraç '{ayrac}' olarak algılandı"]
    df = pd.read_csv(yol, sep=ayrac, encoding="utf-8-sig")
    df, n_ond = _ondalik_virgul_duzelt(df)
    if n_ond:
        notlar.append(f"{n_ond} kolonda ondalık virgül düzeltildi")
    id_kolon = _id_kolonunu_bul(df)
    if id_kolon != ID_COL:
        notlar.append(f"ID kolonu '{id_kolon}' -> '{ID_COL}' olarak eşlendi")
        df = df.rename(columns={id_kolon: ID_COL})
    if df[ID_COL].isna().any():
        notlar.append(f"UYARI: {int(df[ID_COL].isna().sum())} satırda {ID_COL} boş")
    if df[ID_COL].duplicated().any():
        notlar.append(f"UYARI: {int(df[ID_COL].duplicated().sum())} tekrarlı {ID_COL} "
                       "(çıktıda aynen korunuyor)")
    if notlar:
        print(f"  [{Path(yol).name}] " + "; ".join(notlar))
    return df


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


def decide_classes(p, threshold, expected_pos_rate):
    """Sabit eşiği uygular; ürettiği pozitif oran beklenenin (expected_pos_rate)
    GUARD_HI/GUARD_LO katları dışına çıkarsa oran-tabanlı (rank-based top-k)
    eşiğe döner -- kalibrasyon kayması güvenlik ağı (bkz. dosya başındaki not).
    Döndürür: (0/1 sınıf dizisi, kullanılan mod açıklaması)."""
    y_fixed = (p >= threshold).astype(int)
    rate_fixed = float(y_fixed.mean())
    if rate_fixed > GUARD_HI * expected_pos_rate or rate_fixed < GUARD_LO * expected_pos_rate:
        k = max(1, int(round(expected_pos_rate * len(p))))
        cut = np.sort(p)[::-1][k - 1]
        y_final = (p >= cut).astype(int)
        mode = (f"ORAN-TABANLI (sigorta devrede): sabit eşik %{100*rate_fixed:.1f} pozitif "
                f"verdi, beklenen %{100*expected_pos_rate:.1f} idi -- oran-tabanlı eşiğe dönüldü")
        return y_final, mode
    return y_fixed, f"sabit eşik (%{100*rate_fixed:.1f} pozitif, beklenen %{100*expected_pos_rate:.1f})"


def predict_panel(panel, df):
    bundle = load_bundle(panel)
    id_col = "Variant_ID" if "Variant_ID" in df.columns else df.columns[0]
    p = predict_probabilities(bundle, df)
    threshold = bundle["threshold"]
    y_final, mode = decide_classes(p, threshold, bundle["expected_pos_rate"])
    print(f"{panel:8s}: {mode}")
    preds = []
    for vid, cls, prob in zip(df[id_col].astype(str), y_final, p):
        preds.append({
            "id": vid,
            "panel": panel,
            "predicted_class": "1" if cls else "0",
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


def find_panel_files_by_name(klasor):
    """`klasor` içindeki .csv dosyalarını, DOSYA ADINDA panel adı (MASTER/
    KANSER/PAH/CFTR, büyük/küçük harf duyarsız) geçmesine göre otomatik eşler
    -- 4 panel aynı sütun şemasını paylaştığı için İÇERİKTEN otomatik panel
    tespiti YAPILAMAZ (doğrulandı), bu yüzden eşleştirme dosya adına dayanır.
    Bulunamayan/belirsiz (birden fazla eşleşen) paneller için elle
    --master/--kanser/--pah/--cftr kullanılması gerektiği açıkça söylenir."""
    klasor = Path(klasor)
    if not klasor.is_dir():
        raise SystemExit(f"--testdir klasörü bulunamadı: {klasor}")
    csv_dosyalari = sorted(klasor.glob("*.csv"))
    if not csv_dosyalari:
        raise SystemExit(f"{klasor} içinde hiç .csv dosyası yok.")

    result = {}
    belirsiz = {}
    for panel in PANELS:
        eslesen = [f for f in csv_dosyalari if panel.lower() in f.stem.lower()]
        if len(eslesen) == 1:
            result[panel] = eslesen[0]
        elif len(eslesen) > 1:
            belirsiz[panel] = eslesen

    print(f"[--testdir] {klasor} içinde {len(csv_dosyalari)} CSV bulundu, adına göre eşleştirme:")
    for panel in PANELS:
        if panel in result:
            print(f"  {panel:8s} -> {result[panel].name}")
        elif panel in belirsiz:
            print(f"  {panel:8s} -> BELİRSİZ, birden fazla dosya eşleşti: "
                  f"{[f.name for f in belirsiz[panel]]}")
        else:
            print(f"  {panel:8s} -> eşleşme YOK")

    if belirsiz:
        raise SystemExit(
            f"Şu panel(ler) için birden fazla dosya adında eşleşme var: {list(belirsiz)}. "
            "Dosya adı otomatik eşleştirme için yeterince açık değil -- bunun yerine "
            "--master/--kanser/--pah/--cftr ile elle belirtin."
        )
    if not result:
        raise SystemExit(
            f"{klasor} içindeki hiçbir dosya adında MASTER/KANSER/PAH/CFTR geçmiyor "
            f"(dosyalar: {[f.name for f in csv_dosyalari]}). Otomatik eşleştirme "
            "yapılamıyor -- --master/--kanser/--pah/--cftr ile elle belirtin."
        )
    return {panel: guvenli_oku(path) for panel, path in result.items()}


def split_combined_input(csv_path):
    """Organizasyon 4 ayrı dosya yerine TEK bir dosyada, panel adını bir
    sütunda vererek gönderirse (kılavuzda format belirtilmediği için ihtimal
    dahilinde) bu dosyayı panel başına alt tabloya böler. Sütun adı büyük/
    küçük harfe duyarsız aranır ("Panel", "panel", "PANEL" hepsi kabul edilir);
    değerler de büyük/küçük harfe duyarsız MASTER/KANSER/PAH/CFTR'ye eşlenir."""
    df = guvenli_oku(csv_path)
    panel_col = next((c for c in df.columns if c.strip().lower() == "panel"), None)
    if panel_col is None:
        raise ValueError(
            f"--input dosyasında bir 'Panel' sütunu bulunamadı (sütunlar: "
            f"{list(df.columns)[:10]}...). Panel başına ayrı dosyaysa bunun "
            "yerine --master/--kanser/--pah/--cftr kullanın."
        )
    result = {}
    values_upper = df[panel_col].astype(str).str.strip().str.upper()
    for panel in PANELS:
        sub = df[values_upper == panel]
        if len(sub):
            result[panel] = sub.drop(columns=[panel_col])
    bulunmayan = set(values_upper.unique()) - set(PANELS)
    if bulunmayan:
        print(f"UYARI: '{panel_col}' sütununda tanınmayan değer(ler) var, atlanıyor: {bulunmayan}")
    return result


def build_submission(panel_df_map, out_path):
    all_preds = []
    for panel in PANELS:
        if panel not in panel_df_map:
            print(f"UYARI: {panel} için test dosyası verilmedi, atlanıyor.")
            continue
        preds = predict_panel(panel, panel_df_map[panel])
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
    panel_df_map = {p: pd.read_csv(DATA_DIR / TRAIN_FILES[p]) for p in PANELS}
    out_path = ROOT / "reports" / "DRY_RUN_submission.json"
    build_submission(panel_df_map, out_path)

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
    ap.add_argument("--input", help="Panel başına ayrı dosya yerine, hepsini içeren TEK bir "
                                     "CSV (bir 'Panel' sütunuyla panel adı belirtilmiş).")
    ap.add_argument("--testdir", nargs="?", const=str(DATA_DIR / "final_test"), default=None,
                     help="Dosyaları tek tek vermek yerine, bu klasördeki .csv dosyalarını "
                          "ADLARINA göre (içinde MASTER/KANSER/PAH/CFTR geçen) otomatik eşler. "
                          "Değer verilmezse ../data/final_test kullanılır.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    hicbir_sey_verilmedi = not (args.master or args.kanser or args.pah or args.cftr
                                 or args.input or args.testdir)
    if hicbir_sey_verilmedi:
        args.testdir = str(DATA_DIR / "final_test")
        print(f"Hiçbir dosya/klasör belirtilmedi -- varsayılan olarak {args.testdir} taranıyor.")

    if args.input:
        if args.master or args.kanser or args.pah or args.cftr or args.testdir:
            raise SystemExit("--input, --master/--kanser/--pah/--cftr/--testdir ile birlikte kullanılamaz.")
        panel_df_map = split_combined_input(args.input)
    elif args.testdir:
        if args.master or args.kanser or args.pah or args.cftr:
            raise SystemExit("--testdir, --master/--kanser/--pah/--cftr ile birlikte kullanılamaz.")
        panel_df_map = find_panel_files_by_name(args.testdir)
    else:
        panel_df_map = {}
        if args.master: panel_df_map["MASTER"] = guvenli_oku(args.master)
        if args.kanser: panel_df_map["KANSER"] = guvenli_oku(args.kanser)
        if args.pah: panel_df_map["PAH"] = guvenli_oku(args.pah)
        if args.cftr: panel_df_map["CFTR"] = guvenli_oku(args.cftr)
    if not panel_df_map:
        raise SystemExit("En az bir panel için test dosyası verin (--master/--kanser/--pah/--cftr, "
                          "--input ya da --testdir) ya da --dry-run kullanın.")
    build_submission(panel_df_map, args.out)


if __name__ == "__main__":
    main()
