# -*- coding: utf-8 -*-
"""
GenVision - Tek Komutla Final JSON Üretici
===========================================
src/predict_final.py'yi doğru dosya yollarıyla (veri/test/*.csv,
models/deploy/*.joblib) otomatik çağırıp TEAM_918091_FINAL.json'u kök
dizinde üretir. Kök dizinden çalıştırın:

    python uret.py

--dry-run bayrağı src/predict_final.py'ye aynen iletilir (gerçek test
verisi olmadan eğitim verisiyle uçtan uca format doğrulaması için).
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
TEST_DIR = ROOT / "veri" / "test"
DEPLOY_DIR = ROOT / "models" / "deploy"
OUT_PATH = ROOT / "TEAM_918091_FINAL.json"

PANEL_FLAGS = {
    "MASTER": "--master",
    "KANSER": "--kanser",
    "PAH": "--pah",
    "CFTR": "--cftr",
}

# predict_final.py'nin joblib.load ile açtığı her paket bir CatBoostClassifier
# içerir ve predict_probabilities bunu HER ZAMAN çağırır (selected_model ne
# olursa olsun) -- bu yüzden catboost eksikse hata en pahalı anda gelir.
GEREKLI_MODULLER = ["joblib", "numpy", "pandas", "sklearn", "xgboost", "lightgbm", "catboost"]


def _on_kontrol():
    eksik_dosya = []
    for panel in PANEL_FLAGS:
        csv_path = TEST_DIR / f"YARISMA_TEST_{panel}.csv"
        if not csv_path.exists():
            eksik_dosya.append(str(csv_path))
        joblib_path = DEPLOY_DIR / f"{panel}_deploy.joblib"
        if not joblib_path.exists():
            eksik_dosya.append(str(joblib_path))
    if eksik_dosya:
        print("HATA: Aşağıdaki dosyalar bulunamadı:")
        for yol in eksik_dosya:
            print(f"  - {yol}")
        sys.exit(1)

    eksik_modul = []
    for modul in GEREKLI_MODULLER:
        try:
            __import__(modul)
        except ImportError:
            eksik_modul.append(modul)
    if eksik_modul:
        print(f"HATA: Bu Python ortamında ({sys.executable}) şu paket(ler) kurulu değil: "
              f"{', '.join(eksik_modul)}")
        print("Çözüm: pip install -r requirements-inference.txt")
        print("(Final günü internet YOK -- bunu önceden, kendi bilgisayarınızda kurup test edin.)")
        sys.exit(1)


def main():
    _on_kontrol()
    cmd = [sys.executable, str(SRC / "predict_final.py"), "--out", str(OUT_PATH)]
    for panel, flag in PANEL_FLAGS.items():
        cmd += [flag, str(TEST_DIR / f"YARISMA_TEST_{panel}.csv")]
    if "--dry-run" in sys.argv[1:]:
        cmd = [sys.executable, str(SRC / "predict_final.py"), "--dry-run"]

    print("Çalıştırılıyor:", " ".join(cmd))
    sonuc = subprocess.run(cmd, cwd=str(SRC))
    sys.exit(sonuc.returncode)


if __name__ == "__main__":
    main()
