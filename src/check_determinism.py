# -*- coding: utf-8 -*-
"""
GenVision - Determinizm Denetimi
====================================
Aynı test CSV'sini aynı deploy paketiyle İKİ KEZ tahmin ettirir ve olasılıkların
BİREBİR aynı çıktığını doğrular. Jüri kodu final günü yeniden çalıştırabilir
(şartname 7.5) -- iki koşum arasında en ufak bir sapma bile beyan edilen
sonucu tekrar üretilemez kılar.

Kullanım:
    cd src && python check_determinism.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

from predict_final import PANELS, predict_panel

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test"


def main():
    all_ok = True
    for panel in PANELS:
        csv_path = TEST_DIR / f"YARISMA_TEST_{panel}.csv"
        if not csv_path.exists():
            print(f"{panel:8s} ATLANDI -- {csv_path} yok")
            continue

        df = pd.read_csv(csv_path)
        preds1 = predict_panel(panel, df)
        preds2 = predict_panel(panel, df)

        p1 = np.array([p["predicted_prob"] for p in preds1])
        p2 = np.array([p["predicted_prob"] for p in preds2])
        c1 = [p["predicted_class"] for p in preds1]
        c2 = [p["predicted_class"] for p in preds2]

        max_diff = float(np.max(np.abs(p1 - p2))) if len(p1) else 0.0
        class_match = c1 == c2
        ok = (max_diff == 0.0) and class_match
        all_ok = all_ok and ok
        flag = "OK" if ok else "SORUN -- DETERMINIZM BOZUK"
        print(f"{panel:8s} maks olasılık farkı={max_diff:.2e}  sınıf eşleşti={class_match}  [{flag}]")

    print("\nSonuç:", "TÜM PANELLER DETERMİNİSTİK" if all_ok else "EN AZ BİR PANELDE SORUN VAR -- yukarı bakın")


if __name__ == "__main__":
    main()
