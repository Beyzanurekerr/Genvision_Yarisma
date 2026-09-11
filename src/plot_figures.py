# -*- coding: utf-8 -*-
"""
GenVision - Rapor/Sunum Görselleri
=====================================
threshold.py'nin eşik-tarama mantığını ve models/*.json sonuçlarını kullanarak
raporda/sunumda kanıt olarak kullanılacak 3 yüksek çözünürlüklü şekli üretir:

  Şekil 2 (fig2_esik_tarama.png)   -- panel başına, eğitim (ham OOF) F1 eğrisi ile
                                       test-projekte F1 eğrisinin eşiğe göre karşılaştırması.
  Şekil 3 (fig3_shap_kategori.png) -- panel başına SHAP kategori katkı yüzdeleri (yığılmış çubuk).
  Şekil 4 (fig4_karmasiklik.png)   -- panel başına, test prevalansına projekte edilmiş 2x2
                                       karmaşıklık matrisi (seçilen model + seçilen eşik).

Çalıştırma:
    cd src && python plot_figures.py
Çıktı:
    reports/figures/fig2_esik_tarama.png
    reports/figures/fig3_shap_kategori.png
    reports/figures/fig4_karmasiklik.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from threshold import (
    PANELS as THRESH_PANELS, MODEL_KEYS, THRESH_GRID,
    naive_threshold, project_to_test,
)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(MODELS_DIR / "threshold_results.json", encoding="utf-8") as f:
    THRESH_RESULTS = json.load(f)
with open(MODELS_DIR / "shap_summary.json", encoding="utf-8") as f:
    SHAP = json.load(f)

PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]
COLOR_TRAIN = "#8E9AAF"
COLOR_TEST = "#C1440E"
COLOR_NAIVE_LINE = "#5C6784"
COLOR_TEST_LINE = "#8C2E0B"

CAT_COLORS = {
    "AL (frekans)": "#2F5D8A",
    "EK (korunmuşluk)": "#C1440E",
    "AA (amino asit)": "#3E8E5B",
    "CAT (kategorik)": "#B8860B",
    "diğer (eksiklik özeti)": "#8E9AAF",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Şekil 2: Eşik Tarama Eğrisi
# ---------------------------------------------------------------------------

def train_f1_curve(y, p):
    """Ham OOF havuzu üzerinde (eğitim dağılımı altında) her eşik için F1."""
    from sklearn.metrics import f1_score
    return np.array([f1_score(y, (p >= t).astype(int)) for t in THRESH_GRID])


def test_f1_curve(y, p, test_pat, test_ben):
    """Bilinen test Pat/Ben sayısına projekte edilmiş her eşik için F1 (threshold.py ile aynı mantık)."""
    return np.array([project_to_test(y, p, t, test_pat, test_ben)["f1"] for t in THRESH_GRID])


def plot_threshold_scan():
    cfg = THRESH_PANELS
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    fig.suptitle("Şekil 2 — Eşik Tarama Eğrisi: Eğitim Dağılımı vs. Test-Projekte Performans",
                 fontsize=13, fontweight="bold", y=0.995)

    for ax, panel in zip(axes.flat, PANELS):
        data = np.load(MODELS_DIR / f"{panel}_oof.npz")
        y = data["y"]
        model_name = THRESH_RESULTS[panel]["selected_model"]
        p = data[MODEL_KEYS[model_name]]
        c = cfg[panel]

        f1_train = train_f1_curve(y, p)
        f1_test = test_f1_curve(y, p, c["test_pat"], c["test_ben"])

        ax.plot(THRESH_GRID, f1_train, color=COLOR_NAIVE_LINE, lw=2, label="Eğitim F1 (ham OOF)")
        ax.plot(THRESH_GRID, f1_test, color=COLOR_TEST_LINE, lw=2, label="Test-projekte F1")

        nt = THRESH_RESULTS[panel]["naive_threshold"]
        tt = THRESH_RESULTS[panel]["test_threshold"]
        ax.axvline(nt, color=COLOR_NAIVE_LINE, ls="--", lw=1.2, alpha=0.7)
        ax.axvline(tt, color=COLOR_TEST_LINE, ls="--", lw=1.2, alpha=0.7)
        ax.scatter([nt], [THRESH_RESULTS[panel]["naive_f1_projected"]], color=COLOR_NAIVE_LINE, zorder=5, s=35)
        ax.scatter([tt], [THRESH_RESULTS[panel]["projected_test_f1"]], color=COLOR_TEST_LINE, zorder=5, s=35)

        ax.set_title(f"{panel}  ({model_name})", fontweight="bold", fontsize=11)
        ax.set_xlabel("Karar eşiği (t)")
        ax.set_ylabel("F1")
        ax.set_xlim(0.03, 0.97)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.25)
        ax.annotate(f"naif t={nt:.2f}", (nt, 0.03), fontsize=7.5, color=COLOR_NAIVE_LINE,
                    ha="center", rotation=90, va="bottom")
        ax.annotate(f"test t={tt:.2f}", (tt, 0.03), fontsize=7.5, color=COLOR_TEST_LINE,
                    ha="center", rotation=90, va="bottom")

    handles = [plt.Line2D([0], [0], color=COLOR_NAIVE_LINE, lw=2, label="Eğitim F1 (ham OOF)"),
               plt.Line2D([0], [0], color=COLOR_TEST_LINE, lw=2, label="Test-projekte F1")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = FIG_DIR / "fig2_esik_tarama.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Kaydedildi:", out)


# ---------------------------------------------------------------------------
# Şekil 3: SHAP Kategori Katkısı (yığılmış çubuk)
# ---------------------------------------------------------------------------

def plot_shap_category():
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    cat_order = ["AL (frekans)", "EK (korunmuşluk)", "AA (amino asit)", "CAT (kategorik)", "diğer (eksiklik özeti)"]

    bottoms = np.zeros(len(PANELS))
    x = np.arange(len(PANELS))
    for cat in cat_order:
        vals = np.array([SHAP[p]["category_contribution_pct"].get(cat, 0.0) for p in PANELS])
        bars = ax.bar(x, vals, bottom=bottoms, color=CAT_COLORS[cat], label=cat, width=0.55,
                       edgecolor="white", linewidth=0.6)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 4:
                ax.text(xi, b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8.5,
                        color="white" if cat in ("AL (frekans)", "EK (korunmuşluk)") else "#222222", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}\n({SHAP[p]['model_used']})" for p in PANELS], fontsize=10)
    ax.set_ylabel("SHAP katkı yüzdesi (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Şekil 3 — Panel Bazlı SHAP Kategori Katkısı", fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "fig3_shap_kategori.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Kaydedildi:", out)


# ---------------------------------------------------------------------------
# Şekil 4: Test Prevalansına Projekte Karmaşıklık Matrisleri
# ---------------------------------------------------------------------------

def plot_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    fig.suptitle("Şekil 4 — Test Prevalansına Projekte Karmaşıklık Matrisleri",
                 fontsize=13, fontweight="bold", y=0.98)

    for ax, panel in zip(axes.flat, PANELS):
        t = THRESH_RESULTS[panel]
        cfg = THRESH_PANELS[panel]
        test_pat, test_ben = cfg["test_pat"], cfg["test_ben"]
        recall, spec = t["projected_recall"], t["projected_specificity"]
        tp, fn = recall * test_pat, (1 - recall) * test_pat
        tn, fp = spec * test_ben, (1 - spec) * test_ben

        mat = np.array([[tp, fn], [fp, tn]])  # satır: gerçek Pat/Ben, kolon: tahmin Pat/Ben
        mat_norm = mat.copy()
        mat_norm[0] /= test_pat
        mat_norm[1] /= test_ben

        im = ax.imshow(mat_norm, cmap="Blues", vmin=0, vmax=1)
        labels = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                txt_color = "white" if mat_norm[i, j] > 0.55 else "#222222"
                ax.text(j, i, f"{labels[i][j]}\n{mat[i, j]:.0f}\n(%{mat_norm[i, j]*100:.0f})",
                        ha="center", va="center", fontsize=9.5, color=txt_color, fontweight="bold")

        ax.set_xticks([0, 1]); ax.set_xticklabels(["Tahmin: Patojenik", "Tahmin: Benign"], fontsize=8.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Gerçek: Patojenik", "Gerçek: Benign"], fontsize=8.5)
        ax.set_title(f"{panel}  (model: {t['selected_model']}, eşik: {t['test_threshold']:.2f})\n"
                     f"n_test={test_pat + test_ben} (Pat={test_pat}, Ben={test_ben})  |  F1={t['projected_test_f1']:.3f}",
                     fontsize=9.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / "fig4_karmasiklik.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Kaydedildi:", out)


if __name__ == "__main__":
    plot_threshold_scan()
    plot_shap_category()
    plot_confusion_matrices()
