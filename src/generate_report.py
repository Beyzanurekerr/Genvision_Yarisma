# -*- coding: utf-8 -*-
"""
GenVision - Proje Değerlendirme Raporu Üretici
=================================================
models/*.json ve reports/*.csv dosyalarındaki DOĞRULANMIŞ sonuçları okuyup
tek bir .docx raporu üretir. Anlatı metni burada elle yazıldı (şablon değil);
sayısal tablolar ise kaynak dosyalardan dinamik okunuyor -- rapor
güncellendiğinde (örn. yeni bir ablasyon eklenince) tekrar çalıştırılabilir.

Çalıştırma:
    cd src && python generate_report.py
Çıktı:
    reports/GenVision_Rapor.docx
"""
import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.shared import Pt, Cm, RGBColor
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
HEADER_FILL = "1F3A5F"

# ---------------------------------------------------------------------------
# Veri yükleme
# ---------------------------------------------------------------------------

def load_json(name):
    with open(MODELS_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def load_csv(name):
    with open(REPORTS_DIR / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


THRESH = load_json("threshold_results.json")
LEAK = load_json("leakage_check.json")
SMOTE = load_json("smote_experiment.json")
SHAP = load_json("shap_summary.json")
FINALCV = load_json("final_cv_results.json")
HPARAMS = load_json("best_hparams.json")
SEED_STAB = load_json("seed_stability.json")

CLASS_DIST = load_csv("class_distribution.csv")
AL_SUMMARY = load_csv("al_summary.csv")
EK_SUMMARY = load_csv("ek_summary.csv")
CAT_SUMMARY = load_csv("cat_summary.csv")

PANELS = ["MASTER", "KANSER", "PAH", "CFTR"]

# ---------------------------------------------------------------------------
# docx yardımcıları
# ---------------------------------------------------------------------------

def set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def style_cell(cell, text, bold=False, size=9, color=None, align_center=False):
    cell.text = ""
    p = cell.paragraphs[0]
    if align_center:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(str(text))
    run.bold = bold
    run.font.size = Pt(size)
    if color:
        run.font.color.rgb = color


def add_table(doc, headers, rows, widths=None, caption=None, caption_num=None):
    if caption:
        cap = doc.add_paragraph()
        run = cap.add_run(f"Tablo {caption_num}. {caption}")
        run.bold = True
        run.font.size = Pt(9.5)
        cap.paragraph_format.space_before = Pt(10)
        cap.paragraph_format.space_after = Pt(4)

    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        style_cell(hdr_cells[i], h, bold=True, size=9, color=RGBColor(0xFF, 0xFF, 0xFF), align_center=True)
        set_cell_shading(hdr_cells[i], HEADER_FILL)
    for row in rows:
        cells = table.add_row().cells
        for i, val in enumerate(row):
            style_cell(cells[i], val, align_center=(i > 0))
    if widths:
        for row in table.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return table


def h1(doc, text):
    p = doc.add_heading(text, level=1)
    for run in p.runs:
        run.font.color.rgb = ACCENT
    return p


def h2(doc, text):
    p = doc.add_heading(text, level=2)
    for run in p.runs:
        run.font.color.rgb = ACCENT
    return p


def para(doc, text, bold=False, italic=False, size=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    p.paragraph_format.space_after = Pt(8)
    return p


def bullet(doc, text):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    return p


def add_figure(doc, filename, caption, caption_num, width_cm=16):
    doc.add_picture(str(FIG_DIR / filename), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = cap.add_run(f"Şekil {caption_num}. {caption}")
    run.bold = True
    run.font.size = Pt(9.5)
    cap.paragraph_format.space_after = Pt(10)


# ---------------------------------------------------------------------------
# Belge iskeleti
# ---------------------------------------------------------------------------

doc = Document()

# Genel stil
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.15

for level, size in [(1, 15), (2, 12.5), (3, 11)]:
    st = doc.styles[f"Heading {level}"]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.color.rgb = ACCENT
    st.font.bold = True

sections = doc.sections
for s in sections:
    s.top_margin = Cm(2.2)
    s.bottom_margin = Cm(2.2)
    s.left_margin = Cm(2.3)
    s.right_margin = Cm(2.3)

# ---------------------------------------------------------------------------
# Kapak
# ---------------------------------------------------------------------------

title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
title.paragraph_format.space_before = Pt(60)
run = title.add_run("GenVision")
run.bold = True
run.font.size = Pt(30)
run.font.color.rgb = ACCENT

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
run = sub.add_run("Klinik Genomik Veride Missense Varyant Patojenite Tahmini")
run.font.size = Pt(15)
run.italic = True

sub2 = doc.add_paragraph()
sub2.alignment = WD_ALIGN_PARAGRAPH.CENTER
sub2.paragraph_format.space_before = Pt(6)
run = sub2.add_run("TEKNOFEST Sağlıkta Yapay Zeka Yarışması — Proje Değerlendirme Raporu")
run.font.size = Pt(12)

doc.add_paragraph().paragraph_format.space_after = Pt(40)

team_p = doc.add_paragraph()
team_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
team_p.paragraph_format.space_before = Pt(30)
for line in [
    "Beyzanur EKER — Takım Kaptanı",
    "Melike YAŞAR — Veri Analizi",
    "Berat SOYKUVVET — Makine Öğrenmesi",
    "Erdoğan Yasin PEKER — Kalibrasyon",
    "İsmail AKBOĞA — MLOps",
]:
    r = team_p.add_run(line + "\n")
    r.font.size = Pt(11)

date_p = doc.add_paragraph()
date_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
date_p.paragraph_format.space_before = Pt(30)
r = date_p.add_run("7 Eylül 2026")
r.font.size = Pt(11)
r.italic = True

doc.add_page_break()

# ---------------------------------------------------------------------------
# 1. Giriş ve Problem Tanımı
# ---------------------------------------------------------------------------

h1(doc, "1. Giriş ve Problem Tanımı")

para(doc,
     "Missense varyantlar, bir genin kodlayan bölgesinde tek bir nükleotidin değişerek "
     "protein dizisinde farklı bir amino asidin yerleşmesine yol açan nokta mutasyonlarıdır. "
     "Bu değişimlerin büyük çoğunluğu proteinin işlevini bozmayan zararsız (benign) "
     "polimorfizmlerdir; ancak bir kısmı proteinin yapısını veya işlevini bozarak hastalığa "
     "yol açar (patojenik). Klinik genomikte hangi varyantın hangi sınıfa girdiğini ayırt "
     "etmek -- özellikle yeni nesil dizileme (NGS) ile üretilen binlerce varyantlık listeler "
     "içinden klinik olarak anlamlı olanı önceliklendirmek -- laboratuvar iş akışının en "
     "emek-yoğun adımlarından biridir. GenVision projesi bu sınıflandırma problemini "
     "makine öğrenmesiyle otomatikleştirmeyi hedeflemektedir.")

para(doc,
     "Yarışma verisi bu problemi TEK bir havuz olarak değil, dört ayrı “panel” halinde "
     "sunmaktadır: MASTER (genel varyant havuzu), KANSER (kanserle ilişkili gen paneli), "
     "PAH (fenilketonüri gen paneli) ve CFTR (tek gen -- kistik fibrozis "
     "transmembran iletkenlik düzenleyicisi). Panellerin öznitelik dağılımları, patojenite "
     "önselleri ve örneklem büyüklükleri birbirinden belirgin şekilde farklıdır (Bölüm 2.1); "
     "bu nedenle GenVision panel başına bağımsız bir model eğitmeyi ve panele özgü karar "
     "eşiği belirlemeyi metodolojik bir gereklilik olarak ele almıştır -- tek bir genel model "
     "her panelin kendine özgü sınıf dengesini ve öznitelik davranışını temsil edemez.")

h2(doc, "1.1 Yarışmanın Kasıtlı “Klinik Stres Testi” Tasarımı")
para(doc,
     "Veri sözlüğü tablolarından doğrulanan en kritik bulgu şudur: eğitim setinde patojenik "
     "sınıf baskındır (panellere göre %69-83 arası), ancak yarışmanın asıl değerlendirme "
     "yapılacağı test seti yaklaşık TERS bir dağılıma sahiptir -- benign sınıf baskındır "
     "(panellere göre patojenik oranı %14-29 arasına düşer). Bu, organizatörün veri "
     "sözlüğünde açıkça verdiği, kasıtlı bir tasarım kararıdır: gerçek klinik ortamda "
     "test edilen varyantların büyük çoğunluğu zaten benign çıkar (nadir hastalık taraması "
     "yapılırken bile), bu yüzden bir modelin sadece eğitim setindeki sınıf oranını "
     "ezberleyip ezberlemediğini -- yani genelleme gücünü -- ölçmek için bilinçli olarak "
     "ters çevrilmiş bir test dağılımı kullanılmaktadır. Bu tek gerçek, projedeki neredeyse "
     "her metodolojik kararı (eşik kalibrasyonu, metrik seçimi, SMOTE'a temkinli yaklaşım) "
     "doğrudan şekillendirmiştir ve Bölüm 3.6'da detaylandırılmaktadır.")

h2(doc, "1.2 Veri Anonimliği ve Kapsam Kısıtı")
para(doc,
     "Kolon isimleri gerçek biyolojik karşılıklarıyla değil anonimleştirilmiş öneklerle "
     "(AL_, EK_, CAT_, AA_) verilmektedir; genomik adres bilgisi (kromozom, pozisyon, "
     "rsID) de veri setinde bulunmamaktadır. Bu iki kısıt birlikte, ClinVar, gnomAD veya "
     "REVEL gibi dış veritabanlarıyla SATIR BAZLI (varyant-varyant) eşleştirme yapılmasını "
     "imkânsız kılmaktadır -- böyle bir eşleştirme en az bir benzersiz genomik koordinat ya "
     "da tanımlayıcı gerektirir. Bu nedenle GenVision, modelleme sürecinin tamamında "
     "sağlanan veriyle kendi kendine yeterli (self-contained) kalmıştır; dış veri "
     "genişletme olasılığı Bölüm 6'da ayrı bir gelecek çalışma maddesi olarak ele "
     "alınmaktadır.")

h2(doc, "1.3 Değerlendirme Metriği")
para(doc,
     "Yarışmanın final metriği, dört panelin PATOJENİK-odaklı F1 skorunun ortalamasıdır ve "
     "modelden olasılık değil ikili (0/1) çıktı istenmektedir. Bu, bu raporun tamamında "
     "F1'in neden merkezi metrik olarak seçildiğini ve neden yalnızca model seçiminin değil "
     "karar EŞİĞİNİN de ayrı bir optimizasyon hedefi haline geldiğini açıklamaktadır "
     "(Bölüm 3.6). Rapor aşamasında serbest bırakılan ROC-AUC, precision, recall ve MCC "
     "metrikleri ise modelin genel ayırt etme gücünü ve dengesini göstermek için ek olarak "
     "sunulmaktadır.")

# ---------------------------------------------------------------------------
# 2. Veri Seti ve Keşifsel Analiz
# ---------------------------------------------------------------------------

h1(doc, "2. Veri Seti ve Keşifsel Analiz")

h2(doc, "2.1 Panel Yapısı ve Sınıf Dağılımı")
para(doc,
     "Tablo 1, her panelin eğitim ve (organizatör tarafından bildirilen) test kümesi "
     "büyüklüklerini ve sınıf dağılımlarını göstermektedir. Ters çevrilmiş oran her "
     "panelde açıkça görülmektedir; CFTR, 111 satırlık eğitim kümesiyle -- projede en "
     "yüksek varyans riski taşıyan panel olarak Bölüm 5'te ayrıca ele alınmaktadır.")

rows = []
for r in CLASS_DIST:
    rows.append([
        r["Panel"], r["Train_n"], f"{r['Train_Pat']} / {r['Train_Ben']}",
        f"%{float(r['Train_Pat_Oran'])*100:.1f}", r["Test_n"],
        f"{r['Test_Pat']} / {r['Test_Ben']}", f"%{float(r['Test_Pat_Oran'])*100:.1f}",
        r["N_Ozellik"],
    ])
add_table(doc,
          ["Panel", "Train n", "Train Pat/Ben", "Train Pat %", "Test n", "Test Pat/Ben", "Test Pat %", "Öznitelik"],
          rows,
          caption="Panel bazlı eğitim/test büyüklükleri ve sınıf dağılımı (test sayıları organizatör tarafından bildirilmiştir).",
          caption_num=1)

h2(doc, "2.2 Öznitelik Grupları")
para(doc,
     "Öznitelikler dört önekle gruplanmaktadır. AL_ (334 kolon) -- CAT_1/CAT_2 kategorik "
     "kolonlarındaki örnek değerlerin (gnomADe_*, AllofUs_* gibi popülasyon/kohort etiketleri) "
     "işaret ettiği üzere, muhtemelen popülasyon bazlı allel frekansı benzeri bir yapıya "
     "sahiptir; ancak kesin biyolojik karşılığı organizatör tarafından teyit edilmemiştir, bu "
     "yüzden raporun geri kalanında bu kolonlara nötr olarak “frekans” öbeği "
     "denmektedir. EK_ (9 kolon) -- ekip içi adlandırmayla “korunmuşluk” öbeği, "
     "değer aralıkları ve 0-1 sınırlı olanların varlığı evrimsel korunmuşluk skorlarıyla "
     "(örn. GERP/phyloP/SIFT benzeri) tutarlıdır. CAT_ (6 kolon, biri yedekli) -- kategorik "
     "meta-veri; CAT_1/CAT_2 popülasyon etiketleri, CAT_3/4/5 genotip-benzeri üçlü kodlar "
     "(“C/C, T/T, ./.”), CAT_6 tekrar-bölgesi bayrakları (“lcr, segdup”) "
     "gibi görünmektedir. AA_ (2 kolon) -- değişim öncesi/sonrası amino asit kimliği, "
     "Grantham fizikokimyasal mesafesi hesaplamak için kullanılmaktadır (Bölüm 3.1).")

h2(doc, "2.3 Eksik Veri Analizi")
rows = []
for r in AL_SUMMARY:
    rows.append([r["Panel"], r["N_AL_Kolon"], f"%{float(r['Missing_%']):.1f}",
                 f"%{float(r['Exact_Zero_%']):.1f}", r["Median"]])
add_table(doc, ["Panel", "AL_ Kolon Sayısı", "Eksik %", "Tam Sıfır %", "Medyan"], rows,
          caption="AL_ (frekans) kolon grubunun panel bazlı eksiklik ve sıfır-değer oranları.",
          caption_num=2)

para(doc,
     "EK_ (korunmuşluk) grubunda eksiklik kolon bazında değişmektedir: çoğu kolon "
     "%12.2 eksikken, EK_3 %45.4 ve EK_9 %20.7 ile belirgin şekilde daha fazla eksiktir "
     "(Tablo 3). CAT_ grubunda ise eksiklik %12.3 ile %97.7 arasında geniş bir aralığa "
     "yayılmaktadır -- CAT_6 (tekrar-bölgesi bayrağı) neredeyse hiç dolu değildir "
     "(%97.7 eksik), bu da bu bilginin yalnızca istisnai durumlarda kaydedildiğini "
     "düşündürmektedir.")

rows = []
for r in EK_SUMMARY:
    rows.append([r[""], r["min"], r["max"], r["mean"], r["std"], f"%{float(r['missing_pct']):.1f}"])
add_table(doc, ["Kolon", "Min", "Maks", "Ortalama", "Std", "Eksik %"], rows,
          caption="EK_ (korunmuşluk) kolon grubu değer istatistikleri (tüm paneller birleşik).",
          caption_num=3)

para(doc,
     "Tablo 3, EK_ öbeğinin TEK bir ölçekte olmadığını göstermektedir: EK_4, EK_5 ve "
     "EK_6 [0, 1] aralığına sıkı şekilde sınırlıdır (olasılıksal/skor-benzeri, SIFT gibi "
     "sınırlı-aralıklı korunmuşluk tahmincilerini andırmaktadır), buna karşılık EK_1, "
     "EK_2, EK_3, EK_7, EK_8 ve EK_9 negatif değerler de içeren SÜREKLİ bir aralığa "
     "sahiptir (örn. EK_9: -20 ile 11.934 arası; GERP++/phyloP gibi sınırsız-aralıklı "
     "korunmuşluk skorlarını andırmaktadır). Bu ölçek farkı metodolojik olarak önemlidir: "
     "featurizer bu iki alt grubu ayırt etmeden, her EK_ kolonunun medyanını (yalnızca "
     "train fold'undan) kendi ham ölçeğinde kullanmaktadır -- ağaç tabanlı modeller "
     "(XGBoost/LightGBM/RandomForest) mesafe-bazlı değil eşik-bölme-bazlı çalıştığından, "
     "farklı ölçeklerdeki öznitelikler arasında normalizasyon gerektirmemektedir; bu "
     "nedenle ölçek farkının model performansına doğrudan bir olumsuz etkisi "
     "beklenmemektedir, ancak SHAP yorumlamasında (Bölüm 4.5) kolonlar arası ham katkı "
     "büyüklüklerinin doğrudan karşılaştırılmaması gerektiği anlamına gelmektedir.")

rows = []
for r in CAT_SUMMARY:
    rows.append([r["Kolon"], r["N_Kategori"], f"%{float(r['Missing_%']):.1f}", r["Ornekler"]])
add_table(doc, ["Kolon", "Kategori Sayısı", "Eksik %", "Örnek Değerler"], rows,
          caption="CAT_ (kategorik) kolon grubu özet istatistikleri.",
          caption_num=4)

h2(doc, "2.4 “Eksik Değer ≠ Sıfır” Bulgusu")
para(doc,
     "AL_ (frekans) grubunda eksiklik ve tam-sıfır değerleri veri sözlüğünde AYRI AYRI "
     "kaydedilmiş, ölçülebilir iki farklı durumdur (Tablo 2 -- örn. MASTER'da %56.85 "
     "eksik VE %19.2 tam sıfır, aynı anda, birbirinden bağımsız oranlarla). Bu, eksikliğin "
     "“ölçülmüş ama sıfır çıkmış” durumunun bir yan etkisi olmadığını, gerçekten "
     "“bu veritabanında/kohortta hiç gözlenmemiş” anlamına gelen ayrı bir durum "
     "olduğunu göstermektedir. Genomikte popülasyon veritabanlarında hiç gözlenmemiş ya da "
     "aşırı nadir varyantların patojenite önseli, gözlenip de sıklığı gerçekten sıfıra yakın "
     "çıkan varyantlardan farklı yorumlanır -- bu yüzden eksikliği 0 ile doldurmak bu ayrımı "
     "yok ederek bilgi kaybına yol açar. features.py bu ayrımı iki şekilde koruma altına "
     "almaktadır: (i) XGBoost/LightGBM için ham NaN değeri hiç dokunulmadan modele native "
     "olarak veriliyor (bu modeller eksik değer için ayrı bir dallanma yönü öğrenebiliyor); "
     "(ii) RandomForest/stacking meta-öğrenici gibi NaN kabul etmeyen modeller için "
     "0 yerine, gerçek frekans değerlerinin asla alamayacağı bir nadirlik sentineli "
     "(-1.0) kullanılıyor -- böylece “gerçekten sıfır” ile “gözlenmemiş” "
     "hiçbir zaman aynı sayısal değere karışmıyor.")

h2(doc, "2.5 Yedekli Kolon: CAT_5 ≡ CAT_3")
para(doc,
     "Dört panelin tamamında CAT_5 kolonunun CAT_3 ile satır satır birebir aynı olduğu "
     "(pandas .equals() ile) doğrulanmış ve bu tespit yalnızca train fold'undan öğrenilerek "
     "(sızıntı riski olmadan) öznitelik mühendisliği aşamasında otomatik olarak CAT_5 "
     "düşürülmüştür. Bu adım bilgi kaybına yol açmaz, yalnızca gereksiz kolinearite ve "
     "boyutu önler.")

h2(doc, "2.6 Panel İçinde Kasıtlı Tekrarlanan Satırlar")
para(doc,
     "Organizatör Q&A'da panel içindeki bazı satırların (tam öznitelik profili aynı) "
     "kasıtlı olarak tekrarlandığı teyit edilmiştir ve bu satırlar veri setinden "
     "silinmemektedir. Ancak standart StratifiedKFold rastgele satır bazlı böldüğü için, "
     "aynı profildeki iki satırdan biri eğitim, diğeri doğrulama katına düşerse, "
     "doğrulama skoru yapay olarak şişer (model, doğrulama satırını aslında zaten "
     "“görmüş” olur) -- klasik bir grup-sızıntısı riskidir. Bu riske karşı "
     "features.py içindeki row_group_ids() fonksiyonu, Variant_ID ve Label hariç TÜM "
     "kolonların satır imzasını (hash) çıkarıp aynı profildeki satırlara aynı grup "
     "kimliğini atamakta; tüm çapraz doğrulama adımlarında (hiperparametre arama, final "
     "eğitim, SMOTE deneyi) StratifiedGroupKFold kullanılarak bu gruplar asla train/val "
     "arasında bölünmemektedir.")

doc.add_page_break()

# ---------------------------------------------------------------------------
# 3. Metodoloji
# ---------------------------------------------------------------------------

h1(doc, "3. Metodoloji")

h2(doc, "3.1 Ön İşleme ve Öznitelik Mühendisliği")
para(doc,
     "Tüm istatistiksel öğrenme (kategori haritaları, EK_ medyanları, yedekli kolon "
     "tespiti) SADECE ilgili çapraz doğrulama katının train bölümünden öğrenilmekte, "
     "val bölümüne yalnızca uygulanmaktadır (fit/transform ayrımı) -- bu, sonraki sızıntı "
     "denetiminin (Bölüm 3.7) neden temiz çıktığının temel nedenlerinden biridir. "
     "Featurizer iki paralel görünüm üretir: transform_native (XGBoost/LightGBM için, "
     "ham NaN korunur) ve transform_imputed (RandomForest ve stacking meta-öğrenicisi "
     "için, AL_ nadirlik sentineli -1.0, EK_ train-fold medyanı ile dolduruluyor). "
     "Amino asit değişiminin fizikokimyasal büyüklüğü Grantham (1974) formülüyle "
     "(kompozisyon, polarite, hacim farklarının ağırlıklı öklid mesafesi) tek bir sayısal "
     "değere (AA_grantham) indirgenmekte; bilinmeyen veya çok-harfli kodlar (indel/delins "
     "gibi) NaN + ayrı bir AA_unknown_code bayrağıyla işaretlenmektedir. Ayrıca eksikliğin "
     "kendisi de (AL_missing_ratio/count, EK_missing_ratio, CAT_missing_ratio, "
     "TOTAL_missing_ratio) doğrudan öznitelik olarak modele verilmektedir.")

h2(doc, "3.2 Grup-Farkında Çapraz Doğrulama")
para(doc,
     "Tüm CV adımlarında 5 katlı StratifiedGroupKFold kullanılmaktadır: hem sınıf "
     "dengesini (Stratified) hem de Bölüm 2.6'da açıklanan tekrarlanan-satır gruplarını "
     "(Group) aynı anda koruyan tek scikit-learn sınıfı budur.")

h2(doc, "3.3 Model Ailesi")
para(doc,
     "Panel başına dört aday model karşılaştırılmaktadır: XGBoost ve LightGBM (native "
     "eksik-değer desteğiyle, tree-native görünüm üzerinde), RandomForest (imputed "
     "görünüm, class_weight=‘balanced’) ve Stacking -- üç temel modelin "
     "out-of-fold olasılık çıktılarını girdi alan, kendisi de ayrı bir 5-katlı OOF "
     "döngüsüyle eğitilen lojistik regresyon meta-öğrenicisi. Stacking'in meta-öğrenicisi "
     "de OOF prensibiyle eğitildiği için, temel modellerin kendi eğitim verilerini "
     "“görmüş” olmasından kaynaklanan iyimser yanlılık meta-katmana sızmaz.")

h2(doc, "3.4 Hiperparametre Optimizasyonu")
para(doc,
     "Panel başına, model başına (XGBoost/LightGBM/RandomForest) ayrı Optuna TPE arama "
     "çalıştırılmıştır (panel büyüklüğüne göre 12-25 deneme). Amaç fonksiyonu "
     "0.5×OOF-ROC-AUC + 0.5×OOF-F1(@0.5) olarak tanımlanmıştır; böylece hem "
     "genel sıralama gücü hem de varsayılan eşikteki denge birlikte gözetilmektedir. "
     "XGBoost için scale_pos_weight de arama uzayına dahil edilmiştir. "
     "ÖNEMLİ metodolojik karar: hiperparametre arama seed'i (101), final "
     "değerlendirmede kullanılan seed'lerden ({42, 123, 456}) BİLİNÇLİ olarak "
     "farklı tutulmuştur. Aksi halde arama ve final raporlama aynı fold bölünmesini "
     "kullanır ve arama sürecinin o spesifik bölünmeye aşırı uyum sağlamasından "
     "kaynaklanan iyimser yanlılık (optimistic bias) ortaya çıkar -- bu düzeltme "
     "danışmanımızın geri bildirimi üzerine uygulanmıştır.")

h2(doc, "3.5 Final Eğitim ve Tekrarlı OOF Üretimi")
para(doc,
     "Ayarlanmış hiperparametrelerle final eğitim, 3 farklı seed'te (42, 123, 456) "
     "StratifiedGroupKFold tekrarlanarak yapılmakta; her seed'in ürettiği OOF olasılıkları "
     "ortalanarak (bagged OOF) nihai olasılık skorları elde edilmektedir. Bu, tek bir "
     "seed'in şans eseri elverişli/elverişsiz fold bölünmesine bağımlılığı azaltmaktadır.")

h2(doc, "3.6 Test-Bilinçli Eşik Seçimi ve Bootstrap Doğrulama")
para(doc,
     "Bölüm 1.1'de açıklanan ters çevrilmiş sınıf dağılımı nedeniyle, eğitim OOF havuzu "
     "üzerinde F1'i maksimize eden “naif” bir eşik seçmek, test kümesindeki "
     "gerçek performansı YANLIŞ temsil eder -- çünkü F1 doğrudan sınıf önseline duyarlıdır. "
     "Bunun yerine, modelin sınıf-koşullu skor dağılımının (TPR(t) ve TNR(t) fonksiyonları) "
     "yalnızca önsel değiştiği sürece train ile test arasında transfer olduğu varsayımıyla "
     "-- ki bu, yarışmanın kendi tasarımıyla (“aynı panel, sadece oran ters çevrilmiş”) "
     "birebir örtüşmektedir -- eğitim OOF'undan tahmin edilen TPR(t)/TNR(t) fonksiyonları, "
     "PANELİN BİLİNEN test Pat/Ben sayısına (organizatör tarafından bildirilen, Tablo 1) "
     "projekte edilerek her eşik için beklenen test F1'i hesaplanmakta ve bu projekte "
     "F1'i maksimize eden eşik seçilmektedir. Seçimin güvenilirliğini sınamak için, "
     "eğitim OOF havuzundan (patojenik ve benign ayrı ayrı) test büyüklüğüne göre 2000 "
     "kez tekrarlı örneklem (bootstrap) alınmakta, her örneklemde GERÇEK F1/MCC "
     "hesaplanmakta ve ortalama ile %90 güven aralığı (5.-95. yüzdelik) raporlanmaktadır.")

h2(doc, "3.7 Veri Sızıntısı Denetimi")
para(doc,
     "Her panelde her ham öznitelik (AL_, EK_) TEK BAŞINA, 5 katlı StratifiedKFold ile "
     "out-of-fold AUC hesaplanarak etiketle karşılaştırılmaktadır (eksik değerler "
     "yalnızca train fold medyanıyla dolduruluyor). Tek bir öznitelik başlı başına "
     "0.90'ın üzerinde AUC veriyorsa bu, olası bir sızıntı (örn. etiketle doğrudan "
     "türetilmiş bir kolon) sinyali olarak işaretlenmektedir.")

h2(doc, "3.8 SMOTE Deneyi")
para(doc,
     "Organizatör Q&A'da sentetik veri çoğaltmanın (SMOTE) kullanılabileceği açıkça "
     "onaylanmıştır. Bu deney, SMOTE'un panel bazlı katkısını kontrollü şekilde ölçmek "
     "için tasarlanmıştır: SMOTE SADECE train fold'una uygulanmakta (val fold'una asla "
     "-- aksi halde val'deki gerçek örneklerin sentetik komşuları train'de bulunur ve bu "
     "sızıntıya yol açar), yalnızca imputed (NaN'siz) öznitelik uzayında çalışmaktadır "
     "(k-en-yakın-komşu mesafesi NaN ile hesaplanamaz) ve bu yüzden yalnızca RandomForest "
     "üzerinde, class_weight=‘balanced’ temel çizgisine karşı test edilmektedir "
     "-- XGBoost/LightGBM'in native eksik-değer avantajını bozmamak için bu iki modele "
     "SMOTE uygulanmamıştır.")

h2(doc, "3.9 SHAP ile Kategori Bazlı Açıklanabilirlik")
para(doc,
     "Panel başına, threshold_results.json'da seçilen modelin (Stacking seçildiğinde "
     "kendisi doğrudan açıklanamadığından en güçlü temel bileşeni -- genelde XGBoost -- "
     "kullanılmaktadır) TÜM panel verisi üzerinde YENİDEN eğitilmiş bir kopyasına "
     "SHAP TreeExplainer uygulanmaktadır. Bu adımın tek amacı açıklanabilirliktir; "
     "performans sayıları bu yeniden eğitilmiş kopyadan DEĞİL, CV/OOF sürecinden "
     "raporlanmaktadır (karışıklık olmaması için kodda ve bu raporda bilinçli olarak "
     "ayrılmıştır). Ortalama mutlak SHAP katkıları öznitelik önekine (AL_/EK_/CAT_/AA_) "
     "göre toplanıp yüzdeye çevrilmektedir.")

h2(doc, "3.10 Seed Kararlılığı Doğrulaması")
para(doc,
     "Bölüm 3.5'teki 3-seed bagged OOF yaklaşımı (olasılıkların seed'ler arasında "
     "ORTALANMASI) tek-seed rastlantısallığını zaten azaltan bir tasarımdır; ancak bu "
     "yaklaşım TEK BAŞINA, 3 seed'in birbirinden ne kadar farklı sonuç verdiğini "
     "(saçılımını) açıkça göstermez. Danışmanımızın “tek seed rastlantısal "
     "varyansı gizler, en az 3-5 seed ile Ortalama ± Standart Sapma raporlanmalı” "
     "uyarısını doğrudan karşılamak için ayrı bir doğrulama adımı eklenmiştir: panel "
     "başına seçilen model, panele zaten atanmış SABİT test-eşiğinde, 3 seed'in HER "
     "BİRİ için TAM BAĞIMSIZ bir 5-katlı CV (stacking meta-adımı dahil) ile tekrar "
     "çalıştırılmakta, her seed'in kendi test-projekte F1/MCC'si ayrı ayrı "
     "hesaplanmakta ve 3 değer arası ortalama ± standart sapma raporlanmaktadır "
     "(ddof=1). Bu, raporun ana metriği olan 3-seed bagged sonucun YERİNE değil, "
     "ONA EK bir kararlılık/güvenilirlik kanıtı olarak sunulmaktadır -- sonuçlar "
     "Bölüm 4.6'dadır.")

doc.add_page_break()

# ---------------------------------------------------------------------------
# 4. Deneysel Sonuçlar
# ---------------------------------------------------------------------------

h1(doc, "4. Deneysel Sonuçlar")

h2(doc, "4.1 Panel Bazlı Final Model Karşılaştırması (0.5 Eşiğinde)")
para(doc,
     "Tablo 5, ayarlanmış hiperparametrelerle eğitilen dört adayın, henüz test-bilinçli "
     "kalibrasyon uygulanmadan, varsayılan 0.5 eşiğinde (yani eğitim dağılımı altında) "
     "OOF performansını göstermektedir. Bu tablo panel içi model seçiminin ham "
     "gerekçesini oluşturmakta; nihai model + eşik seçimi Bölüm 4.2'de test-bilinçli "
     "projeksiyonla yapılmaktadır.")

for panel in PANELS:
    rows = []
    for model_name, m in FINALCV[panel].items():
        rows.append([model_name, f"{m['cv_f1_at_0.5']:.3f}", f"{m['cv_mcc_at_0.5']:.3f}",
                     f"{m['cv_balanced_acc_at_0.5']:.3f}",
                     f"{m['roc_auc']:.3f}", f"{m['auprc']:.3f}", f"{m['brier_score']:.3f}"])
    add_table(doc, ["Model", "F1@0.5", "MCC@0.5", "Dengeli Doğruluk", "ROC-AUC", "AUPRC", "Brier"], rows,
              caption=f"{panel} paneli -- dört adayın OOF performansı (0.5 eşiği, eğitim dağılımı).",
              caption_num=f"5.{PANELS.index(panel)+1}")

h2(doc, "4.2 Seçilen Modeller, Kalibre Eşikler ve Test-Projekte Performans")
para(doc,
     "Tablo 6, her panel için seçilen modeli, test-bilinçli eşiği (Bölüm 3.6) ve bu "
     "eşikteki projekte performansı; ayrıca karşılaştırma için “naif” "
     "(eğitim dağılımına göre F1-maksimize eden) eşiğin aynı projeksiyonda verdiği "
     "sonucu göstermektedir.")

rows = []
for panel in PANELS:
    t = THRESH[panel]
    boot = t["bootstrap"]
    rows.append([
        panel, t["selected_model"],
        f"{t['naive_threshold']:.2f}", f"{t['naive_f1_projected']:.3f}",
        f"{t['test_threshold']:.2f}", f"{t['projected_test_f1']:.3f}",
        f"{t['projected_mcc']:.3f}",
        f"[{boot['f1_ci'][0]:.3f}, {boot['f1_ci'][1]:.3f}]",
    ])
add_table(doc,
          ["Panel", "Model", "Naif Eşik", "Naif F1(proj.)", "Test Eşik", "Test F1(proj.)", "Test MCC(proj.)", "Boot. F1 %90 GA"],
          rows,
          caption="Naif eşiğe karşı test-bilinçli eşiğin projekte performansı ve bootstrap doğrulaması.",
          caption_num=6)

para(doc,
     "Test-bilinçli kalibrasyonun katkısı panel bazında büyük farklılık göstermektedir: "
     "MASTER'da naif eşik (0.37) yalnızca 0.371 projekte F1 verirken, test-bilinçli eşik "
     "(0.86) 0.532'ye çıkmaktadır (+0.161 mutlak fark). KANSER'de naif 0.29→0.490'a "
     "karşı test-bilinçli 0.77→0.645 (+0.155); PAH'ta naif 0.49→0.524'e karşı "
     "test-bilinçli 0.76→0.613 (+0.089); CFTR'de ise en büyük farkla naif 0.63→"
     "0.408'e karşı test-bilinçli 0.81→0.606 (+0.197). Bu farklar, sadece “doğru "
     "modeli seçmenin” değil, “doğru eşiği seçmenin” de en az o kadar kritik "
     "olduğunu göstermektedir -- ters çevrilmiş test dağılımı göz ardı edilirse, "
     "eğitim-dağılımına göre “optimal” görünen bir eşik test kümesinde belirgin "
     "şekilde daha kötü performans verecektir. CFTR'nin bootstrap %90 güven aralığı "
     "([0.489, 0.723]) diğer panellere göre en geniştir; bu, n=111'lik küçük örneklem "
     "büyüklüğünün doğal bir sonucu olarak Bölüm 5'te ayrıca tartışılmaktadır.")

add_figure(doc, "fig2_esik_tarama.png",
           "Panel başına eğitim (ham OOF) F1 eğrisi ile test-projekte F1 eğrisinin eşiğe göre "
           "karşılaştırması. İki eğrinin tepe noktalarının belirgin şekilde ayrışması, naif "
           "eşiğin neden sistematik olarak yanlış olduğunu görsel olarak doğrulamaktadır.",
           caption_num=2)

add_figure(doc, "fig4_karmasiklik.png",
           "Seçilen model + test-bilinçli eşiğin, panelin bilinen test Pat/Ben sayısına "
           "projekte edilmiş 2×2 karmaşıklık matrisi (mutlak sayı ve satır-yüzdesi).",
           caption_num=4)

h2(doc, "4.3 Veri Sızıntısı Denetim Sonuçları")
rows = []
for panel in PANELS:
    l = LEAK[panel]
    flag = "Şüpheli" if l["suspicious"] else "Temiz"
    rows.append([panel, l["max_feature_name"], f"{l['max_single_feature_auc']:.3f}", flag])
add_table(doc, ["Panel", "En Yüksek Tekil Öznitelik", "OOF AUC", "Durum"], rows,
          caption="Panel bazlı sızıntı denetimi -- en yüksek tekil öznitelik AUC'si (eşik: 0.90).",
          caption_num=7)
para(doc,
     "Dört panelin tamamında en yüksek tekil öznitelik AUC'si 0.90 eşiğinin altında "
     "kalmıştır (0.701-0.803 aralığında); hiçbir panelde şüpheli sızıntı sinyali "
     "tespit edilmemiştir. En yüksek değer beklenildiği gibi CFTR'de (0.803, EK_9) "
     "görülmektedir -- en küçük örneklem büyüklüğü (n=111), tekil öznitelik AUC "
     "tahminlerinin doğal olarak daha gürültülü/yüksek varyanslı çıkmasına yol "
     "açmaktadır, ancak bu yine de eşik altında kalmaktadır.")

h2(doc, "4.4 SMOTE Ablasyon Sonuçları")
rows = []
for panel in PANELS:
    b, s = SMOTE[panel]["baseline_class_weight"], SMOTE[panel]["smote"]
    d_f1, d_mcc = s["f1"] - b["f1"], s["mcc"] - b["mcc"]
    rows.append([panel, f"{b['f1']:.3f}", f"{s['f1']:.3f}", f"{d_f1:+.3f}",
                 f"{b['mcc']:.3f}", f"{s['mcc']:.3f}", f"{d_mcc:+.3f}"])
add_table(doc, ["Panel", "Baz F1", "SMOTE F1", "ΔF1", "Baz MCC", "SMOTE MCC", "ΔMCC"], rows,
          caption="class_weight=balanced temel çizgisine karşı SMOTE'un RandomForest üzerindeki etkisi.",
          caption_num=8)
para(doc,
     "SMOTE yalnızca CFTR panelinde anlamlı ve tutarlı bir katkı sağlamaktadır "
     "(F1 +0.019, MCC +0.174). MCC'deki kazancın F1'dekinden çok daha büyük olması "
     "dikkat çekicidir -- bu, SMOTE'un CFTR'de ham doğru-tahmin sayısından çok, "
     "TP/TN dengesini (yani modelin hem patojenik hem benign sınıfı simetrik şekilde "
     "ayırt etme gücünü) iyileştirdiğini göstermektedir; CFTR'nin en küçük panel "
     "olması (n=111, yalnızca 21 benign örnek) nedeniyle sentetik azınlık-sınıf "
     "örneklemesinden en çok fayda gören panel olması beklenen bir sonuçtur. Diğer üç "
     "panelde (MASTER, KANSER, PAH) SMOTE ya düz ya da hafif negatif etki göstermektedir "
     "-- bu panellerde zaten class_weight=‘balanced’ ile yeterli örneklem "
     "büyüklüğü sağlandığından sentetik örneklerin ek bir fayda sağlamadığı "
     "değerlendirilmektedir. Bu bulgu doğrultusunda SMOTE, organizatörün genel onayına "
     "rağmen tüm panellere körlemesine uygulanmamış, yalnızca ampirik olarak fayda "
     "gösterdiği CFTR panelinde değerlendirmeye alınmıştır.")

h2(doc, "4.5 SHAP Kategori Bazlı Açıklanabilirlik")
rows = []
for panel in PANELS:
    cc = SHAP[panel]["category_contribution_pct"]
    rows.append([panel, SHAP[panel]["model_used"],
                 f"%{cc.get('AL (frekans)', 0):.1f}", f"%{cc.get('EK (korunmuşluk)', 0):.1f}",
                 f"%{cc.get('AA (amino asit)', 0):.1f}", f"%{cc.get('CAT (kategorik)', 0):.1f}"])
add_table(doc, ["Panel", "Kullanılan Model", "AL %", "EK %", "AA %", "CAT %"], rows,
          caption="Panel bazlı SHAP kategori katkı yüzdeleri.",
          caption_num=9)
para(doc,
     "Dört panelin tamamında tutarlı bir örüntü görülmektedir: AL_ (frekans) öbeği "
     "kararın çoğunluğunu açıklamaktadır (%62.9-72.0), EK_ (korunmuşluk) öbeği ikinci "
     "sırada gelmektedir (%17.8-33.0), AA_ ve CAT_ öbekleri ise görece küçük katkı "
     "sağlamaktadır. En etkili tekil öznitelikler panel bazında farklılık göstermektedir: "
     "MASTER'da EK_7 tek başına toplam katkının önemli bir kısmını oluştururken "
     "(SHAP değeri 0.421, listenin geri kalanının toplamına yakın), KANSER'de AL_16, "
     "PAH'ta EK_7, CFTR'de ise EK_9 en etkili öznitelik olarak öne çıkmaktadır. Bu "
     "örüntü, modelin klinik olarak makul bir sinyale dayandığını göstermektedir: hem "
     "popülasyon veritabanlarındaki nadirlik/gözlenmeme durumu hem de evrimsel "
     "korunmuşluk, missense varyant patojenitesinin bilinen iki temel göstergesidir.")

add_figure(doc, "fig3_shap_kategori.png",
           "Panel bazlı SHAP kategori katkı yüzdelerinin yığılmış çubuk gösterimi.",
           caption_num=3)

h2(doc, "4.6 Seed Kararlılığı Sonuçları")
para(doc,
     "Bölüm 3.5'teki 3-seed bagged OOF yaklaşımı, tek bir seed'in şans eseri elverişli "
     "fold bölünmesine bağımlılığı zaten azaltmaktadır; ancak bu, seed'ler arası ne kadar "
     "SAÇILIM olduğunu doğrudan göstermez. Bu soruyu ayrıca yanıtlamak için, panel başına "
     "seçilen modelin PANELE ZATEN ATANMIŞ SABİT test-eşiğinde, 3 seed'in HER BİRİ "
     "tam bağımsız bir 5-katlı CV (gerekiyorsa stacking meta-adımı dahil) ile TEK BAŞINA "
     "çalıştırılmış, test-projekte F1/MCC'si ayrı ayrı hesaplanmış ve seed'ler arası "
     "Ortalama ± Standart Sapma raporlanmıştır (Tablo 10). Raporun ana metriği olan "
     "3-seed bagged değer, karşılaştırma için aynı tabloda yeniden verilmiştir.")

rows = []
for panel in PANELS:
    s = SEED_STAB[panel]
    rows.append([
        panel, s["model"],
        f"{s['f1_mean']:.3f} ± {s['f1_std']:.3f}",
        f"{s['mcc_mean']:.3f} ± {s['mcc_std']:.3f}",
        f"{s['bagged_f1_reference']:.3f}",
        f"{s['bagged_mcc_reference']:.3f}",
    ])
add_table(doc,
          ["Panel", "Model", "Tek-Seed F1 (Ort.±Std)", "Tek-Seed MCC (Ort.±Std)", "3-Seed Bagged F1", "3-Seed Bagged MCC"],
          rows,
          caption="Sabit test-eşiğinde, 3 seed'in ayrı ayrı çalıştırılmasından elde edilen "
                  "F1/MCC ortalaması ve standart sapması; 3-seed bagged OOF sonucuyla karşılaştırma.",
          caption_num=10)

para(doc,
     "MASTER, KANSER ve PAH'ta seed-arası standart sapma düşüktür (F1 std sırasıyla "
     "0.009, 0.014, 0.009) ve bagged sonuç tek-seed ortalamasının belirgin şekilde "
     "üzerindedir (örn. PAH'ta 0.574→0.613, +0.039) -- bu, 3-seed bagging'in gerçekten "
     "varyans azaltıcı bir etkisi olduğunu doğrulamaktadır. CFTR ise burada da beklenen "
     "örüntüyü tekrarlamaktadır: seed-arası F1 standart sapması (±0.081) diğer üç "
     "panelin 6-9 katı büyüklüğündedir (tek-seed F1 değerleri 0.502-0.664 arasında geniş "
     "bir aralığa yayılmaktadır) -- bu, Bölüm 4.2'deki bootstrap güven aralığı bulgusundan "
     "TAMAMEN BAĞIMSIZ bir yöntemle (seed varyansı vs. örneklem varyansı) aynı sonuca "
     "ulaşmaktadır: CFTR panelinin projekte performansı, n=111'lik küçük örneklem "
     "büyüklüğü nedeniyle diğer panellere göre yapısal olarak daha az kararlıdır.")

doc.add_page_break()

# ---------------------------------------------------------------------------
# 5. Tartışma ve Sınırlamalar
# ---------------------------------------------------------------------------

h1(doc, "5. Tartışma ve Sınırlamalar")

bullet(doc,
       "Anonim kolonlar ve genomik adres eksikliği: dış veritabanlarıyla (ClinVar, "
       "gnomAD, REVEL) satır bazlı eşleştirme yapılamamaktadır (Bölüm 1.2). Bu, hem "
       "projenin sağlanan veriyle kendi kendine yeterli kalmasını zorunlu kılan hem de "
       "yarışmanın kasıtlı olarak koyduğu bir kısıttır (dış veriden “kopya "
       "çekilmesini” önlemek amacıyla anlaşılmaktadır).")
bullet(doc,
       "Tahmini metriklerin doğası: Bölüm 3.6'daki test-bilinçli projeksiyon, "
       "sınıf-koşullu skor dağılımının (yalnızca önsel değişse de) train ile test "
       "arasında transfer olduğu varsayımına dayanmaktadır. Bu varsayım yarışmanın "
       "kendi tasarım açıklamasıyla tutarlı olsa da, gerçek yarışma test verisi elimizde "
       "olmadığı için doğrulanamamaktadır. Bootstrap güven aralıkları bu belirsizliğin "
       "bir kısmını yansıtmakta, ancak nihai skorun gerçek yarışma testinde bu aralığın "
       "dışına çıkması olasılık dışı değildir.")
bullet(doc,
       "CFTR küçük örneklem riski: n=111 eğitim satırı (yalnızca 21 benign örnek), "
       "projede en dar değil en GENİŞ bootstrap güven aralığına sahip panel olmasına "
       "yol açmaktadır (F1 için [0.489, 0.723]) -- bu panelin projekte performansı "
       "diğer üç panele kıyasla en az güvenilir tahmindir. Bu bulgu, TAMAMEN BAĞIMSIZ "
       "bir yöntemle -- seed-arası varyans (Bölüm 4.6) -- doğrulanmaktadır: CFTR'nin "
       "seed-arası F1 standart sapması (±0.081) diğer üç panelin (±0.009-0.014) "
       "6-9 katıdır; iki farklı belirsizlik kaynağı (örneklem varyansı ve seed "
       "varyansı) birbirinden bağımsız olarak aynı sonuca işaret etmektedir.")
bullet(doc,
       "Panel-arası performans farkının yorumu: MASTER en düşük projekte F1'e (0.532) "
       "sahip olsa da, bu MASTER modelinin “daha kötü” olduğu anlamına "
       "gelmemektedir -- MASTER, test kümesinde patojenik oranın en agresif şekilde "
       "düştüğü (test'te yalnızca %14.3) paneldir ve bu da F1 metriğini yapısal olarak "
       "diğer panellere göre daha fazla cezalandırmaktadır.")
bullet(doc,
       "Stacking modelinin açıklanamazlığı: Stacking, MASTER panelinde en iyi sonucu "
       "veren model olmasına rağmen SHAP ile doğrudan açıklanamadığından (meta-öğrenici "
       "üç ayrı modelin çıktısını birleştirir), Bölüm 4.5'teki açıklanabilirlik analizi "
       "MASTER için stacking yerine en güçlü temel bileşeni (XGBoost) kullanmak zorunda "
       "kalmıştır -- bu, klinik yorumlanabilirlik açısından stacking'in bir dezavantajı "
       "olarak not edilmelidir.")

doc.add_page_break()

# ---------------------------------------------------------------------------
# 6. Sonuç ve Gelecek Çalışmalar
# ---------------------------------------------------------------------------

h1(doc, "6. Sonuç ve Gelecek Çalışmalar")

para(doc,
     "GenVision, dört panelin her biri için ayrı eğitilmiş, grup-farkında çapraz "
     "doğrulama ile sızıntıdan arındırılmış, Optuna ile ayarlanmış ve yarışmanın "
     "kasıtlı ters-çevrilmiş test dağılımına göre eşik-kalibre edilmiş modeller "
     "üretmiştir. Tablo 10, nihai seçilen model + eşik kombinasyonlarının özetini "
     "vermektedir.")

rows = []
for panel in PANELS:
    t = THRESH[panel]
    rows.append([panel, t["selected_model"], f"{t['test_threshold']:.2f}",
                 f"{t['projected_test_f1']:.3f}", f"{t['projected_mcc']:.3f}"])
add_table(doc, ["Panel", "Seçilen Model", "Karar Eşiği", "Tahmini Test F1", "Tahmini Test MCC"], rows,
          caption="Nihai panel bazlı model, eşik ve tahmini test performansı özeti.",
          caption_num=11)

para(doc,
     "Dört panelin tahmini F1 skorlarının ortalaması ~0.599'dur; bu değer, raporun "
     "başında da vurgulandığı gibi, gerçek yarışma test verisi üzerinde değil, eğitim "
     "verisinden istatistiksel olarak projekte edilmiş ve bootstrap ile doğrulanmış bir "
     "tahmindir.")

h2(doc, "6.1 Planlanan Ek Çalışmalar")
para(doc,
     "Danışmanımızın geri bildirimi doğrultusunda, bu rapor tesliminden sonra "
     "tamamlanması planlanan iki ek ablasyon bulunmaktadır:")
bullet(doc,
       "Outlier dayanıklılığı karşılaştırması: ham, aykırı-değer-düzeltilmiş ve "
       "winsorized öznitelik versiyonlarının panel bazlı model performansına etkisinin "
       "sistematik olarak karşılaştırılması.")
bullet(doc,
       "Eksiklik stratejisi ablasyonu: tek bir model üzerinde, native-NaN, sentinel "
       "değer ve medyan doldurma stratejilerinin kontrollü şekilde karşılaştırılması "
       "-- Bölüm 2.4'teki “eksik ≠ sıfır” bulgusunun performansa gerçek "
       "katkısının izole edilmesi.")

para(doc,
     "Ayrıca, veri anonimliği kısıtı (Bölüm 1.2) satır bazlı dış veri eşleştirmesini "
     "engellese de, panel/varyant-tipi düzeyinde DOLAYLI istatistiksel karşılaştırmalar "
     "(örneğin ClinVar'daki genel patojenik/benign oran dağılımlarıyla panel önsellerinin "
     "karşılaştırılması gibi) değerlendirilmeye açık bir yön olarak not edilmektedir; "
     "ancak bu yaklaşımın kesin bir performans katkısı garanti edilmemektedir.")

# ---------------------------------------------------------------------------
# Kaynakça
# ---------------------------------------------------------------------------

h1(doc, "Kaynakça")
refs = [
    "Grantham, R. (1974). Amino acid difference formula to help explain protein evolution. Science, 185(4154), 862-864.",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD.",
    "Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework. KDD.",
    "Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.",
    "Chawla, N. V., et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique. JAIR, 16, 321-357.",
]
for r in refs:
    bullet(doc, r)

# ---------------------------------------------------------------------------
out_path = REPORTS_DIR / "GenVision_Rapor.docx"
doc.save(out_path)
print("Kaydedildi:", out_path)
