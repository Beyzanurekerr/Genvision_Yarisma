import os
import json

def validate_json_schema():
    print("GENVISION TEKNOFEST JSON ŞEMA DOĞRULAMA TESTİ BAŞLATILIYOR...\n")
    
    json_path = os.path.join('sonuclar', 'TEAM_918091_FINAL.json')
    
    if not os.path.exists(json_path):
        print(f"[HATA]: {json_path} dosyası bulunamadı! Önce predict_test.py çalıştırmalısın.")
        return

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[HATA]: JSON dosyası okunamadı veya geçersiz formatta! Detay: {e}")
        return

    # 1. Kök Anahtar Kontrolleri (TEKNOFEST Kılavuz Şeması)
    required_keys = ['team_name', 'team_id', 'application_id', 'competition_level', 'predictions']
    for key in required_keys:
        if key not in data:
            print(f"[HATA]: Eksik kök anahtar: '{key}'")
            return

    # 2. Takım Bilgisi Doğrulaması
    if str(data['team_id']) != '918091' or str(data['application_id']) != '918091':
        print(f"[HATA]: Takım ID veya Başvuru ID uyuşmazlığı! Beklenen: 918091[cite: 1]")
        return

    # 3. Predictions Dizisi ve İçerik Şema Kontrolü
    predictions = data['predictions']
    if not isinstance(predictions, list) or len(predictions) == 0:
        print(f"[HATA]: 'predictions' alanı boş veya liste formatında değil![cite: 1]")
        return

    pred_required_keys = ['id', 'panel', 'predicted_class', 'predicted_prob']
    for i, pred in enumerate(predictions):
        for r_key in pred_required_keys:
            if r_key not in pred:
                print(f"[HATA]: Predictions[{i}] içinde eksik alan: '{r_key}'[cite: 1]")
                return
        
        # Olasılık Sınırları Kontrolü (0.0 ile 1.0 arasında olmalı)
        prob = pred['predicted_prob']
        if not (0.0 <= float(prob) <= 1.0):
            print(f"[HATA]: Geçersiz olasılık değeri ({prob}) atandı![cite: 1]")
            return

    print("="*50)
    print(f"[BAŞARILI]: JSON dosyası TEKNOFEST 2026 resmi şema kurallarına %100 uyumlu![cite: 1]")
    print(f"Toplam Denetlenen Tahmin Sayısı: {len(predictions)}[cite: 1]")
    print("="*50)

if __name__ == "__main__":
    validate_json_schema()