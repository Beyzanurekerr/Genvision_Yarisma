import os
import json
import subprocess

def test_determinism():
    print("GENVISION DETERMINISM (TUTARLILIK) TESTİ BAŞLATILIYOR...\n")
    
    json_path = os.path.join('sonuclar', 'TEAM_918091_FINAL.json')
    
    # 1. Çalıştırma: Tahmin üret
    print("-> 1. Tahmin turu çalıştırılıyor...")
    subprocess.run(["python", "predict_test.py"], capture_output=True)
    
    if not os.path.exists(json_path):
        print("[HATA]: İlk turda JSON üretilemedi!")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        run1_data = json.load(f)

    # 2. Çalıştırma: Aynı veriyi tekrar tahmin et
    print("-> 2. Tahmin turu çalıştırılıyor...")
    subprocess.run(["python", "predict_test.py"], capture_output=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        run2_data = json.load(f)

    # Karşılaştırma
    match = True
    preds1 = run1_data['predictions']
    preds2 = run2_data['predictions']

    if len(preds1) != len(preds2):
        match = False
    else:
        for p1, p2 in zip(preds1, preds2):
            if p1['id'] != p2['id'] or p1['predicted_class'] != p2['predicted_class'] or p1['predicted_prob'] != p2['predicted_prob']:
                match = False
                break

    print("="*50)
    if match:
        print("[BAŞARILI]: Determinism testi geçti! Ard arda yapılan iki çalıştırma %100 aynı sonuçları verdi.")
    else:
        print("[HATA]: Tutarsızlık tespit edildi! İki çalıştırma birbirinden farklı sonuçlar üretti.")
    print("="*50)

if __name__ == "__main__":
    test_determinism()