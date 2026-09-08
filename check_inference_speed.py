import os
import time
import subprocess

def test_inference_speed():
    print("İNFERENCE SÜRESİ (HIZ) TESTİ BAŞLATILIYOR...")
    start_time = time.time()
    
    subprocess.run(["python", "predict_test.py"], capture_output=True)
    
    elapsed_time = time.time() - start_time
    print(f"Toplam Tahmin Süresi: {elapsed_time:.2f} saniye")
    
    if elapsed_time < 30.0:
        print("[BAŞARILI]: Hız testi geçti! Sunucu timeout sınırlarının çok altında.")
    else:
        print("[UYARI]: Çalışma süresi 30 saniyeyi aştı, optimizasyon gerekebilir.")

if __name__ == "__main__":
    test_inference_speed()