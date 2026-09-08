import subprocess
import sys

def run_pipeline():
    # Aktif sanal ortamın python çalıştırıcısını dinamik olarak alıyoruz
    python_exec = sys.executable
    
    steps = [
        ("1. Overfitting Kontrolü", [python_exec, "check_overfitting.py"]),
        ("2. Şema Drift Testi", [python_exec, "check_schema_drift.py"]),
        ("3. JSON Şema Doğrulama", [python_exec, "check_json_schema.py"]),
        ("4. NaN Dayanıklılık Testi", [python_exec, "check_nan_robustness.py"]),
        ("5. Determinism Testi", [python_exec, "check_determinism.py"]),
        ("6. Final Tahmin Üretimi", [python_exec, "predict_test.py"])
    ]
    
    for name, cmd in steps:
        print(f"\n[ÇALIŞTIRILIYOR]: {name}...")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"[HATA]: {name} aşamasında hata alındı!")
            return

    print("\n" + "="*50)
    print("TEBRİKLER! TÜM BORU HATTI VE TESTLER HATASIZ TAMAMLANDI.")
    print("="*50)

if __name__ == "__main__":
    run_pipeline()