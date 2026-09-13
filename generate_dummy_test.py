import os
import pandas as pd

def create_dummy_test_files():
    panels = ['MASTER', 'KANSER', 'PAH', 'CFTR']
    train_dir = 'data'
    test_dir = os.path.join('data', 'test')
    os.makedirs(test_dir, exist_ok=True)

    for panel in panels:
        train_path = os.path.join(train_dir, f'YARISMA_TRAIN_{panel}.csv')
        if not os.path.exists(train_path):
            print(f"{train_path} bulunamadı.")
            continue
            
        df = pd.read_csv(train_path)
        
        # Eğitim verisinin ilk 50 satırını test verisi olarak alalım
        df_dummy = df.head(50).copy()
        
        # Gerçek test setinde Label (hedef) olmayabilir, o yüzden Label'ı kaldıralım
        if 'Label' in df_dummy.columns:
            df_dummy = df_dummy.drop(columns=['Label'])
            
        # Test dosyasını kaydet
        test_path = os.path.join(test_dir, f'YARISMA_TEST_{panel}.csv')
        df_dummy.to_csv(test_path, index=False)
        print(f">> Sahte test dosyası oluşturuldu: {test_path}")

if __name__ == "__main__":
    create_dummy_test_files()