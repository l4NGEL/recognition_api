import os
import shutil
import pandas as pd

def organize_dataset(csv_path, image_dir, output_dir):
    # CSV'yi yükle
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"CSV dosyası okunamadı: {e}")
        return

    # Klasörleme işlemi
    for _, row in df.iterrows():
        filename = str(row['id']).strip()
        label = str(row['label']).strip()

        src_path = os.path.join(image_dir, filename)
        dst_folder = os.path.join(output_dir, label)
        dst_path = os.path.join(dst_folder, filename)

        os.makedirs(dst_folder, exist_ok=True)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)  # orijinal dosyayı korumak için kopyalıyoruz
            print(f"✅ {filename} → {label}/ klasörüne kopyalandı")
        else:
            print(f"❌ Dosya bulunamadı: {src_path}")

if __name__ == "__main__":
    # 📂 Bu yolları kendi bilgisayarındaki klasörlere göre ayarla:
    CSV_PATH = "Faces\Dataset.csv"  # VSCode klasörüne göre yol
    IMAGE_DIR = "Faces\Faces"
    OUTPUT_DIR = "known_faces"

    organize_dataset(CSV_PATH, IMAGE_DIR, OUTPUT_DIR)
