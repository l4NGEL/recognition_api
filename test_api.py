import requests
import json
import base64
import os
from PIL import Image
import io

# API base URL
BASE_URL = "http://127.0.0.1:5000"

def test_health_check():
    """Sağlık kontrolü testi"""
    print("=== Sağlık Kontrolü Testi ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Hata: {e}")
        return False

def test_get_users():
    """Kullanıcı listesi testi"""
    print("\n=== Kullanıcı Listesi Testi ===")
    try:
        response = requests.get(f"{BASE_URL}/users")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Hata: {e}")
        return False

def image_to_base64(image_path):
    """Resmi base64'e çevir"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        print(f"Resim yükleme hatası: {e}")
        return None

def test_add_user(name, image_path):
    """Kullanıcı ekleme testi"""
    print(f"\n=== Kullanıcı Ekleme Testi: {name} ===")
    try:
        # Resmi base64'e çevir
        image_base64 = image_to_base64(image_path)
        if not image_base64:
            return False
        
        # API'ye gönder
        data = {
            "name": name,
            "image": image_base64
        }
        
        response = requests.post(f"{BASE_URL}/add_user", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Hata: {e}")
        return False

def test_recognize_face(image_path):
    """Yüz tanıma testi"""
    print(f"\n=== Yüz Tanıma Testi ===")
    try:
        # Resmi base64'e çevir
        image_base64 = image_to_base64(image_path)
        if not image_base64:
            return False
        
        # API'ye gönder
        data = {
            "image": image_base64
        }
        
        response = requests.post(f"{BASE_URL}/recognize", json=data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Hata: {e}")
        return False

def test_delete_user(user_id):
    """Kullanıcı silme testi"""
    print(f"\n=== Kullanıcı Silme Testi: ID {user_id} ===")
    try:
        response = requests.delete(f"{BASE_URL}/delete_user/{user_id}")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Hata: {e}")
        return False

def find_test_images():
    """Test için resim dosyalarını bul"""
    test_images = []
    
    # known_faces klasöründen resim ara
    if os.path.exists("known_faces"):
        for person_folder in os.listdir("known_faces"):
            person_path = os.path.join("known_faces", person_folder)
            if os.path.isdir(person_path):
                # İlk resmi al
                for file in os.listdir(person_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        test_images.append({
                            'name': person_folder,
                            'path': os.path.join(person_path, file)
                        })
                        break  # Sadece ilk resmi al
    
    return test_images

def run_comprehensive_test():
    """Kapsamlı test çalıştır"""
    print("🚀 Yüz Tanıma API Testi Başlıyor...\n")
    
    # 1. Sağlık kontrolü
    if not test_health_check():
        print("❌ API çalışmıyor!")
        return
    
    # 2. Kullanıcı listesi (başlangıçta boş olmalı)
    test_get_users()
    
    # 3. Test resimlerini bul
    test_images = find_test_images()
    if not test_images:
        print("❌ Test resmi bulunamadı!")
        return
    
    print(f"\n📸 {len(test_images)} test resmi bulundu")
    
    # 4. İlk 3 resmi kullanıcı olarak ekle
    added_users = []
    for i, img_info in enumerate(test_images[:3]):
        if test_add_user(img_info['name'], img_info['path']):
            added_users.append(i + 1)  # user_id'ler 1'den başlar
    
    # 5. Kullanıcı listesini kontrol et
    test_get_users()
    
    # 6. Eklenen kullanıcıları tanımayı dene
    for i, img_info in enumerate(test_images[:3]):
        test_recognize_face(img_info['path'])
    
    # 7. Tanınmayan bir resim test et
    if len(test_images) > 3:
        print(f"\n🔍 Tanınmayan resim testi: {test_images[3]['name']}")
        test_recognize_face(test_images[3]['path'])
    
    # 8. Kullanıcı silme testi
    if added_users:
        test_delete_user(added_users[0])
        test_get_users()
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    run_comprehensive_test()