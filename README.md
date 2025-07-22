# Face Recognition API & Flutter Client

## Proje Açıklaması
Bu proje, Python (Flask) ile yazılmış bir yüz tanıma API'si ve Flutter ile yazılmış bir istemci uygulamasından oluşur. API, MobilFaceNet (TFLite) modelini kullanarak yüz embedding'leri çıkarır ve OpenCV ile yüz algılama yapar. Kullanıcılar eklenebilir, silinebilir ve gerçek zamanlı olarak tanınabilir.

---

## Özellikler
- Yüz algılama: OpenCV Haar Cascade
- Yüz embedding: MobileFaceNet (TFLite)
- Kullanıcı ekleme, silme, listeleme
- Gerçek zamanlı yüz tanıma
- Flutter istemci ile mobil/web arayüz

---

## Kurulum

### 1. Python API

#### Gereksinimler
- Python 3.8+
- pip
- TensorFlow, OpenCV, Flask, Pillow, numpy

#### Kurulum Adımları

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# MobileFaceNet TFLite modelini workspace ana dizinine koyun (örn: mobilefacenet (1).tflite)

# API'yi başlat
python app.py
```

API çalışınca şu adreslerden ulaşabilirsin:
- http://127.0.0.1:5000/health

### 2. Flutter Uygulaması

#### Gereksinimler
- Flutter 3.10+
- Android Studio veya VSCode

#### Kurulum Adımları

```bash
# Gerekli paketleri yükle
flutter pub get

# Android/iOS için kamera izinlerini ekle (AndroidManifest.xml, Info.plist)
# Web için camera_web paketini ekle

# Uygulamayı başlat
flutter run -d chrome   # Web için
flutter run -d emulator-5554   # Android emülatör için
```

---

## API Endpointleri

### Sağlık Kontrolü
```
GET /health
```
Yanıt:
```json
{"status": "healthy", "message": "API çalışıyor"}
```

### Kullanıcı Ekleme
```
POST /add_user
```
Body (JSON):
```
{
  "name": "Ad Soyad",
  "id_no": "12345678901",
  "birth_date": "2000-01-01",
  "images": ["base64string1", "base64string2", ...]
}
```

### Yüz Tanıma
```
POST /recognize
```
Body (JSON):
```
{
  "image": "base64string"
}
```

### Kullanıcıları Listele
```
GET /users
```

### Kullanıcı Silme
```
DELETE /delete_user/<user_id>
```

---

## Flutter Entegrasyonu
- `lib/services/face_api_services.dart` dosyasında API adresini kendi IP adresinize göre ayarlayın.
- Mobilde ve webde kamera izinlerini doğru verdiğinizden emin olun.
- Web için: `flutter run -d chrome` ile başlatın ve tarayıcıdan kameraya izin verin.

---

## Önemli Notlar
- API'yi her zaman aynı dizinden başlatın, aksi halde kullanıcı verileri kaybolabilir.
- `users_db.json` ve `known_faces/` klasörü silinmemelidir.
- MobileFaceNet model dosyasını ana dizinde bulundurmalısınız.
- Android emülatöründe kamera için AVD ayarlarından "Webcam0" seçili olmalı.
- Geliştirme sunucusu (Flask) prod ortamı için uygun değildir, dağıtıma çıkarken WSGI sunucusu (gunicorn, uWSGI) kullanın.

---

## Lisans
MIT
