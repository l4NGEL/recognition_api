# Face Recognition API & Flutter Client

## Proje Açıklaması
Bu proje, Python (Flask) ile yazılmış modüler bir yüz tanıma API'si ve Flutter ile yazılmış bir istemci uygulamasından oluşur. API, MobilFaceNet (TFLite) modelini kullanarak yüz embedding'leri çıkarır ve OpenCV ile yüz algılama yapar. Kullanıcılar eklenebilir, silinebilir ve gerçek zamanlı olarak tanınabilir.

---

## Proje Yapısı

### Modüler Mimari
Proje, sürdürülebilirlik ve okunabilirlik için modüler bir yapıya sahiptir:

```
recognition_api/
├── app.py                 # Ana Flask uygulaması (giriş noktası)
├── config.py             # Konfigürasyon sabitleri
├── utils.py              # Genel yardımcı fonksiyonlar
├── face_detection.py     # Yüz algılama ve hizalama
├── face_recognition.py   # Yüz tanıma API sınıfı
├── logging.py            # Loglama fonksiyonları
├── routes.py             # Flask route'ları
├── requirements.txt      # Python bağımlılıkları
├── users_db.json         # Kullanıcı veritabanı
├── known_faces/          # Bilinen yüzler klasörü
└── README.md            # Bu dosya
```

### Modül Açıklamaları

- **`app.py`**: Flask uygulamasının ana giriş noktası. CORS yapılandırması ve route'ların başlatılması.
- **`config.py`**: Proje genelinde kullanılan sabitler (dosya yolları, model yolları vb.).
- **`utils.py`**: Görüntü artırma, JSON düzeltme, otomatik döndürme gibi genel yardımcı fonksiyonlar.
- **`face_detection.py`**: MTCNN ve Haar Cascade kullanarak yüz algılama, hizalama ve ön işleme.
- **`face_recognition.py`**: FaceNet modeli ile yüz embedding'leri çıkarma ve tanıma işlemleri.
- **`logging.py`**: Tanıma girişimleri ve eşik değişiklikleri için loglama fonksiyonları.
- **`routes.py`**: Tüm Flask API endpoint'leri ve işleyicileri.

---

## Özellikler
- **Modüler Mimari**: Sürdürülebilir ve genişletilebilir kod yapısı
- **Yüz Algılama**: MTCNN ve OpenCV Haar Cascade (fallback)
- **Yüz Embedding**: FaceNet (TFLite) modeli
- **Kullanıcı Yönetimi**: Ekleme, silme, listeleme
- **Gerçek Zamanlı Tanıma**: Adaptif eşik değerleri ile
- **Görüntü Artırma**: Otomatik veri artırma teknikleri
- **Loglama**: Detaylı tanıma ve eşik logları
- **Flutter İstemci**: Mobil/web arayüzü

---

## Kurulum

### 1. Python API

#### Gereksinimler
- Python 3.8+
- pip
- TensorFlow, OpenCV, Flask, Pillow, numpy, MTCNN

#### Kurulum Adımları

```bash
# Gerekli paketleri yükle
pip install -r requirements.txt

# MobileFaceNet TFLite modelini workspace ana dizinine koyun (örn: facenet.tflite)

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
```json
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
```json
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

### Tanıma Logları
```
GET /recognition_logs
```

### Eşik Logları
```
GET /threshold_logs
```

### Görüntü Artırma İstatistikleri
```
GET /augmentation_stats
```

---

## Teknik Detaylar

### Yüz Algılama
- **MTCNN**: Birincil yüz algılama yöntemi (landmark'lar ile)
- **Haar Cascade**: MTCNN başarısız olduğunda fallback yöntemi
- **Otomatik Döndürme**: EXIF verilerine göre görüntü yönlendirmesi

### Yüz Tanıma
- **MobilFaceNet**: TFLite modeli ile embedding çıkarma
- **Euclidean Distance**: Embedding karşılaştırma yöntemi
- **Adaptif Eşik**: ROC eğrisi ve yüzdelik tabanlı eşik optimizasyonu

### Görüntü İşleme
- **Veri Artırma**: Parlaklık/kontrast, gürültü, bulanıklık, keskinleştirme
- **Geometrik Dönüşümler**: Döndürme, ölçekleme, çevirme
- **Aydınlatma Simülasyonu**: Farklı aydınlatma koşulları

---

## Flutter Entegrasyonu
- `lib/services/face_api_services.dart` dosyasında API adresini kendi IP adresinize göre ayarlayın.
- Mobilde ve webde kamera izinlerini doğru verdiğinizden emin olun.
- Web için: `flutter run -d chrome` ile başlatın ve tarayıcıdan kameraya izin verin.

---

## Geliştirme

### Yeni Özellik Ekleme
1. İlgili modülde fonksiyon/class ekleyin
2. `routes.py`'de yeni endpoint tanımlayın
3. Gerekirse `config.py`'de yeni sabitler ekleyin
4. Test edin ve dokümantasyonu güncelleyin

### Modül Bağımlılıkları
```
app.py
├── routes.py
    ├── face_recognition.py
    │   ├── face_detection.py
    │   ├── utils.py
    │   └── logging.py
    └── config.py
```

---

## Önemli Notlar
- API'yi her zaman aynı dizinden başlatın, aksi halde kullanıcı verileri kaybolabilir.
- `users_db.json` ve `known_faces/` klasörü silinmemelidir.
- FaceNet model dosyasını ana dizinde bulundurmalısınız.
- Android emülatöründe kamera için AVD ayarlarından "Webcam0" seçili olmalı.
- Geliştirme sunucusu (Flask) prod ortamı için uygun değildir, dağıtıma çıkarken WSGI sunucusu (gunicorn, uWSGI) kullanın.
- Modüler yapı sayesinde kod bakımı ve genişletilmesi kolaylaştırılmıştır.

---
