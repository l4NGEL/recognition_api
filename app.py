from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import os
import json
from datetime import datetime
import base64
from PIL import Image, ImageOps
import io
import pickle
import tensorflow as tf
import shutil
from mtcnn import MTCNN

app = Flask(__name__)
CORS(app)

USERS_DB_FILE = 'users_db.json'
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'known_faces')
RECOGNITION_LOG_FILE = 'recognition_logs.json'
MOBILFACENET_MODEL_PATH = 'mobilefacenet.tflite'

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

def auto_rotate_image(image):
    """
    Görüntüyü EXIF verilerine göre otomatik olarak döndürür
    """
    try:
        # EXIF verilerini kontrol et
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)  # EXIF orientation tag
                if orientation is not None:
                    # Orientation değerine göre döndür
                    if orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
    except Exception as e:
        print(f"EXIF rotasyon hatası: {e}")

    return image

def fix_corrupted_json():
    """
    Bozuk JSON dosyasını düzeltir
    """
    try:
        if not os.path.exists(USERS_DB_FILE):
            print("users_db.json dosyası bulunamadı, yeni dosya oluşturuluyor...")
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return True

        # Dosyayı oku ve temizle
        with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Dosya boyutu: {len(content)} karakter")

        # Sadece ilk geçerli JSON array'i al
        try:
            first_bracket = content.index('[')
            last_bracket = content.rindex(']')
            valid_json = content[first_bracket:last_bracket+1]

            # JSON'u parse et
            users = json.loads(valid_json)
            print(f"Geçerli kullanıcı sayısı: {len(users)}")

            # Temiz JSON'u tekrar kaydet
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)

            print("JSON dosyası başarıyla temizlendi")
            return True

        except ValueError as e:
            print(f"JSON parse hatası: {e}")
            # Dosya tamamen bozuksa, yeni dosya oluştur
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("Yeni JSON dosyası oluşturuldu")
            return True

    except Exception as e:
        print(f"JSON düzeltme hatası: {e}")
        return False

class FaceRecognitionAPI:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.mobilfacenet_interpreter = self.load_mobilfacenet_model(MOBILFACENET_MODEL_PATH)
        self.input_details = self.mobilfacenet_interpreter.get_input_details()
        self.output_details = self.mobilfacenet_interpreter.get_output_details()

        # MTCNN yüz tespit edici - Haar Cascade yerine
        self.mtcnn_detector = MTCNN()

        # Tanınan kişileri takip etmek için set
        self.recognized_persons = set()
        # Gerçek zamanlı tanıma kayıtları
        self.realtime_recognition_logs = []
        # Performans optimizasyonu için cache
        self.known_embeddings_array = None
        self.users_data_cache = None
        self.last_cache_update = None
        self.load_known_faces()

    def update_cache(self):
        """
        Performans için cache'i günceller
        """
        try:
            # Embeddings array'ini önceden hesapla
            if len(self.known_face_encodings) > 0:
                self.known_embeddings_array = np.array(self.known_face_encodings)

            # Users data cache'ini güncelle
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                self.users_data_cache = json.load(f)

            self.last_cache_update = datetime.now()
        except Exception as e:
            print(f"Cache güncelleme hatası: {e}")

    def load_mobilfacenet_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def get_embedding(self, face_img):
        try:
            # Yüz boyutunu kontrol et
            height, width = face_img.shape[:2]
            print(f"Yüz boyutu: {width}x{height}")

            # Minimum boyut kontrolü
            if width < 80 or height < 80:
                print(f"Yüz çok küçük: {width}x{height}, minimum 80x80 gerekli")
                return None

            # Model için 112x112'ye yeniden boyutlandır
            img = cv2.resize(face_img, (112, 112), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            self.mobilfacenet_interpreter.set_tensor(self.input_details[0]['index'], img)
            self.mobilfacenet_interpreter.invoke()
            embedding = self.mobilfacenet_interpreter.get_tensor(self.output_details[0]['index'])
            return embedding.flatten()
        except Exception as e:
            print(f"Embedding oluşturma hatası: {e}")
            return None

    def compare_embeddings(self, emb1, emb2, threshold=0.502):
        """
        İki embedding arasındaki mesafeyi hesaplar
        Orta seviye threshold kullanarak dengeli eşleşme sağlar
        """
        dist = np.linalg.norm(emb1 - emb2)
        return dist < threshold

    def compare_embeddings_batch(self, query_embedding, known_embeddings, threshold=0.502):
        """
        Tek seferde tüm bilinen yüzlerle karşılaştırma yapar (daha hızlı)
        Dengeli threshold ile eşleşme kontrolü
        """
        if len(known_embeddings) == 0:
            return -1, float('inf')

        # Vektörize edilmiş mesafe hesaplama
        query_embedding = np.array(query_embedding).reshape(1, -1)
        known_embeddings_array = np.array(known_embeddings)

        # Euclidean mesafe hesaplama
        distances = np.linalg.norm(known_embeddings_array - query_embedding, axis=1)

        # En yakın mesafeyi bul
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]

        # Dengeli eşleşme kontrolü
        if min_distance < threshold:
            return min_distance_idx, min_distance
        else:
            return -1, min_distance

    def detect_faces(self, image_array):
        """
        MTCNN kullanarak yüz tespiti yapar
        """
        try:
            # BGR'den RGB'ye çevir (MTCNN RGB bekler)
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # MTCNN ile yüz tespiti
            faces = self.mtcnn_detector.detect_faces(rgb_image)

            # MTCNN formatını OpenCV formatına çevir
            opencv_faces = []
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']

                # Güven skoru kontrolü
                if confidence > 0.9:  # Yüksek güven skoru
                    opencv_faces.append((x, y, w, h))

            return np.array(opencv_faces)
        except Exception as e:
            print(f"MTCNN yüz tespit hatası: {e}")
            return np.array([])

    def detect_faces_with_rotation(self, image_array):
        """
        MTCNN ile görüntüyü farklı açılarda döndürerek yüz tespit etmeye çalışır
        """
        # Orijinal görüntüde yüz tespit etmeyi dene
        faces = self.detect_faces(image_array)
        if len(faces) > 0:
            return faces, 0  # 0 derece (orijinal)

        # 90 derece döndür
        rotated_90 = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
        faces = self.detect_faces(rotated_90)
        if len(faces) > 0:
            return faces, 90

        # 180 derece döndür
        rotated_180 = cv2.rotate(image_array, cv2.ROTATE_180)
        faces = self.detect_faces(rotated_180)
        if len(faces) > 0:
            return faces, 180

        # 270 derece döndür
        rotated_270 = cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces = self.detect_faces(rotated_270)
        if len(faces) > 0:
            return faces, 270

        return [], 0  # Hiçbir açıda yüz bulunamadı

    def get_face_with_rotation(self, image_array, faces, rotation_angle):
        """
        MTCNN ile döndürülmüş görüntüden yüzü çıkarır ve hizalar
        """
        if len(faces) == 0:
            return None

        # İlk yüzü al
        (x, y, w, h) = faces[0]

        # Döndürülmüş görüntüden yüzü kes
        face_img = image_array[y:y+h, x:x+w]

        # Yüzü orijinal yöne döndür
        if rotation_angle == 90:
            face_img = cv2.rotate(face_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_angle == 180:
            face_img = cv2.rotate(face_img, cv2.ROTATE_180)
        elif rotation_angle == 270:
            face_img = cv2.rotate(face_img, cv2.ROTATE_90_CLOCKWISE)

        return face_img

    def detect_faces_with_landmarks(self, image_array):
        """
        MTCNN ile yüz tespiti ve landmark'ları alır
        """
        try:
            # BGR'den RGB'ye çevir (MTCNN RGB bekler)
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

            # MTCNN ile yüz tespiti ve landmark'lar
            faces = self.mtcnn_detector.detect_faces(rgb_image)

            return faces
        except Exception as e:
            print(f"MTCNN landmark tespit hatası: {e}")
            return []

    def align_face(self, image_array, landmarks):
        """
        Yüz landmark'larını kullanarak yüzü hizalar - siyah kenarlar olmadan
        """
        try:
            # MTCNN landmark formatını kontrol et
            if isinstance(landmarks, dict):
                # Göz noktalarını al
                left_eye = landmarks.get('left_eye', landmarks.get('leftEye'))
                right_eye = landmarks.get('right_eye', landmarks.get('rightEye'))

                if left_eye is None or right_eye is None:
                    print("Göz landmark'ları bulunamadı, orijinal görüntü döndürülüyor")
                    return image_array

                # Gözler arası açıyı hesapla
                eye_angle = np.degrees(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))

                # Çok küçük açılar için hizalama yapma (5 dereceden az)
                if abs(eye_angle) < 5:
                    print("Açı çok küçük, hizalama yapılmıyor")
                    return image_array

                # Gözler arası merkez nokta
                eye_center = (
                    int((left_eye[0] + right_eye[0]) / 2),
                    int((left_eye[1] + right_eye[1]) / 2)
                )

                # Görüntüyü genişlet (siyah kenarları önlemek için)
                height, width = image_array.shape[:2]
                diagonal = int(np.sqrt(width**2 + height**2))

                # Yeni boyutlar
                new_width = diagonal
                new_height = diagonal

                # Yeni görüntü oluştur
                new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

                # Eski görüntüyü yeni görüntünün merkezine yerleştir
                x_offset = (new_width - width) // 2
                y_offset = (new_height - height) // 2
                new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image_array

                # Yeni merkez nokta
                new_center = (x_offset + eye_center[0], y_offset + eye_center[1])

                # Rotasyon matrisi oluştur
                rotation_matrix = cv2.getRotationMatrix2D(new_center, eye_angle, 1.0)

                # Görüntüyü döndür
                aligned_face = cv2.warpAffine(new_image, rotation_matrix, (new_width, new_height))

                # Siyah kenarları kaldır
                aligned_face = self.remove_black_borders(aligned_face)

                return aligned_face
            else:
                print("Landmark formatı tanınmadı, orijinal görüntü döndürülüyor")
                return image_array

        except Exception as e:
            print(f"Yüz hizalama hatası: {e}")
            return image_array

    def remove_black_borders(self, image):
        """
        Görüntüdeki siyah kenarları kaldırır - daha agresif yaklaşım
        """
        try:
            # Gri tonlamaya çevir
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Çok düşük eşik değeri kullan (çok agresif)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            # Morfolojik işlemler uygula
            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            # Konturları bul
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # En büyük konturu al
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Minimum boyut kontrolü
                min_size = 30
                if w < min_size or h < min_size:
                    return image

                # Margin ekle (çok az margin)
                margin = 2
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2 * margin)
                h = min(image.shape[0] - y, h + 2 * margin)

                # Kırp
                cropped = image[y:y+h, x:x+w]

                # Kırpılan görüntünün boyutunu kontrol et
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    return cropped

            return image
        except Exception as e:
            print(f"Siyah kenar kaldırma hatası: {e}")
            return image

    def process_image_with_rotation(self, image_data_base64):
        """
        Base64 görüntüyü alır, rotasyonunu düzeltir ve numpy array'e çevirir
        """
        try:
            # Base64'ten PIL Image'e çevir
            image_data = base64.b64decode(image_data_base64.split(',')[1] if ',' in image_data_base64 else image_data_base64)
            image = Image.open(io.BytesIO(image_data))

            # Görüntü modunu kontrol et ve düzelt
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Otomatik rotasyon uygula
            image = auto_rotate_image(image)

            # PIL Image'i numpy array'e çevir
            image_array = np.array(image)

            # RGB'den BGR'ye çevir (OpenCV için)
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

            # Görüntü boyutunu kontrol et ve logla
            height, width = image_array.shape[:2]
            print(f"İşlenen görüntü boyutu: {width}x{height}")

            return image_array
        except Exception as e:
            print(f"Görüntü işleme hatası: {e}")
            return None

    def load_known_faces(self):
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_ids.clear()

        if os.path.exists(USERS_DB_FILE):
            try:
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                print(f"load_known_faces: {len(users_data)} kullanıcı yüklendi")
            except json.JSONDecodeError as e:
                print(f"load_known_faces JSON hatası: {e}")
                print("JSON dosyası düzeltiliyor...")
                if fix_corrupted_json():
                    try:
                        with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                            users_data = json.load(f)
                        print(f"Düzeltilmiş dosyadan {len(users_data)} kullanıcı yüklendi")
                    except Exception as e2:
                        print(f"Düzeltilmiş dosya okuma hatası: {e2}")
                        users_data = []
                else:
                    print("JSON düzeltme başarısız")
                    users_data = []
            except Exception as e:
                print(f"load_known_faces genel hatası: {e}")
                users_data = []
        else:
            print("users_db.json dosyası bulunamadı")
            users_data = []

        for user in users_data:
            user_dir = os.path.join(KNOWN_FACES_DIR, user['id_no'])
            encodings_file = os.path.join(user_dir, 'encodings.pkl')
            if os.path.exists(encodings_file):
                try:
                    with open(encodings_file, 'rb') as ef:
                        encodings = pickle.load(ef)
                        for enc in encodings:
                            self.known_face_encodings.append(enc)
                            self.known_face_names.append(user['name'])
                            self.known_face_ids.append(user['id_no'])
                    print(f"Kullanıcı {user['name']} yüklendi: {len(encodings)} encoding")
                except Exception as e:
                    print(f"Encoding yükleme hatası ({user['name']}): {e}")

        print(f"Toplam yüklenen encoding: {len(self.known_face_encodings)}")

        # Cache'i güncelle
        self.update_cache()

    def add_user(self, name, images_base64, id_no=None, birth_date=None):
        try:
            print(f"Add user başlatıldı: {name}, {id_no}")

            # id_no 11 hane kontrolü
            if not id_no or not str(id_no).isdigit() or len(str(id_no)) != 11:
                return False, "Kimlik numarası 11 haneli olmalıdır."

            # Mevcut kullanıcı kontrolü
            if os.path.exists(USERS_DB_FILE):
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                if any(user['id_no'] == id_no for user in users_data):
                    return False, "Bu kimlik numarasına sahip kullanıcı zaten kayıtlı."
            else:
                users_data = []

            user_id = str(id_no)
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)

            # Kullanıcı klasörünü oluştur
            try:
                os.makedirs(user_dir, exist_ok=True)
                print(f"Kullanıcı klasörü oluşturuldu: {user_dir}")
            except Exception as e:
                print(f"Klasör oluşturma hatası: {e}")
                return False, f"Klasör oluşturulamadı: {e}"

            encodings = []
            saved_photos = 0

            for idx, img_b64 in enumerate(images_base64):
                print(f"Resim {idx+1} işleniyor...")

                # Rotasyon düzeltmesi ile görüntüyü işle
                image_array = self.process_image_with_rotation(img_b64)
                if image_array is None:
                    print(f"Resim {idx+1} işlenemedi")
                    continue

                # MTCNN ile yüz tespiti ve landmark'lar - farklı rotasyonlarda dene
                faces_with_landmarks = self.detect_faces_with_landmarks(image_array)
                rotation_angle = 0

                # Eğer orijinal görüntüde yüz bulunamazsa, farklı rotasyonlarda dene
                if len(faces_with_landmarks) == 0:
                    print(f"Resim {idx+1} için orijinal görüntüde yüz bulunamadı, farklı rotasyonlarda deneniyor...")

                    # 90 derece döndür
                    rotated_90 = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
                    faces_with_landmarks = self.detect_faces_with_landmarks(rotated_90)
                    if len(faces_with_landmarks) > 0:
                        rotation_angle = 90
                        image_array = rotated_90
                        print(f"Resim {idx+1} için 90° rotasyonda yüz bulundu")

                    # 180 derece döndür
                    if len(faces_with_landmarks) == 0:
                        rotated_180 = cv2.rotate(image_array, cv2.ROTATE_180)
                        faces_with_landmarks = self.detect_faces_with_landmarks(rotated_180)
                        if len(faces_with_landmarks) > 0:
                            rotation_angle = 180
                            image_array = rotated_180
                            print(f"Resim {idx+1} için 180° rotasyonda yüz bulundu")

                    # 270 derece döndür
                    if len(faces_with_landmarks) == 0:
                        rotated_270 = cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        faces_with_landmarks = self.detect_faces_with_landmarks(rotated_270)
                        if len(faces_with_landmarks) > 0:
                            rotation_angle = 270
                            image_array = rotated_270
                            print(f"Resim {idx+1} için 270° rotasyonda yüz bulundu")

                if len(faces_with_landmarks) == 0:
                    print(f"Resim {idx+1} için yüz tespit edilemedi")
                    continue

                # İlk yüzü al
                face_data = faces_with_landmarks[0]
                x, y, w, h = face_data['box']
                landmarks = face_data['keypoints']

                # Yüzü kes (daha az margin - daha kesin yüz alanı)
                margin = int(min(w, h) * 0.15)  # %15 margin (daha az margin)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image_array.shape[1], x + w + margin)
                y2 = min(image_array.shape[0], y + h + margin)

                face_img = image_array[y1:y2, x1:x2]

                # Siyah kenarları kaldır
                face_img = self.remove_black_borders(face_img)

                # Landmark'ları yeni koordinatlara göre ayarla
                adjusted_landmarks = {}
                for key, point in landmarks.items():
                    adjusted_landmarks[key] = (point[0] - x1, point[1] - y1)

                # Yüz hizalama uygula
                aligned_face = self.align_face(face_img, adjusted_landmarks)

                # Siyah kenarları kaldır
                aligned_face = self.remove_black_borders(aligned_face)

                # Ana embedding'i oluştur
                main_embedding = self.get_embedding(aligned_face)
                if main_embedding is None:
                    print(f"Resim {idx+1} için embedding oluşturulamadı - yüz çözünürlüğü yetersiz")
                    continue

                encodings.append(main_embedding)

                # Farklı rotasyonlarda fotoğraflar oluştur ve kaydet - tüm rotasyonlar
                rotations = [0, 90, 180, 270]  # 90°, 180°, 270° rotasyonları dahil
                for rot_idx, rotation in enumerate(rotations):
                    try:
                        # Rotasyon uygula
                        if rotation == 0:
                            rotated_face = aligned_face
                        elif rotation == 90:
                            rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_90_CLOCKWISE)
                        elif rotation == 180:
                            rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_180)
                        elif rotation == 270:
                            rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_90_COUNTERCLOCKWISE)

                        # Rotasyonlu embedding oluştur
                        rotated_embedding = self.get_embedding(rotated_face)
                        if rotated_embedding is not None:
                            encodings.append(rotated_embedding)

                        # Fotoğrafı dosyaya kaydet
                        face_rgb = cv2.cvtColor(rotated_face, cv2.COLOR_BGR2RGB)
                        face_pil = Image.fromarray(face_rgb)
                        photo_path = os.path.join(user_dir, f'{idx+1}_rot{rotation}.jpg')
                        face_pil.save(photo_path)
                        saved_photos += 1
                        print(f"Rotasyon {rotation}° fotoğrafı kaydedildi: {photo_path}")

                    except Exception as e:
                        print(f"Rotasyon {rotation}° fotoğraf kaydetme hatası: {e}")
                        continue

            if not encodings:
                return False, "Hiçbir resimde yüz tespit edilemedi"

            # Embeddings'i kaydet
            try:
                encodings_path = os.path.join(user_dir, 'encodings.pkl')
                with open(encodings_path, 'wb') as ef:
                    pickle.dump(encodings, ef)
                print(f"Embeddings kaydedildi: {encodings_path}")
            except Exception as e:
                print(f"Embeddings kaydetme hatası: {e}")
                return False, f"Embeddings kaydedilemedi: {e}"

            # JSON'a kullanıcı bilgilerini kaydet
            try:
                self.save_users_db(user_id, name, id_no, birth_date)
                print(f"Kullanıcı JSON'a kaydedildi: {user_id}")
            except Exception as e:
                print(f"JSON kaydetme hatası: {e}")
                return False, f"Kullanıcı bilgileri kaydedilemedi: {e}"

            # Memory'ye ekle
            for enc in encodings:
                self.known_face_encodings.append(enc)
                self.known_face_names.append(name)
                self.known_face_ids.append(user_id)

            # Cache'i güncelle
            try:
                self.update_cache()
                print("Cache güncellendi")
            except Exception as e:
                print(f"Cache güncelleme hatası: {e}")

            log_recognition(user_id=user_id, name=name, result='success', image_b64=None, action='kayıt')
            print(f"Kullanıcı başarıyla eklendi: {name}, {saved_photos} fotoğraf kaydedildi")
            return True, f"Kullanıcı {name} başarıyla eklendi ({saved_photos} fotoğraf)"

        except Exception as e:
            print(f"Add user genel hatası: {e}")
            return False, f"Hata: {str(e)}"

    def save_users_db(self, user_id, name, id_no=None, birth_date=None):
        try:
            print(f"save_users_db başlatıldı: {user_id}, {name}")

            users_data = []
            if os.path.exists(USERS_DB_FILE):
                try:
                    with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                        users_data = json.load(f)
                    print(f"Mevcut kullanıcı sayısı: {len(users_data)}")
                except json.JSONDecodeError as e:
                    print(f"JSON okuma hatası (extra data): {e}")
                    print("JSON dosyası düzeltiliyor...")
                    if fix_corrupted_json():
                        # Düzeltilmiş dosyayı tekrar oku
                        try:
                            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                                users_data = json.load(f)
                            print(f"Düzeltilmiş dosyadan okunan kullanıcı sayısı: {len(users_data)}")
                        except Exception as e2:
                            print(f"Düzeltilmiş dosya okuma hatası: {e2}")
                            users_data = []
                    else:
                        print("JSON düzeltme başarısız, yeni liste oluşturuluyor")
                        users_data = []
                except Exception as e:
                    print(f"JSON okuma hatası: {e}")
                    users_data = []

            # Yeni kullanıcı bilgilerini ekle
            new_user = {
                'id': user_id,
                'name': name,
                'id_no': id_no,
                'birth_date': birth_date,
                'created_at': datetime.now().isoformat()
            }
            users_data.append(new_user)

            print(f"Yeni kullanıcı eklendi: {new_user}")
            print(f"Toplam kullanıcı sayısı: {len(users_data)}")

            # JSON dosyasına kaydet
            try:
                with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                    json.dump(users_data, f, ensure_ascii=False, indent=2)
                print(f"JSON dosyasına başarıyla kaydedildi: {USERS_DB_FILE}")
            except Exception as e:
                print(f"JSON yazma hatası: {e}")
                raise e

        except Exception as e:
            print(f"save_users_db genel hatası: {e}")
            raise e

    def recognize_face(self, face_image_base64):
        try:
            # Rotasyon düzeltmesi ile görüntüyü işle
            image_array = self.process_image_with_rotation(face_image_base64)
            if image_array is None:
                return False, "Görüntü işlenemedi"

            # MTCNN ile yüz tespiti ve landmark'lar - farklı rotasyonlarda dene
            faces_with_landmarks = self.detect_faces_with_landmarks(image_array)
            rotation_angle = 0

            # Eğer orijinal görüntüde yüz bulunamazsa, farklı rotasyonlarda dene
            if len(faces_with_landmarks) == 0:
                print("Orijinal görüntüde yüz bulunamadı, farklı rotasyonlarda deneniyor...")

                # 90 derece döndür
                rotated_90 = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
                faces_with_landmarks = self.detect_faces_with_landmarks(rotated_90)
                if len(faces_with_landmarks) > 0:
                    rotation_angle = 90
                    image_array = rotated_90
                    print("90° rotasyonda yüz bulundu")

                # 180 derece döndür
                if len(faces_with_landmarks) == 0:
                    rotated_180 = cv2.rotate(image_array, cv2.ROTATE_180)
                    faces_with_landmarks = self.detect_faces_with_landmarks(rotated_180)
                    if len(faces_with_landmarks) > 0:
                        rotation_angle = 180
                        image_array = rotated_180
                        print("180° rotasyonda yüz bulundu")

                # 270 derece döndür
                if len(faces_with_landmarks) == 0:
                    rotated_270 = cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    faces_with_landmarks = self.detect_faces_with_landmarks(rotated_270)
                    if len(faces_with_landmarks) > 0:
                        rotation_angle = 270
                        image_array = rotated_270
                        print("270° rotasyonda yüz bulundu")

            if len(faces_with_landmarks) == 0:
                return False, "Yüz tespit edilemedi"

            # Cache'den users data'yı al
            if self.users_data_cache is None:
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    self.users_data_cache = json.load(f)

            # İlk yüzü al
            face_data = faces_with_landmarks[0]
            x, y, w, h = face_data['box']
            landmarks = face_data['keypoints']

            # Yüzü kes (daha az margin - daha kesin yüz alanı)
            margin = int(min(w, h) * 0.15)  # %15 margin (daha az margin)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_array.shape[1], x + w + margin)
            y2 = min(image_array.shape[0], y + h + margin)

            face_img = image_array[y1:y2, x1:x2]

            # Siyah kenarları kaldır
            face_img = self.remove_black_borders(face_img)

            # Landmark'ları yeni koordinatlara göre ayarla
            adjusted_landmarks = {}
            for key, point in landmarks.items():
                adjusted_landmarks[key] = (point[0] - x1, point[1] - y1)

            # Yüz hizalama uygula
            aligned_face = self.align_face(face_img, adjusted_landmarks)

            # Siyah kenarları kaldır
            aligned_face = self.remove_black_borders(aligned_face)

            # Ana embedding'i oluştur
            main_embedding = self.get_embedding(aligned_face)
            if main_embedding is None:
                return False, "Yüz çözünürlüğü yetersiz veya embedding oluşturulamadı"

            # Farklı rotasyonlarda da deneme yap - daha az rotasyon
            embeddings_to_test = [main_embedding]

            # 90, 180, 270 derece rotasyonları dene
            rotations = [90, 180, 270]
            for rotation in rotations:
                try:
                    if rotation == 90:
                        rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_90_CLOCKWISE)
                    elif rotation == 180:
                        rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_180)
                    elif rotation == 270:
                        rotated_face = cv2.rotate(aligned_face, cv2.ROTATE_90_COUNTERCLOCKWISE)

                    rotated_embedding = self.get_embedding(rotated_face)
                    if rotated_embedding is not None:
                        embeddings_to_test.append(rotated_embedding)
                except Exception as e:
                    print(f"Rotasyon {rotation}° embedding hatası: {e}")
                    continue

            # Tüm rotasyonlarda tanıma dene
            best_match_idx = -1
            min_distance = float('inf')
            match_confidence = 0.0

            for test_embedding in embeddings_to_test:
                if self.known_embeddings_array is not None and len(self.known_embeddings_array) > 0:
                    current_match_idx, current_distance = self.compare_embeddings_batch(
                        test_embedding, self.known_embeddings_array, threshold=0.502
                    )

                    if current_match_idx != -1 and current_distance < min_distance:
                        best_match_idx = current_match_idx
                        min_distance = current_distance
                        # Güven skorunu hesapla (mesafe ne kadar düşükse o kadar güvenilir)
                        match_confidence = max(0, 1 - (current_distance / 0.502))

            # En iyi eşleşmeyi kontrol et - dengeli eşleşme kontrolü
            if best_match_idx != -1 and min_distance < 0.502:  # Dengeli eşleşme için
                user_id = self.known_face_ids[best_match_idx]
                user_info = next((u for u in self.users_data_cache if u['id'] == user_id), None)

                if user_info is None:
                    return False, "Kullanıcı bilgisi bulunamadı"

                # Eğer bu kişi zaten tanınmışsa, tekrar tanıma
                if user_id in self.recognized_persons:
                    return False, user_info

                # Tanınan kişiyi set'e ekle
                self.recognized_persons.add(user_id)

                # Tanınan kişinin fotoğrafını kaydet
                self.save_recognition_photo(user_id, image_array, aligned_face)

                # Gerçek zamanlı tanıma kaydı ekle (sadece bir kez ekle)
                if not any(log['user_id'] == user_id for log in self.realtime_recognition_logs):
                    recognition_record = {
                        'user_id': user_id,
                        'name': user_info['name'],
                        'id_no': user_info.get('id_no'),
                        'birth_date': user_info.get('birth_date'),
                        'timestamp': datetime.now().isoformat(),
                        'image_base64': face_image_base64,
                        'confidence': match_confidence,
                        'distance': min_distance
                    }
                    self.realtime_recognition_logs.append(recognition_record)

                log_recognition(user_id, user_info['name'], 'success', face_image_base64, action='tanıma')
                return True, user_info

            log_recognition(None, None, 'fail', face_image_base64, action='tanıma')
            return False, "Tanınmayan kişi"
        except Exception as e:
            return False, f"Hata: {str(e)}"

    def reset_recognition_session(self):
        """Yeni tanıma oturumu başlatmak için tanınan kişileri temizle"""
        self.recognized_persons.clear()
        self.realtime_recognition_logs.clear()
        return True, "Tanıma oturumu sıfırlandı"

    def get_realtime_recognition_logs(self):
        """Gerçek zamanlı tanıma kayıtlarını döndür"""
        return self.realtime_recognition_logs

    def save_recognition_photo(self, user_id, full_image, face_img):
        """Tanınan kişinin fotoğrafını zaman damgası ile kaydet"""
        try:
            # Zaman damgası oluştur
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Kullanıcının recognition_logs klasörünü oluştur
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)
            recognition_logs_dir = os.path.join(user_dir, 'recognition_logs')
            timestamp_dir = os.path.join(recognition_logs_dir, timestamp)

            os.makedirs(timestamp_dir, exist_ok=True)

            # Tam resmi kaydet (orijinal çekilen fotoğraf)
            full_image_pil = Image.fromarray(full_image)
            full_image_path = os.path.join(timestamp_dir, 'full_image.jpg')
            full_image_pil.save(full_image_path, quality=95)

            # Yüz kısmını kaydet (hizalanmış yüz) - sadece embedding için
            face_pil = Image.fromarray(face_img)
            face_image_path = os.path.join(timestamp_dir, 'face_crop.jpg')
            face_pil.save(face_image_path, quality=95)

            # Tam resmin base64'ünü kaydet (görüntüleme için)
            full_img_bytes = cv2.imencode('.jpg', full_image)[1].tobytes()
            full_base64 = base64.b64encode(full_img_bytes).decode('utf-8')

            # Tanıma bilgilerini JSON dosyasına kaydet
            recognition_info = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'files': {
                    'full_image': 'full_image.jpg',
                    'face_crop': 'face_crop.jpg',
                    'full_base64': full_base64  # Orijinal fotoğrafın base64'ü
                }
            }

            info_path = os.path.join(timestamp_dir, 'recognition_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(recognition_info, f, ensure_ascii=False, indent=2)

            print(f"Tanıma fotoğrafı kaydedildi: {timestamp_dir}")

        except Exception as e:
            print(f"Fotoğraf kaydetme hatası: {str(e)}")

    def delete_user(self, user_id):
        try:
            if not os.path.exists(USERS_DB_FILE):
                return False, "Kullanıcı bulunamadı"
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            user_index = next((i for i, u in enumerate(users_data) if u['id'] == user_id), None)
            if user_index is None:
                return False, "Kullanıcı bulunamadı"
            user_name = users_data[user_index]['name']
            users_data.pop(user_index)
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)

            # Kullanıcının tüm dosyalarını sil (profil fotoğrafları + tanıma fotoğrafları)
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)

            self.load_known_faces()
            return True, f"Kullanıcı {user_name} başarıyla silindi"
        except Exception as e:
            return False, f"Hata: {str(e)}"

    def get_users(self):
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []

face_api = FaceRecognitionAPI()

def log_recognition(user_id, name, result, image_b64=None, action=None):
    log_entry = {
        'user_id': user_id,
        'name': name,
        'date': datetime.now().isoformat(),
        'result': result,
        'action': action,
        'image': image_b64
    }
    logs = []
    if os.path.exists(RECOGNITION_LOG_FILE):
        with open(RECOGNITION_LOG_FILE, 'r', encoding='utf-8') as f:
            logs = json.load(f)
    logs.append(log_entry)
    with open(RECOGNITION_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

@app.route('/fix_json', methods=['POST'])
def fix_json_endpoint():
    """JSON dosyasını düzeltmek için endpoint"""
    try:
        success = fix_corrupted_json()
        if success:
            return jsonify({'success': True, 'message': 'JSON dosyası başarıyla düzeltildi'})
        else:
            return jsonify({'success': False, 'message': 'JSON dosyası düzeltilemedi'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'})

@app.route('/health', methods=['GET'])
def health_check():
    import os
    return jsonify({
        'status': 'OK',
        'message': 'API çalışıyor',
        'timestamp': datetime.now().isoformat(),
        'current_directory': os.getcwd(),
        'known_faces_path': os.path.abspath(KNOWN_FACES_DIR)
    })

@app.route('/reset_recognition_session', methods=['POST'])
def reset_recognition_session():
    """Yeni tanıma oturumu başlatmak için endpoint"""
    try:
        success, message = face_api.reset_recognition_session()
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/realtime_recognition_logs', methods=['GET'])
def get_realtime_recognition_logs():
    """Gerçek zamanlı tanıma kayıtlarını döndüren endpoint"""
    try:
        logs = face_api.get_realtime_recognition_logs()
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_recognition_logs/<id_no>', methods=['GET'])
def test_recognition_logs(id_no):
    """Test endpoint - tanıma loglarını kontrol et"""
    try:
        import os
        current_dir = os.getcwd()
        recognition_logs_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs')
        abs_path = os.path.abspath(recognition_logs_dir)
        exists = os.path.exists(recognition_logs_dir)
        files = []
        subdirs = []
        if exists:
            files = os.listdir(recognition_logs_dir)
            for item in files:
                item_path = os.path.join(recognition_logs_dir, item)
                if os.path.isdir(item_path):
                    subdirs.append({
                        'name': item,
                        'files': os.listdir(item_path)
                    })
        return jsonify({
            'success': True,
            'id_no': id_no,
            'current_directory': current_dir,
            'directory_exists': exists,
            'directory_path': recognition_logs_dir,
            'absolute_path': abs_path,
            'files': files,
            'subdirectories': subdirs
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/recognize', methods=['POST'])
def recognize_face():
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400
        face_image_base64 = data['image']
        success, result = face_api.recognize_face(face_image_base64)
        if success:
            return jsonify({
                'success': True,
                'recognized': True,
                'name': result.get('name'),
                'id_no': result.get('id_no'),
                'birth_date': result.get('birth_date'),
                'message': f"Kişi tanındı: {result.get('name')}"
            })
        else:
            # Eğer result bir dict ise (zaten tanınan kişi durumu)
            if isinstance(result, dict):
                return jsonify({
                    'success': True,
                    'recognized': False,
                    'name': result.get('name'),
                    'id_no': result.get('id_no'),
                    'birth_date': result.get('birth_date'),
                    'message': result
                })
            else:
                return jsonify({'success': False, 'recognized': False, 'message': result})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/add_user', methods=['POST'])
def add_user():
    try:
        data = request.get_json()
        name = data['name']
        images_base64 = data['images']
        id_no = data['id_no']
        birth_date = data['birth_date']
        success, message = face_api.add_user(name, images_base64, id_no, birth_date)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        success, message = face_api.delete_user(user_id)
        return jsonify({'success': success, 'message': message})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/users', methods=['GET'])
def get_users():
    try:
        users = face_api.get_users()
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/user_logs/<user_id>', methods=['GET'])
def get_user_logs(user_id):
    try:
        logs = []
        if os.path.exists(RECOGNITION_LOG_FILE):
            with open(RECOGNITION_LOG_FILE, 'r', encoding='utf-8') as f:
                all_logs = json.load(f)
            logs = [log for log in all_logs if log.get('user_id') == user_id]
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/user_photo/<id_no>/<filename>')
def user_photo(id_no, filename):
    user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
    return send_from_directory(user_dir, filename)

@app.route('/user_photos/<id_no>', methods=['GET'])
def get_user_photos(id_no):
    """Kullanıcının profil fotoğraflarını listeler"""
    try:
        user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
        if not os.path.exists(user_dir):
            return jsonify({'success': True, 'photos': []})

        photos = []
        for filename in os.listdir(user_dir):
            if filename.endswith('.jpg') and filename != 'encodings.pkl':
                photos.append(filename)

        # Dosya adlarına göre sırala (1.jpg, 2.jpg, 3.jpg...)
        photos.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)

        return jsonify({'success': True, 'photos': photos})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_user_photos/<id_no>', methods=['GET'])
def test_user_photos(id_no):
    """Test endpoint - kullanıcı fotoğrafları klasörünü kontrol et"""
    try:
        import os
        current_dir = os.getcwd()
        user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
        abs_path = os.path.abspath(user_dir)
        exists = os.path.exists(user_dir)
        files = []
        if exists:
            files = os.listdir(user_dir)
        return jsonify({
            'success': True,
            'id_no': id_no,
            'current_directory': current_dir,
            'directory_exists': exists,
            'directory_path': user_dir,
            'absolute_path': abs_path,
            'files': files
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/recognition_photos/<id_no>/<timestamp>/<filename>')
def recognition_photo(id_no, timestamp, filename):
    """Tanıma fotoğraflarını görüntülemek için endpoint"""
    recognition_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs', timestamp)
    return send_from_directory(recognition_dir, filename)

@app.route('/recognition_logs/<id_no>', methods=['GET'])
def get_recognition_photos(id_no):
    """Bir kullanıcının tüm tanıma fotoğraflarını listeler"""
    try:
        recognition_logs_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs')
        if not os.path.exists(recognition_logs_dir):
            return jsonify({'success': True, 'logs': []})

        logs = []
        for timestamp_dir in os.listdir(recognition_logs_dir):
            timestamp_path = os.path.join(recognition_logs_dir, timestamp_dir)
            if os.path.isdir(timestamp_path):
                info_file = os.path.join(timestamp_path, 'recognition_info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                        logs.append({
                            'timestamp': timestamp_dir,
                            'datetime': info.get('timestamp'),
                            'files': info.get('files', {}),
                            'photo_urls': {
                                'full_image': f'/recognition_photos/{id_no}/{timestamp_dir}/full_image.jpg',
                                'face_crop': f'/recognition_photos/{id_no}/{timestamp_dir}/face_crop.jpg'
                            },
                            'full_base64': info.get('files', {}).get('full_base64', '')  # Orijinal fotoğraf
                        })

        # Tarihe göre sırala (en yeni en üstte)
        logs.sort(key=lambda x: x['datetime'], reverse=True)

        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/test_image_resolution', methods=['POST'])
def test_image_resolution():
    """Görüntü çözünürlüğünü test etmek için endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

        face_image_base64 = data['image']
        image_array = face_api.process_image_with_rotation(face_image_base64)

        if image_array is None:
            return jsonify({'success': False, 'message': 'Görüntü işlenemedi'}), 400

        height, width = image_array.shape[:2]
        faces_with_landmarks = face_api.detect_faces_with_landmarks(image_array)

        # Çözünürlük analizi
        resolution_quality = "İyi"
        if width < 640 or height < 640:
            resolution_quality = "Düşük"
        elif width < 1280 or height < 1280:
            resolution_quality = "Orta"

        # Yüz tespit analizi
        face_detection_quality = "Başarılı"
        if len(faces_with_landmarks) == 0:
            face_detection_quality = "Yüz tespit edilemedi"
        elif len(faces_with_landmarks) > 1:
            face_detection_quality = f"{len(faces_with_landmarks)} yüz tespit edildi"

        # Yüz boyutu analizi
        face_size_quality = "Uygun"
        if len(faces_with_landmarks) > 0:
            x, y, w, h = faces_with_landmarks[0]['box']
            if w < 80 or h < 80:
                face_size_quality = "Çok küçük"
            elif w < 120 or h < 120:
                face_size_quality = "Küçük"

        return jsonify({
            'success': True,
            'image_info': {
                'width': width,
                'height': height,
                'resolution_quality': resolution_quality
            },
            'face_detection': {
                'faces_found': len(faces_with_landmarks),
                'detection_quality': face_detection_quality,
                'face_size_quality': face_size_quality
            },
            'recommendations': {
                'resolution': "1280x1280 veya daha yüksek önerilir" if resolution_quality == "Düşük" else "Çözünürlük uygun",
                'face_size': "Yüz daha büyük olmalı" if face_size_quality in ["Çok küçük", "Küçük"] else "Yüz boyutu uygun",
                'lighting': "İyi aydınlatma önerilir" if len(faces_with_landmarks) == 0 else "Aydınlatma uygun"
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    """Yüz tespit koordinatlarını döndüren endpoint"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

        face_image_base64 = data['image']
        image_array = face_api.process_image_with_rotation(face_image_base64)

        if image_array is None:
            return jsonify({'success': False, 'message': 'Görüntü işlenemedi'}), 400

        height, width = image_array.shape[:2]
        faces_with_landmarks = face_api.detect_faces_with_landmarks(image_array)
        rotation_angle = 0

        # Eğer orijinal görüntüde yüz bulunamazsa, farklı rotasyonlarda dene
        if len(faces_with_landmarks) == 0:
            print("Orijinal görüntüde yüz bulunamadı, rotasyonlar deneniyor...")

            # 90 derece döndür
            rotated_90 = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
            faces_with_landmarks = face_api.detect_faces_with_landmarks(rotated_90)
            if len(faces_with_landmarks) > 0:
                rotation_angle = 90
                image_array = rotated_90
                height, width = image_array.shape[:2]
                print("90° rotasyonda yüz bulundu")

            # 180 derece döndür
            if len(faces_with_landmarks) == 0:
                rotated_180 = cv2.rotate(image_array, cv2.ROTATE_180)
                faces_with_landmarks = face_api.detect_faces_with_landmarks(rotated_180)
                if len(faces_with_landmarks) > 0:
                    rotation_angle = 180
                    image_array = rotated_180
                    height, width = image_array.shape[:2]
                    print("180° rotasyonda yüz bulundu")

            # 270 derece döndür
            if len(faces_with_landmarks) == 0:
                rotated_270 = cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                faces_with_landmarks = face_api.detect_faces_with_landmarks(rotated_270)
                if len(faces_with_landmarks) > 0:
                    rotation_angle = 270
                    image_array = rotated_270
                    height, width = image_array.shape[:2]
                    print("270° rotasyonda yüz bulundu")

        # Yüz koordinatlarını hazırla
        faces_data = []
        for face in faces_with_landmarks:
            x, y, w, h = face['box']
            confidence = face['confidence']

            # Margin ekle (daha geniş margin)
            margin = int(min(w, h) * 0.3)  # %30 margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)

            # Kare şeklinde yap (en büyük boyutu kullan)
            size = max(x2 - x1, y2 - y1)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Kare koordinatları
            square_x = max(0, center_x - size // 2)
            square_y = max(0, center_y - size // 2)
            square_size = min(size, width - square_x, height - square_y)

            faces_data.append({
                'x': square_x,
                'y': square_y,
                'width': square_size,
                'height': square_size,
                'confidence': confidence,
                'rotation_angle': rotation_angle,
                'original_box': {
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h
                }
            })

        return jsonify({
            'success': True,
            'faces': faces_data,
            'total_faces': len(faces_data),
            'image_info': {
                'width': width,
                'height': height,
                'rotation_angle': rotation_angle
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

if __name__ == '__main__':
    print("Yüz Tanıma API'si başlatılıyor...")
    app.run(host='0.0.0.0', port=5000, debug=True)
