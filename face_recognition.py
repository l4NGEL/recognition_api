import cv2
import numpy as np
import os
import json
from datetime import datetime
import base64
from PIL import Image
import io
import pickle
import tensorflow as tf
import shutil
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import roc_curve, auc

from config import USERS_DB_FILE, KNOWN_FACES_DIR, MOBILFACENET_MODEL_PATH
from utils import auto_rotate_image, fix_corrupted_json, create_augmented_versions
from face_detection import FaceDetector
from logging import log_recognition, log_threshold_event

class FaceRecognitionAPI:
    def __init__(self):
        self.mobilfacenet_interpreter = None
        self.input_details = None
        self.output_details = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.known_embeddings_array = None
        self.users_data_cache = []
        self.auto_threshold_enabled = True
        self.optimal_threshold = 0.7437
        self.threshold_history = []
        
        self.face_detector = FaceDetector()
        self.recognized_persons = set()
        self.realtime_recognition_logs = []
        self.last_cache_update = None
        
        self.load_mobilfacenet_model(MOBILFACENET_MODEL_PATH)
        self.load_known_faces()
        self.update_cache()

    def update_cache(self):
        try:
            if len(self.known_face_encodings) > 0:
                self.known_embeddings_array = np.array(self.known_face_encodings)

            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                self.users_data_cache = json.load(f)

            self.last_cache_update = datetime.now()
        except Exception as e:
            print(f"Cache güncelleme hatası: {e}")

    def load_mobilfacenet_model(self, model_path):
        self.mobilfacenet_interpreter = tf.lite.Interpreter(model_path=model_path)
        self.mobilfacenet_interpreter.allocate_tensors()
        self.input_details = self.mobilfacenet_interpreter.get_input_details()
        self.output_details = self.mobilfacenet_interpreter.get_output_details()

    def get_embedding(self, face_img):
        try:
            height, width = face_img.shape[:2]
            print(f"Yüz boyutu: {width}x{height}")

            if width < 80 or height < 80:
                print(f"Yüz çok küçük: {width}x{height}, minimum 80x80 gerekli")
                return None

            img = cv2.resize(face_img, (160, 160), interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)

            self.mobilfacenet_interpreter.set_tensor(self.input_details[0]['index'], img)
            self.mobilfacenet_interpreter.invoke()
            embedding = self.mobilfacenet_interpreter.get_tensor(self.output_details[0]['index'])
            embedding = embedding.flatten()
            
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            print(f"Embedding oluşturma hatası: {e}")
            return None

    def calculate_optimal_threshold(self, test_embeddings, known_embeddings):
        try:
            if len(test_embeddings) == 0 or len(known_embeddings) == 0:
                return self.optimal_threshold
            
            distances = []
            for test_emb in test_embeddings:
                for known_emb in known_embeddings:
                    dist = np.linalg.norm(test_emb - known_emb)
                    distances.append(dist)
            
            if len(distances) == 0:
                return self.optimal_threshold
            
            print(f"{len(distances)} çift oluşturuldu.")
            print(f"{len(self.known_face_encodings)} adet encoding kullanıldı.")
            
            distances.sort()
            
            percentile_95 = np.percentile(distances, 95)
            percentile_90 = np.percentile(distances, 90)
            percentile_85 = np.percentile(distances, 85)
            
            mean_dist = np.mean(distances)
            std_dist = np.std(distances)
            
            threshold_candidates = [
                percentile_95,
                percentile_90,
                percentile_85,
                mean_dist + std_dist,
                mean_dist + 0.5 * std_dist,
                self.optimal_threshold
            ]
            
            optimal_threshold = min(threshold_candidates)
            optimal_threshold = max(0.3, min(0.8, optimal_threshold))
            
            self.threshold_history.append({
                'timestamp': datetime.now().isoformat(),
                'threshold': optimal_threshold,
                'mean_distance': mean_dist,
                'std_distance': std_dist,
                'percentile_95': percentile_95,
                'percentile_90': percentile_90,
                'percentile_85': percentile_85
            })
            
            if len(self.threshold_history) > 100:
                self.threshold_history = self.threshold_history[-100:]
            
            self.optimal_threshold = optimal_threshold
            print(f"Optimal threshold hesaplandı: {optimal_threshold:.4f}")
            return optimal_threshold
            
        except Exception as e:
            print(f"Threshold hesaplama hatası: {e}")
            return self.optimal_threshold

    def get_adaptive_threshold(self):
        if not self.auto_threshold_enabled:
            return 0.7437
        
        if len(self.threshold_history) > 0:
            recent_thresholds = [h['threshold'] for h in self.threshold_history[-10:]]
            return np.mean(recent_thresholds)
        
        return self.optimal_threshold

    def calculate_threshold_via_roc(self):
        try:
            print("ROC tabanlı threshold hesaplaması başlatıldı...")

            embeddings_dict = defaultdict(list)
            for name, id_no, emb in zip(self.known_face_names, self.known_face_ids, self.known_face_encodings):
                embeddings_dict[id_no].append(emb)

            if len(embeddings_dict) < 2:
                print("ROC için yeterli kullanıcı yok (en az 2 kullanıcı gerekli)")
                return self.optimal_threshold

            y_true = []
            distances = []

            print("Pozitif çiftler hesaplanıyor...")
            for emb_list in embeddings_dict.values():
                if len(emb_list) >= 2:
                    for emb1, emb2 in combinations(emb_list, 2):
                        dist = np.linalg.norm(emb1 - emb2)
                        distances.append(dist)
                        y_true.append(1)

            print("Negatif çiftler hesaplanıyor...")
            person_ids = list(embeddings_dict.keys())
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    for emb1 in embeddings_dict[person_ids[i]]:
                        for emb2 in embeddings_dict[person_ids[j]]:
                            dist = np.linalg.norm(emb1 - emb2)
                            distances.append(dist)
                            y_true.append(0)

            if not distances or not y_true:
                print("ROC için yeterli veri yok")
                return self.optimal_threshold

            print(f"Toplam {len(distances)} çift hesaplandı ({sum(y_true)} pozitif, {len(y_true) - sum(y_true)} negatif)")
            print(f"{len(distances)} çift oluşturuldu.")
            print(f"{len(self.known_face_encodings)} adet encoding kullanıldı.")

            fpr, tpr, thresholds = roc_curve(y_true, [-d for d in distances])
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = abs(thresholds[optimal_idx])

            roc_auc = auc(fpr, tpr)

            print(f"✅ ROC üzerinden hesaplanan yeni threshold: {optimal_threshold:.4f}")
            print(f"📊 AUC: {roc_auc:.4f}")
            print(f"📈 TPR: {tpr[optimal_idx]:.4f}, FPR: {fpr[optimal_idx]:.4f}")

            self.optimal_threshold = max(0.3, min(0.8, optimal_threshold))
            self.auto_threshold_enabled = True

            self.threshold_history.append({
                'timestamp': datetime.now().isoformat(),
                'threshold': self.optimal_threshold,
                'method': 'roc_curve',
                'auc': roc_auc,
                'tpr': float(tpr[optimal_idx]),
                'fpr': float(fpr[optimal_idx]),
                'total_pairs': len(distances),
                'positive_pairs': sum(y_true),
                'negative_pairs': len(y_true) - sum(y_true)
            })

            if len(self.threshold_history) > 100:
                self.threshold_history = self.threshold_history[-100:]

            return self.optimal_threshold

        except Exception as e:
            print(f"ROC tabanlı threshold hesaplama hatası: {e}")
            return self.optimal_threshold

    def compare_embeddings(self, emb1, emb2, threshold=None):
        if threshold is None:
            threshold = self.get_adaptive_threshold()
        
        try:
            dist = np.linalg.norm(emb1 - emb2)
            return dist < threshold
        except Exception as e:
            print(f"Embedding karşılaştırma hatası: {e}")
            return False

    def compare_embeddings_batch(self, query_embedding, known_embeddings, threshold=None):
        if threshold is None:
            threshold = self.get_adaptive_threshold()
        
        try:
            if len(known_embeddings) == 0:
                return -1, float('inf')
            
            distances = np.linalg.norm(known_embeddings - query_embedding, axis=1)
            
            min_idx = np.argmin(distances)
            min_distance = distances[min_idx]
            
            return min_idx, min_distance
            
        except Exception as e:
            print(f"Batch embedding karşılaştırma hatası: {e}")
            return -1, float('inf')

    def process_image_with_rotation(self, image_data_base64):
        try:
            image_data = base64.b64decode(image_data_base64.split(',')[1] if ',' in image_data_base64 else image_data_base64)
            image = Image.open(io.BytesIO(image_data))

            if image.mode != 'RGB':
                image = image.convert('RGB')

            image = auto_rotate_image(image)

            image_array = np.array(image)

            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

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

        self.update_cache()

    def add_user(self, name, images_base64, id_no=None, birth_date=None):
        try:
            print(f"Add user başlatıldı: {name}, {id_no}")

            if not id_no or not str(id_no).isdigit() or len(str(id_no)) != 11:
                return False, "Kimlik numarası 11 haneli olmalıdır."

            if os.path.exists(USERS_DB_FILE):
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                if any(user['id_no'] == id_no for user in users_data):
                    return False, "Bu kimlik numarasına sahip kullanıcı zaten kayıtlı."
            else:
                users_data = []

            user_id = str(id_no)
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)

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

                image_array = self.process_image_with_rotation(img_b64)
                if image_array is None:
                    print(f"Resim {idx+1} işlenemedi")
                    continue

                faces_with_landmarks = self.face_detector.detect_faces_with_landmarks(image_array)
                rotation_angle = 0

                if len(faces_with_landmarks) == 0:
                    print(f"Resim {idx+1} için orijinal görüntüde yüz bulunamadı")
                    continue

                face_data = faces_with_landmarks[0]
                x, y, w, h = face_data['box']
                landmarks = face_data['keypoints']

                margin = int(min(w, h) * 0.15)
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(image_array.shape[1], x + w + margin)
                y2 = min(image_array.shape[0], y + h + margin)

                face_img = image_array[y1:y2, x1:x2]

                face_img = self.face_detector.remove_black_borders(face_img)

                adjusted_landmarks = {}
                for key, point in landmarks.items():
                    adjusted_landmarks[key] = (point[0] - x1, point[1] - y1)

                aligned_face = self.face_detector.align_face(face_img, adjusted_landmarks)

                aligned_face = self.face_detector.remove_black_borders(aligned_face)

                main_embedding = self.get_embedding(aligned_face)
                if main_embedding is None:
                    print(f"Resim {idx+1} için embedding oluşturulamadı - yüz çözünürlüğü yetersiz")
                    continue

                encodings.append(main_embedding)

                try:
                    face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    original_photo_path = os.path.join(user_dir, f'{idx+1}_original.jpg')
                    face_pil.save(original_photo_path)
                    saved_photos += 1
                    print(f"Orijinal fotoğraf kaydedildi: {original_photo_path}")
                except Exception as e:
                    print(f"Orijinal fotoğraf kaydetme hatası: {e}")

                try:
                    face_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    
                    augmented_versions = create_augmented_versions(face_pil, num_versions=4)
                    
                    for aug_idx, aug_img in enumerate(augmented_versions):
                        try:
                            aug_array = np.array(aug_img)
                            aug_bgr = cv2.cvtColor(aug_array, cv2.COLOR_RGB2BGR)
                            aug_embedding = self.get_embedding(aug_bgr)
                            
                            if aug_embedding is not None:
                                encodings.append(aug_embedding)
                            
                            aug_path = os.path.join(user_dir, f'{idx+1}_aug{aug_idx+1}.jpg')
                            aug_img.save(aug_path)
                            saved_photos += 1
                            print(f"Augmented fotoğraf {aug_idx+1} kaydedildi: {aug_path}")
                            
                        except Exception as e:
                            print(f"Augmented fotoğraf {aug_idx+1} kaydetme hatası: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Augmentation hatası: {e}")
                    continue

            if not encodings:
                return False, "Hiçbir resimde yüz tespit edilemedi"

            try:
                encodings_path = os.path.join(user_dir, 'encodings.pkl')
                with open(encodings_path, 'wb') as ef:
                    pickle.dump(encodings, ef)
                print(f"Embeddings kaydedildi: {encodings_path}")
            except Exception as e:
                print(f"Embeddings kaydetme hatası: {e}")
                return False, f"Embeddings kaydedilemedi: {e}"

            try:
                self.save_users_db(user_id, name, id_no, birth_date)
                print(f"Kullanıcı JSON'a kaydedildi: {user_id}")
            except Exception as e:
                print(f"JSON kaydetme hatası: {e}")
                return False, f"Kullanıcı bilgileri kaydedilemedi: {e}"

            for enc in encodings:
                self.known_face_encodings.append(enc)
                self.known_face_names.append(name)
                self.known_face_ids.append(user_id)

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
            image_array = self.process_image_with_rotation(face_image_base64)
            if image_array is None:
                return False, "Görüntü işlenemedi"

            faces_with_landmarks = self.face_detector.detect_faces_with_landmarks(image_array)
            rotation_angle = 0

            if len(faces_with_landmarks) == 0:
                print("Orijinal görüntüde yüz bulunamadı")
                return False, "Yüz tespit edilemedi"

            if len(faces_with_landmarks) == 0:
                return False, "Yüz tespit edilemedi"

            if self.users_data_cache is None:
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    self.users_data_cache = json.load(f)

            face_data = faces_with_landmarks[0]
            x, y, w, h = face_data['box']
            landmarks = face_data['keypoints']

            margin = int(min(w, h) * 0.15)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(image_array.shape[1], x + w + margin)
            y2 = min(image_array.shape[0], y + h + margin)

            face_img = image_array[y1:y2, x1:x2]

            face_img = self.face_detector.remove_black_borders(face_img)

            adjusted_landmarks = {}
            for key, point in landmarks.items():
                adjusted_landmarks[key] = (point[0] - x1, point[1] - y1)

            aligned_face = self.face_detector.align_face(face_img, adjusted_landmarks)

            aligned_face = self.face_detector.remove_black_borders(aligned_face)

            main_embedding = self.get_embedding(aligned_face)
            if main_embedding is None:
                return False, "Yüz çözünürlüğü yetersiz veya embedding oluşturulamadı"

            if self.auto_threshold_enabled and len(self.known_face_encodings) > 0:
                test_embeddings = [main_embedding]
                
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
                            test_embeddings.append(rotated_embedding)
                    except Exception as e:
                        print(f"Rotasyon {rotation}° embedding hatası: {e}")
                        continue
                
                optimal_threshold = self.calculate_optimal_threshold(test_embeddings, self.known_embeddings_array)
                print(f"Kullanılan threshold: {optimal_threshold:.4f}")
            else:
                optimal_threshold = self.get_adaptive_threshold()

            best_match_idx = -1
            min_distance = float('inf')
            match_confidence = 0.0

            if self.known_embeddings_array is not None and len(self.known_embeddings_array) > 0:
                current_match_idx, current_distance = self.compare_embeddings_batch(
                    main_embedding, self.known_embeddings_array, threshold=optimal_threshold
                )

                if current_match_idx != -1 and current_distance < min_distance:
                    best_match_idx = current_match_idx
                    min_distance = current_distance
                    match_confidence = max(0, 1 - (current_distance / optimal_threshold))

            if best_match_idx != -1 and min_distance < optimal_threshold:
                user_id = self.known_face_ids[best_match_idx]
                user_info = next((u for u in self.users_data_cache if u['id'] == user_id), None)

                if user_info is None:
                    return False, "Kullanıcı bilgisi bulunamadı"

                log_threshold_event(
                    user_id=user_id,
                    name=user_info['name'],
                    threshold_used=optimal_threshold,
                    distance=min_distance,
                    confidence=match_confidence,
                    success=True,
                    image_b64=face_image_base64
                )

                if user_id in self.recognized_persons:
                    return False, user_info

                self.recognized_persons.add(user_id)

                self.save_recognition_photo(user_id, image_array, aligned_face)

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

            if best_match_idx != -1:
                closest_user_id = self.known_face_ids[best_match_idx]
                closest_user_info = next((u for u in self.users_data_cache if u['id'] == closest_user_id), None)
                closest_name = closest_user_info['name'] if closest_user_info else "Bilinmeyen"
                
                log_threshold_event(
                    user_id=closest_user_id,
                    name=closest_name,
                    threshold_used=optimal_threshold,
                    distance=min_distance,
                    confidence=match_confidence,
                    success=False,
                    image_b64=face_image_base64
                )
            else:
                log_threshold_event(
                    user_id="unknown",
                    name="Tanınmayan Kişi",
                    threshold_used=optimal_threshold,
                    distance=float('inf'),
                    confidence=0.0,
                    success=False,
                    image_b64=face_image_base64
                )

            log_recognition(None, None, 'fail', face_image_base64, action='tanıma')
            return False, "Tanınmayan kişi"
        except Exception as e:
            return False, f"Hata: {str(e)}"

    def reset_recognition_session(self):
        self.recognized_persons.clear()
        self.realtime_recognition_logs.clear()
        return True, "Tanıma oturumu sıfırlandı"

    def get_realtime_recognition_logs(self):
        return self.realtime_recognition_logs

    def save_recognition_photo(self, user_id, full_image, face_img):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)
            recognition_logs_dir = os.path.join(user_dir, 'recognition_logs')
            timestamp_dir = os.path.join(recognition_logs_dir, timestamp)

            os.makedirs(timestamp_dir, exist_ok=True)

            full_image_pil = Image.fromarray(full_image)
            full_image_path = os.path.join(timestamp_dir, 'full_image.jpg')
            full_image_pil.save(full_image_path, quality=95)

            face_pil = Image.fromarray(face_img)
            face_image_path = os.path.join(timestamp_dir, 'face_crop.jpg')
            face_pil.save(face_image_path, quality=95)

            full_img_bytes = cv2.imencode('.jpg', full_image)[1].tobytes()
            full_base64 = base64.b64encode(full_img_bytes).decode('utf-8')

            recognition_info = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'files': {
                    'full_image': 'full_image.jpg',
                    'face_crop': 'face_crop.jpg',
                    'full_base64': full_base64
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