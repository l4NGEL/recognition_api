from flask import request, jsonify, send_from_directory
from datetime import datetime
import json
import os
import base64
import io
import cv2
import numpy as np
from PIL import Image

from config import USERS_DB_FILE, KNOWN_FACES_DIR, RECOGNITION_LOG_FILE
from face_recognition import FaceRecognitionAPI
from utils import get_augmentation_stats, create_augmented_versions, apply_brightness_contrast, apply_noise, apply_blur, apply_sharpening, apply_geometric_transform, apply_lighting_simulation
from logging import log_recognition, log_threshold_event

face_api = FaceRecognitionAPI()

def init_routes(app):
    @app.route('/fix_json', methods=['POST'])
    def fix_json_endpoint():
        try:
            from utils import fix_corrupted_json
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
        try:
            success, message = face_api.reset_recognition_session()
            new_threshold = face_api.calculate_threshold_via_roc()
            
            return jsonify({
                'success': success, 
                'message': message,
                'new_threshold': new_threshold,
                'threshold_method': 'roc_curve'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/realtime_recognition_logs', methods=['GET'])
    def get_realtime_recognition_logs():
        try:
            logs = face_api.get_realtime_recognition_logs()
            return jsonify({'success': True, 'logs': logs})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/test_recognition_logs/<id_no>', methods=['GET'])
    def test_recognition_logs(id_no):
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

    @app.route('/update_user_name/<user_id>', methods=['PUT'])
    def update_user_name(user_id):
        try:
            print(f"update_user_name çağrıldı: user_id={user_id}")
            data = request.get_json()
            print(f"Gelen veri: {data}")
            new_name = data.get('name')
            
            if not new_name:
                print("Yeni isim belirtilmedi")
                return jsonify({'success': False, 'message': 'Yeni isim belirtilmedi'}), 400
            
            print(f"Yeni isim: {new_name}")
            
            if not os.path.exists(USERS_DB_FILE):
                print("Users.json dosyası bulunamadı")
                return jsonify({'success': False, 'message': 'Kullanıcı veritabanı bulunamadı'}), 404
            
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            
            print(f"Toplam kullanıcı sayısı: {len(users_data)}")
            
            user_found = False
            for user in users_data:
                print(f"🔍 Kullanıcı kontrol ediliyor: {user.get('id')} veya {user.get('id_no')} == {user_id}")
                if user.get('id') == user_id or user.get('id_no') == user_id:
                    print(f"✅ Kullanıcı bulundu: {user}")
                    user['name'] = new_name
                    user_found = True
                    break
            
            if not user_found:
                print(f"Kullanıcı bulunamadı: {user_id}")
                return jsonify({'success': False, 'message': 'Kullanıcı bulunamadı'}), 404
            
            with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)
            
            print("Veri kaydedildi")
            
            face_api.update_cache()
            
            print("Kullanıcı adı başarıyla güncellendi")
            return jsonify({'success': True, 'message': 'Kullanıcı adı başarıyla güncellendi'})
            
        except Exception as e:
            print(f"update_user_name hatası: {e}")
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
        try:
            user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
            if not os.path.exists(user_dir):
                return jsonify({'success': True, 'photos': []})

            photos = []
            for filename in os.listdir(user_dir):
                if filename.endswith('.jpg') and filename != 'encodings.pkl':
                    photos.append(filename)

            photos.sort(key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 0)

            return jsonify({'success': True, 'photos': photos})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/user_photos/<id_no>/delete', methods=['POST'])
    def delete_user_photos(id_no):
        try:
            data = request.get_json()
            photo_names = data.get('photo_names', [])
            
            if not photo_names:
                return jsonify({'success': False, 'message': 'Silinecek fotoğraf belirtilmedi'}), 400
            
            user_dir = os.path.join(KNOWN_FACES_DIR, id_no)
            if not os.path.exists(user_dir):
                return jsonify({'success': False, 'message': 'Kullanıcı bulunamadı'}), 404
            
            deleted_count = 0
            errors = []
            
            for photo_name in photo_names:
                try:
                    photo_path = os.path.join(user_dir, photo_name)
                    if os.path.exists(photo_path):
                        os.remove(photo_path)
                        deleted_count += 1
                        print(f"Fotoğraf silindi: {photo_path}")
                    else:
                        errors.append(f"Fotoğraf bulunamadı: {photo_name}")
                except Exception as e:
                    errors.append(f"Silme hatası ({photo_name}): {str(e)}")
            
            face_api.load_known_faces()
            
            return jsonify({
                'success': True,
                'deleted_count': deleted_count,
                'errors': errors,
                'message': f'{deleted_count} fotoğraf silindi'
            })
            
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500
    
    @app.route('/test_user_photos/<id_no>', methods=['GET'])
    def test_user_photos(id_no):
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
        recognition_dir = os.path.join(KNOWN_FACES_DIR, id_no, 'recognition_logs', timestamp)
        return send_from_directory(recognition_dir, filename)

    @app.route('/recognition_logs/<id_no>', methods=['GET'])
    def get_recognition_photos(id_no):
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
                                'full_base64': info.get('files', {}).get('full_base64', '') 
                            })

            logs.sort(key=lambda x: x['datetime'], reverse=True)

            return jsonify({'success': True, 'logs': logs})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/augmentation_stats', methods=['GET'])
    def augmentation_stats():
        try:
            stats = get_augmentation_stats()
            return jsonify({'success': True, 'stats': stats})
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/test_augmentation', methods=['POST'])
    def test_augmentation():
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

            image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')

            num_versions = data.get('num_versions', 4)
            augmented_versions = create_augmented_versions(image, num_versions)
            
            results = []
            for i, aug_img in enumerate(augmented_versions):
                img_buffer = io.BytesIO()
                aug_img.save(img_buffer, format='JPEG', quality=95)
                img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                results.append({
                    'index': i + 1,
                    'image': f'data:image/jpeg;base64,{img_str}'
                })

            return jsonify({
                'success': True,
                'original_image': data['image'],
                'augmented_versions': results,
                'total_versions': len(results)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/apply_single_augmentation', methods=['POST'])
    def apply_single_augmentation():
        try:
            data = request.get_json()
            if not data or 'image' not in data or 'augmentation_type' not in data:
                return jsonify({'success': False, 'message': 'Resim ve augmentation türü gerekli'}), 400

            image_data = base64.b64decode(data['image'].split(',')[1] if ',' in data['image'] else data['image'])
            image = Image.open(io.BytesIO(image_data))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')

            augmentation_type = data['augmentation_type']
            params = data.get('params', {})

            if augmentation_type == 'brightness_contrast':
                brightness = params.get('brightness', 1.2)
                contrast = params.get('contrast', 1.1)
                result = apply_brightness_contrast(image, brightness, contrast)
            elif augmentation_type == 'noise':
                noise_factor = params.get('noise_factor', 0.02)
                result = apply_noise(image, noise_factor)
            elif augmentation_type == 'blur':
                blur_factor = params.get('blur_factor', 1.0)
                result = apply_blur(image, blur_factor)
            elif augmentation_type == 'sharpening':
                sharpness = params.get('sharpness', 1.5)
                result = apply_sharpening(image, sharpness)
            elif augmentation_type == 'geometric_transform':
                rotation_range = params.get('rotation_range', 10)
                scale_range = params.get('scale_range', 0.1)
                result = apply_geometric_transform(image, rotation_range, scale_range)
            elif augmentation_type == 'lighting_simulation':
                lighting_factor = params.get('lighting_factor', 0.3)
                result = apply_lighting_simulation(image, lighting_factor)
            else:
                return jsonify({'success': False, 'message': 'Geçersiz augmentation türü'}), 400

            img_buffer = io.BytesIO()
            result.save(img_buffer, format='JPEG', quality=95)
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

            return jsonify({
                'success': True,
                'original_image': data['image'],
                'augmented_image': f'data:image/jpeg;base64,{img_str}',
                'augmentation_type': augmentation_type,
                'params': params
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/test_image_resolution', methods=['POST'])
    def test_image_resolution():
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

            face_image_base64 = data['image']
            image_array = face_api.process_image_with_rotation(face_image_base64)

            if image_array is None:
                return jsonify({'success': False, 'message': 'Görüntü işlenemedi'}), 400

            height, width = image_array.shape[:2]
            faces_with_landmarks = face_api.face_detector.detect_faces_with_landmarks(image_array)

            resolution_quality = "İyi"
            if width < 640 or height < 640:
                resolution_quality = "Düşük"
            elif width < 1280 or height < 1280:
                resolution_quality = "Orta"

            face_detection_quality = "Başarılı"
            if len(faces_with_landmarks) == 0:
                face_detection_quality = "Yüz tespit edilemedi (sadece orijinal yönde arandı)"
            elif len(faces_with_landmarks) > 1:
                face_detection_quality = f"{len(faces_with_landmarks)} yüz tespit edildi"

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
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

            face_image_base64 = data['image']
            image_array = face_api.process_image_with_rotation(face_image_base64)

            if image_array is None:
                return jsonify({'success': False, 'message': 'Görüntü işlenemedi'}), 400

            height, width = image_array.shape[:2]
            faces_with_landmarks = face_api.face_detector.detect_faces_with_landmarks(image_array)
            rotation_angle = 0

            if len(faces_with_landmarks) == 0:
                print("Orijinal görüntüde yüz bulunamadı")
                return jsonify({
                    'success': True,
                    'faces': [],
                    'total_faces': 0,
                    'image_info': {
                        'width': width,
                        'height': height,
                        'rotation_angle': rotation_angle
                    },
                    'message': 'Yüz tespit edilemedi'
                })

            faces_data = []
            for face in faces_with_landmarks:
                x, y, w, h = face['box']
                confidence = face['confidence']

                margin = int(min(w, h) * 0.3)  
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(width, x + w + margin)
                y2 = min(height, y + h + margin)

                size = max(x2 - x1, y2 - y1)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

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

    @app.route('/threshold/status', methods=['GET'])
    def threshold_status():
        try:
            last_method = None
            last_auc = None
            last_tpr = None
            last_fpr = None
            
            if face_api.threshold_history:
                last_record = face_api.threshold_history[-1]
                last_method = last_record.get('method', 'unknown')
                last_auc = last_record.get('auc')
                last_tpr = last_record.get('tpr')
                last_fpr = last_record.get('fpr')
            
            return jsonify({
                'success': True,
                'auto_threshold_enabled': face_api.auto_threshold_enabled,
                'current_threshold': face_api.optimal_threshold,
                'adaptive_threshold': face_api.get_adaptive_threshold(),
                'threshold_history_count': len(face_api.threshold_history),
                'last_threshold_update': face_api.threshold_history[-1]['timestamp'] if face_api.threshold_history else None,
                'last_method': last_method,
                'last_auc': last_auc,
                'last_tpr': last_tpr,
                'last_fpr': last_fpr
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/toggle', methods=['POST'])
    def toggle_auto_threshold():
        try:
            data = request.get_json()
            enabled = data.get('enabled', True)
            face_api.auto_threshold_enabled = enabled
            
            return jsonify({
                'success': True,
                'auto_threshold_enabled': face_api.auto_threshold_enabled,
                'message': f'Otomatik threshold {"açıldı" if enabled else "kapatıldı"}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/history', methods=['GET'])
    def get_threshold_history():
        try:
            return jsonify({
                'success': True,
                'history': face_api.threshold_history[-20:]  
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/set', methods=['POST'])
    def set_manual_threshold():
        try:
            data = request.get_json()
            threshold = data.get('threshold', 0.7270)
            
            threshold = max(0.3, min(0.8, threshold))
            face_api.optimal_threshold = threshold
            
            return jsonify({
                'success': True,
                'threshold': face_api.optimal_threshold,
                'message': f'Threshold {threshold:.4f} olarak ayarlandı'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/calculate_roc', methods=['POST'])
    def calculate_roc_threshold():
        try:
            new_threshold = face_api.calculate_threshold_via_roc()
            
            return jsonify({
                'success': True,
                'threshold': new_threshold,
                'method': 'roc_curve',
                'message': f'ROC tabanlı threshold hesaplandı: {new_threshold:.4f}'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/optimize_threshold', methods=['POST'])
    def optimize_threshold():
        try:
            print("🔧 Threshold optimizasyonu başlatılıyor...")
            
            current_threshold = face_api.get_adaptive_threshold()
            print(f"Mevcut threshold: {current_threshold}")
            
            if not hasattr(face_api, 'known_face_encodings') or not face_api.known_face_encodings:
                print("Embeddings bulunamadı - yeniden yükleniyor...")
                face_api.load_known_faces()
                
                if not face_api.known_face_encodings:
                    print("Embeddings hala bulunamadı")
                    return jsonify({
                        'success': False,
                        'message': 'Embeddings bulunamadı. Lütfen önce kullanıcı ekleyin.'
                    }), 400
            
            if not hasattr(face_api, 'known_face_names') or not face_api.known_face_names:
                print("Face names bulunamadı")
                return jsonify({
                    'success': False,
                    'message': 'Face names bulunamadı. Lütfen önce kullanıcı ekleyin.'
                }), 400
            
            if not hasattr(face_api, 'known_face_ids') or not face_api.known_face_ids:
                print("Face IDs bulunamadı")
                return jsonify({
                    'success': False,
                    'message': 'Face IDs bulunamadı. Lütfen önce kullanıcı ekleyin.'
                }), 400
            
            print(f"Toplam embeddings: {len(face_api.known_face_encodings)}")
            print(f"Toplam names: {len(face_api.known_face_names)}")
            print(f"Toplam IDs: {len(face_api.known_face_ids)}")
            
            from collections import defaultdict
            embeddings_dict = defaultdict(list)
            for name, id_no, emb in zip(face_api.known_face_names, face_api.known_face_ids, face_api.known_face_encodings):
                embeddings_dict[id_no].append(emb)

            if len(embeddings_dict) < 2:
                print(f"❌ Yeterli kullanıcı yok: {len(embeddings_dict)}")
                return jsonify({
                    'success': False,
                    'message': f'ROC için yeterli kullanıcı yok (en az 2 kullanıcı gerekli, mevcut: {len(embeddings_dict)})'
                }), 400

            y_true = []
            distances = []

            print("Pozitif çiftler hesaplanıyor...")
            positive_pairs = 0
            from itertools import combinations
            for emb_list in embeddings_dict.values():
                if len(emb_list) >= 2:  
                    for emb1, emb2 in combinations(emb_list, 2):
                        try:
                            dist = np.linalg.norm(emb1 - emb2)
                            distances.append(dist)
                            y_true.append(1)
                            positive_pairs += 1
                        except Exception as e:
                            print(f"❌ Pozitif pair hesaplama hatası: {e}")
                            continue

            print("Negatif çiftler hesaplanıyor...")
            negative_pairs = 0
            person_ids = list(embeddings_dict.keys())
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    for emb1 in embeddings_dict[person_ids[i]]:
                        for emb2 in embeddings_dict[person_ids[j]]:
                            try:
                                dist = np.linalg.norm(emb1 - emb2)
                                distances.append(dist)
                                y_true.append(0)
                                negative_pairs += 1
                            except Exception as e:
                                print(f"❌ Negatif pair hesaplama hatası: {e}")
                                continue

            if not distances or not y_true:
                print("❌ Yeterli veri yok")
                return jsonify({
                    'success': False,
                    'message': 'ROC için yeterli veri yok'
                }), 400

            print(f"Toplam {len(distances)} çift hesaplandı ({positive_pairs} pozitif, {negative_pairs} negatif)")

            try:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, thresholds = roc_curve(y_true, [-d for d in distances])
                optimal_idx = np.argmax(tpr - fpr)
                optimal_threshold = abs(thresholds[optimal_idx])

                roc_auc = auc(fpr, tpr)
            except Exception as e:
                print(f"❌ ROC hesaplama hatası: {e}")
                return jsonify({
                    'success': False,
                    'message': f'ROC hesaplama hatası: {str(e)}'
                }), 500

            threshold_change = optimal_threshold - current_threshold
            
            print(f"✅ Threshold optimizasyonu tamamlandı: {optimal_threshold:.4f}")
            print(f"📊 Değişiklik: {current_threshold:.4f} → {optimal_threshold:.4f} ({threshold_change:+.4f})")
            print(f"🔍 Pairs analizi: {positive_pairs} pozitif, {negative_pairs} negatif")
            print(f"📈 ROC analizi: AUC={roc_auc:.4f}, TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")
            
            return jsonify({
                'success': True,
                'optimal_threshold': float(optimal_threshold),
                'current_threshold': float(current_threshold),
                'threshold_change': float(threshold_change),
                'total_pairs': len(distances),
                'positive_pairs': positive_pairs,
                'negative_pairs': negative_pairs,
                'message': f'Threshold optimize edildi: {current_threshold:.4f} → {optimal_threshold:.4f}'
            })
            
        except Exception as e:
            print(f"❌ Threshold optimizasyon hatası: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({
                'success': False,
                'message': f'Threshold optimizasyon hatası: {str(e)}'
            }), 500

    @app.route('/test_face_detection', methods=['POST'])
    def test_face_detection():
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return jsonify({'success': False, 'message': 'Resim verisi gerekli'}), 400

            image_data = data['image']
            if image_data.startswith('data:image'):
                image_data = image_data.split(',')[1]
            
            image_bytes = base64.b64decode(image_data)
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            if image_array is None:
                return jsonify({'success': False, 'message': 'Resim yüklenemedi'}), 400

            faces_with_landmarks = face_api.face_detector.detect_faces_with_landmarks(image_array)
            
            results = {
                'total_faces_detected': len(faces_with_landmarks),
                'faces': []
            }
            
            for i, face in enumerate(faces_with_landmarks):
                face_info = {
                    'face_id': i + 1,
                    'box': face.get('box', []),
                    'confidence': face.get('confidence', 0.0),
                    'landmarks': face.get('keypoints', {})
                }
                results['faces'].append(face_info)

            return jsonify({
                'success': True,
                'message': f'{len(faces_with_landmarks)} yüz tespit edildi',
                'results': results
            })

        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/logs', methods=['GET'])
    def get_threshold_logs():
        try:
            threshold_log_file = 'threshold_logs.json'
            if not os.path.exists(threshold_log_file):
                return jsonify({'success': True, 'logs': []})
            
            with open(threshold_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            recent_logs = logs[-100:] if len(logs) > 100 else logs
            
            return jsonify({
                'success': True,
                'total_logs': len(logs),
                'recent_logs': recent_logs
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/logs/<user_id>', methods=['GET'])
    def get_user_threshold_logs(user_id):
        try:
            threshold_log_file = 'threshold_logs.json'
            if not os.path.exists(threshold_log_file):
                return jsonify({'success': True, 'logs': []})
            
            with open(threshold_log_file, 'r', encoding='utf-8') as f:
                all_logs = json.load(f)
            
            user_logs = [log for log in all_logs if log.get('user_id') == user_id]
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'total_logs': len(user_logs),
                'logs': user_logs
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/analytics', methods=['GET'])
    def get_threshold_analytics():
        try:
            threshold_log_file = 'threshold_logs.json'
            if not os.path.exists(threshold_log_file):
                return jsonify({'success': True, 'analytics': {}})
            
            with open(threshold_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            
            if not logs:
                return jsonify({'success': True, 'analytics': {}})
            
            successful_logs = [log for log in logs if log.get('success', False)]
            failed_logs = [log for log in logs if not log.get('success', False)]
            
            thresholds = [log.get('threshold_used', 0) for log in logs]
            distances = [log.get('distance', 0) for log in logs if log.get('distance') is not None]
            confidences = [log.get('confidence', 0) for log in logs if log.get('confidence') is not None]
            
            analytics = {
                'total_events': len(logs),
                'successful_recognitions': len(successful_logs),
                'failed_recognitions': len(failed_logs),
                'success_rate': len(successful_logs) / len(logs) if logs else 0,
                'threshold_stats': {
                    'min': min(thresholds) if thresholds else 0,
                    'max': max(thresholds) if thresholds else 0,
                    'avg': sum(thresholds) / len(thresholds) if thresholds else 0,
                    'current': thresholds[-1] if thresholds else 0
                },
                'distance_stats': {
                    'min': min(distances) if distances else 0,
                    'max': max(distances) if distances else 0,
                    'avg': sum(distances) / len(distances) if distances else 0
                },
                'confidence_stats': {
                    'min': min(confidences) if confidences else 0,
                    'max': max(confidences) if confidences else 0,
                    'avg': sum(confidences) / len(confidences) if confidences else 0
                },
                'recent_events': logs[-10:] 
            }
            
            return jsonify({
                'success': True,
                'analytics': analytics
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

    @app.route('/threshold/logs/clear', methods=['POST'])
    def clear_threshold_logs():
        try:
            threshold_log_file = 'threshold_logs.json'
            if os.path.exists(threshold_log_file):
                os.remove(threshold_log_file)
            
            return jsonify({
                'success': True,
                'message': 'Threshold logları temizlendi'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500 