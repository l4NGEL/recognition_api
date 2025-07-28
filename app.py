from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
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

app = Flask(__name__)
CORS(app)

USERS_DB_FILE = 'users_db.json'
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'known_faces')
RECOGNITION_LOG_FILE = 'recognition_logs.json'
MOBILFACENET_MODEL_PATH = 'mobilefacenet.tflite'
HAAR_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

class FaceRecognitionAPI:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.mobilfacenet_interpreter = self.load_mobilfacenet_model(MOBILFACENET_MODEL_PATH)
        self.input_details = self.mobilfacenet_interpreter.get_input_details()
        self.output_details = self.mobilfacenet_interpreter.get_output_details()
        self.face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        self.load_known_faces()

    def load_mobilfacenet_model(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def get_embedding(self, face_img):
        img = cv2.resize(face_img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype('float32') / 255.0
        img = np.expand_dims(img, axis=0)
        self.mobilfacenet_interpreter.set_tensor(self.input_details[0]['index'], img)
        self.mobilfacenet_interpreter.invoke()
        embedding = self.mobilfacenet_interpreter.get_tensor(self.output_details[0]['index'])
        return embedding.flatten()

    def compare_embeddings(self, emb1, emb2, threshold=0.8):
        dist = np.linalg.norm(emb1 - emb2)
        return dist < threshold

    def detect_faces(self, image_array):
        gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        return faces

    def load_known_faces(self):
        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_ids.clear()
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            for user in users_data:
                user_dir = os.path.join(KNOWN_FACES_DIR, user['id_no'])
                encodings_file = os.path.join(user_dir, 'encodings.pkl')
                if os.path.exists(encodings_file):
                    with open(encodings_file, 'rb') as ef:
                        encodings = pickle.load(ef)
                        for enc in encodings:
                            self.known_face_encodings.append(enc)
                            self.known_face_names.append(user['name'])
                            self.known_face_ids.append(user['id_no'])

    def add_user(self, name, images_base64, id_no=None, birth_date=None):
        try:
            # id_no 11 hane kontrolü
            if not id_no or not str(id_no).isdigit() or len(str(id_no)) != 11:
                return False, "Kimlik numarası 11 haneli olmalıdır."
            if os.path.exists(USERS_DB_FILE):
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
                if any(user['id_no'] == id_no for user in users_data):
                    return False, "Bu kimlik numarasına sahip kullanıcı zaten kayıtlı."
            user_id = str(id_no)
            user_dir = os.path.join(KNOWN_FACES_DIR, user_id)
            os.makedirs(user_dir, exist_ok=True)
            encodings = []
            for idx, img_b64 in enumerate(images_base64):
                image_data = base64.b64decode(img_b64.split(',')[1] if ',' in img_b64 else img_b64)
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image)
                faces = self.detect_faces(image_array)
                if len(faces) == 0:
                    continue
                for (x, y, w, h) in faces:
                    face_img = image_array[y:y+h, x:x+w]
                    embedding = self.get_embedding(face_img)
                    encodings.append(embedding)
                    face_pil = Image.fromarray(face_img)
                    face_pil.save(os.path.join(user_dir, f'{idx+1}.jpg'))
            if not encodings:
                return False, "Hiçbir resimde yüz tespit edilemedi"
            with open(os.path.join(user_dir, 'encodings.pkl'), 'wb') as ef:
                pickle.dump(encodings, ef)
            self.save_users_db(user_id, name, id_no, birth_date)
            for enc in encodings:
                self.known_face_encodings.append(enc)
                self.known_face_names.append(name)
                self.known_face_ids.append(user_id)
            log_recognition(user_id=user_id, name=name, result='success', image_b64=None, action='kayıt')
            return True, f"Kullanıcı {name} başarıyla eklendi"
        except Exception as e:
            return False, f"Hata: {str(e)}"

    def save_users_db(self, user_id, name, id_no=None, birth_date=None):
        users_data = []
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
        users_data.append({
            'id': user_id,
            'name': name,
            'id_no': id_no,
            'birth_date': birth_date,
            'created_at': datetime.now().isoformat()
        })
        with open(USERS_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(users_data, f, ensure_ascii=False, indent=2)

    def recognize_face(self, face_image_base64):
        try:
            image_data = base64.b64decode(face_image_base64.split(',')[1] if ',' in face_image_base64 else face_image_base64)
            image = Image.open(io.BytesIO(image_data))
            image_array = np.array(image)
            faces = self.detect_faces(image_array)
            if len(faces) == 0:
                return False, "Yüz tespit edilemedi"
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            for (x, y, w, h) in faces:
                face_img = image_array[y:y+h, x:x+w]
                embedding = self.get_embedding(face_img)
                for idx, known_embedding in enumerate(self.known_face_encodings):
                    if self.compare_embeddings(embedding, known_embedding):
                        user_id = self.known_face_ids[idx]
                        user_info = next((u for u in users_data if u['id'] == user_id), None)
                        
                        # Tanınan kişinin fotoğrafını kaydet
                        self.save_recognition_photo(user_id, image_array, face_img)
                        
                        log_recognition(user_id, user_info['name'], 'success', face_image_base64, action='tanıma')
                        return True, user_info
            log_recognition(None, None, 'fail', face_image_base64, action='tanıma')
            return False, "Tanınmayan kişi"
        except Exception as e:
            return False, f"Hata: {str(e)}"

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
            
            # Tam resmi kaydet
            full_image_pil = Image.fromarray(full_image)
            full_image_path = os.path.join(timestamp_dir, 'full_image.jpg')
            full_image_pil.save(full_image_path, quality=95)
            
            # Yüz kısmını kaydet
            face_pil = Image.fromarray(face_img)
            face_image_path = os.path.join(timestamp_dir, 'face_crop.jpg')
            face_pil.save(face_image_path, quality=95)
            
            # Tanıma bilgilerini JSON dosyasına kaydet
            recognition_info = {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'files': {
                    'full_image': 'full_image.jpg',
                    'face_crop': 'face_crop.jpg'
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
        if exists:
            files = os.listdir(recognition_logs_dir)
        return jsonify({
            'success': True, 
            'id_no': id_no,
            'current_directory': current_dir,
            'directory_exists': exists,
            'directory_path': recognition_logs_dir,
            'absolute_path': abs_path,
            'files': files
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
                            }
                        })
        
        # Tarihe göre sırala (en yeni en üstte)
        logs.sort(key=lambda x: x['datetime'], reverse=True)
        
        return jsonify({'success': True, 'logs': logs})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Hata: {str(e)}'}), 500

if __name__ == '__main__':
    print("Yüz Tanıma API'si başlatılıyor...")
    app.run(host='0.0.0.0', port=5000, debug=True)
