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

app = Flask(__name__)
CORS(app)

USERS_DB_FILE = 'users_db.json'
KNOWN_FACES_DIR = 'known_faces'
MOBILFACENET_MODEL_PATH = 'mobilefacenet (1).tflite'
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
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        if os.path.exists(USERS_DB_FILE):
            with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            for user in users_data:
                user_dir = os.path.join(KNOWN_FACES_DIR, user['id'])
                encodings_file = os.path.join(user_dir, 'encodings.pkl')
                if os.path.exists(encodings_file):
                    with open(encodings_file, 'rb') as ef:
                        encodings = pickle.load(ef)
                        for enc in encodings:
                            self.known_face_encodings.append(enc)
                            self.known_face_names.append(user['name'])
                            self.known_face_ids.append(user['id'])

    def add_user(self, name, images_base64, id_no=None, birth_date=None):
        try:
            user_id = str(int(datetime.now().timestamp()))
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
                return False, "Yüz tespit edilemedi", None
            if os.path.exists(USERS_DB_FILE):
                with open(USERS_DB_FILE, 'r', encoding='utf-8') as f:
                    users_data = json.load(f)
            else:
                users_data = []
            for (x, y, w, h) in faces:
                face_img = image_array[y:y+h, x:x+w]
                embedding = self.get_embedding(face_img)
                for idx, known_embedding in enumerate(self.known_face_encodings):
                    if self.compare_embeddings(embedding, known_embedding):
                        user_id = self.known_face_ids[idx]
                        user_info = next((u for u in users_data if u['id'] == user_id), None)
                        return True, user_info, {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            x, y, w, h = faces[0]
            return False, "Tanınmayan kişi", {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
        except Exception as e:
            return False, f"Hata: {str(e)}", None

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
                for file in os.listdir(user_dir):
                    os.remove(os.path.join(user_dir, file))
                os.rmdir(user_dir)
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'API çalışıyor'})

@app.route('/recognize', methods=['POST'])
def recognize_face():
    data = request.get_json()
    face_image_base64 = data.get('image')
    if not face_image_base64:
        return jsonify({'success': False, 'message': 'Resim verisi eksik'}), 400
    success, result, box = face_api.recognize_face(face_image_base64)
    if success:
        return jsonify({
            'success': True,
            'recognized': True,
            'name': result.get('name'),
            'id_no': result.get('id_no'),
            'birth_date': result.get('birth_date'),
            'box': box,
            'message': f"Kişi tanındı: {result.get('name')}"
        })
    else:
        return jsonify({
            'success': False,
            'recognized': False,
            'name': None,
            'id_no': None,
            'birth_date': None,
            'box': box,
            'message': result
        })

@app.route('/add_user', methods=['POST'])
def add_user():
    data = request.get_json()
    name = data.get('name')
    images = data.get('images', [])
    id_no = data.get('id_no')
    birth_date = data.get('birth_date')
    if not name or not images or not id_no or not birth_date:
        return jsonify({'success': False, 'message': 'Tüm bilgiler ve en az 1 resim gerekli'}), 400
    success, message = face_api.add_user(name, images, id_no, birth_date)
    return jsonify({'success': success, 'message': message})

@app.route('/delete_user/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    success, message = face_api.delete_user(user_id)
    return jsonify({'success': success, 'message': message})

@app.route('/users', methods=['GET'])
def get_users():
    users = face_api.get_users()
    return jsonify({'success': True, 'users': users})

@app.route('/photo/<user_id>/<int:index>.jpg', methods=['GET'])
def serve_profile_photo(user_id, index):
    filename = f"{index}.jpg"
    user_dir = os.path.join(KNOWN_FACES_DIR, user_id)
    file_path = os.path.join(user_dir, filename)
    if os.path.exists(file_path):
        return send_from_directory(user_dir, filename)
    else:
        return jsonify({'success': False, 'message': 'Fotoğraf bulunamadı'}), 404

if __name__ == '__main__':
    print("Yüz Tanıma API başlatılıyor...")
    app.run(host='0.0.0.0', port=5000, debug=True)
