import os

USERS_DB_FILE = 'users.json'
KNOWN_FACES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'faces')
RECOGNITION_LOG_FILE = 'logs.json'
MOBILFACENET_MODEL_PATH = 'facenet.tflite'

os.makedirs(KNOWN_FACES_DIR, exist_ok=True) 