import json
import os
from datetime import datetime
from config import RECOGNITION_LOG_FILE

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

def log_threshold_event(user_id, name, threshold_used, distance, confidence, success, image_b64=None):
    log_entry = {
        'user_id': user_id,
        'name': name,
        'threshold_used': threshold_used,
        'distance': distance,
        'confidence': confidence,
        'success': success,
        'timestamp': datetime.now().isoformat(),
        'image': image_b64
    }
    
    threshold_log_file = 'threshold_logs.json'
    logs = []
    
    try:
        if os.path.exists(threshold_log_file):
            with open(threshold_log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
    except Exception as e:
        print(f"Threshold log okuma hatası: {e}")
        logs = []
    
    logs.append(log_entry)
    
    try:
        with open(threshold_log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
        print(f"Threshold log kaydedildi: {threshold_used:.4f} -> {success}")
    except Exception as e:
        print(f"Threshold log yazma hatası: {e}") 