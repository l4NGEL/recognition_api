import json
import os
from datetime import datetime
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import numpy as np
import random
import cv2

def apply_brightness_contrast(image, brightness_factor=1.0, contrast_factor=1.0):
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness_factor)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast_factor)
    return image

def apply_noise(image, noise_factor=0.02):
    img_array = np.array(image)
    noise = np.random.normal(0, noise_factor * 255, img_array.shape)
    noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def apply_blur(image, blur_factor=1.0):
    return image.filter(ImageFilter.GaussianBlur(radius=blur_factor))

def apply_sharpening(image, sharpness_factor=1.5):
    enhancer = ImageEnhance.Sharpness(image)
    return enhancer.enhance(sharpness_factor)

def apply_geometric_transform(image, rotation_range=10, scale_range=0.1):
    angle = random.uniform(-rotation_range, rotation_range)
    scale = random.uniform(1 - scale_range, 1 + scale_range)
    
    if abs(angle) > 1:
        image = image.rotate(angle, expand=True)
    
    if abs(scale - 1.0) > 0.05:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return image

def apply_lighting_simulation(image, lighting_factor=0.3):
    img_array = np.array(image)
    
    height, width = img_array.shape[:2]
    y, x = np.ogrid[:height, :width]
    
    center_x = random.uniform(0, width)
    center_y = random.uniform(0, height)
    
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    max_distance = np.sqrt(width**2 + height**2)
    
    lighting_mask = 1 - (distance / max_distance) * lighting_factor
    lighting_mask = np.clip(lighting_mask, 0.7, 1.3)
    
    if len(img_array.shape) == 3:
        lighting_mask = np.stack([lighting_mask] * 3, axis=2)
    
    lighted_img = np.clip(img_array * lighting_mask, 0, 255).astype(np.uint8)
    return Image.fromarray(lighted_img)

def create_augmented_versions(image, num_versions=8):
    augmented_images = []
    
    for i in range(num_versions):
        aug_img = image.copy()
        
        if random.random() < 0.7:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            aug_img = apply_brightness_contrast(aug_img, brightness, contrast)
        
        if random.random() < 0.5:
            noise_factor = random.uniform(0.01, 0.03)
            aug_img = apply_noise(aug_img, noise_factor)
        
        if random.random() < 0.3:
            blur_factor = random.uniform(0.5, 1.5)
            aug_img = apply_blur(aug_img, blur_factor)
        
        if random.random() < 0.4:
            sharpness = random.uniform(1.2, 1.8)
            aug_img = apply_sharpening(aug_img, sharpness)
        
        if random.random() < 0.6:
            aug_img = apply_geometric_transform(aug_img)
        
        if random.random() < 0.4:
            lighting = random.uniform(0.2, 0.4)
            aug_img = apply_lighting_simulation(aug_img, lighting)
        
        augmented_images.append(aug_img)
    
    return augmented_images

def get_augmentation_stats():
    return {
        'augmentation_types': [
            'brightness_contrast',
            'noise',
            'blur',
            'sharpening',
            'geometric_transform',
            'lighting_simulation'
        ],
        'default_versions_per_image': 8,
        'supported_transformations': [
            'Parlaklık ve kontrast ayarlama',
            'Gürültü ekleme',
            'Bulanıklık uygulama',
            'Keskinleştirme',
            'Geometrik dönüşümler (rotasyon, ölçekleme)',
            'Aydınlatma simülasyonu'
        ]
    }

def auto_rotate_image(image):
    try:
        if hasattr(image, '_getexif') and image._getexif() is not None:
            exif = image._getexif()
            if exif is not None:
                orientation = exif.get(274)
                if orientation is not None:
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
    try:
        if not os.path.exists('users.json'):
            print("users_db.json dosyası bulunamadı, yeni dosya oluşturuluyor...")
            with open('users.json', 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            return True

        with open('users.json', 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"Dosya boyutu: {len(content)} karakter")

        try:
            first_bracket = content.index('[')
            last_bracket = content.rindex(']')
            valid_json = content[first_bracket:last_bracket+1]

            users = json.loads(valid_json)
            print(f"Geçerli kullanıcı sayısı: {len(users)}")

            with open('users.json', 'w', encoding='utf-8') as f:
                json.dump(users, f, ensure_ascii=False, indent=2)

            print("JSON dosyası başarıyla temizlendi")
            return True

        except ValueError as e:
            print(f"JSON parse hatası: {e}")
            with open('users.json', 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            print("Yeni JSON dosyası oluşturuldu")
            return True

    except Exception as e:
        print(f"JSON düzeltme hatası: {e}")
        return False 