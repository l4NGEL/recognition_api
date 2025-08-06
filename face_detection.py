import cv2
import numpy as np
from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        self.mtcnn_detector = MTCNN()

    def detect_faces(self, image_array):
        try:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            faces = self.mtcnn_detector.detect_faces(rgb_image)
            
            opencv_faces = []
            for face in faces:
                x, y, w, h = face['box']
                confidence = face['confidence']
                
                if confidence > 1.0:
                    opencv_faces.append((x, y, w, h))
            
            return np.array(opencv_faces)
        except Exception as e:
            print(f"MTCNN yüz tespit hatası: {e}")
            return np.array([])

    def detect_faces_with_rotation(self, image_array):
        faces = self.detect_faces(image_array)
        if len(faces) > 0:
            return faces, 0

        rotated_90 = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
        faces = self.detect_faces(rotated_90)
        if len(faces) > 0:
            return faces, 90

        rotated_180 = cv2.rotate(image_array, cv2.ROTATE_180)
        faces = self.detect_faces(rotated_180)
        if len(faces) > 0:
            return faces, 180

        rotated_270 = cv2.rotate(image_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        faces = self.detect_faces(rotated_270)
        if len(faces) > 0:
            return faces, 270

        return [], 0

    def detect_faces_with_landmarks(self, image_array):
        try:
            rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            faces = self.mtcnn_detector.detect_faces(rgb_image)
            
            print(f"MTCNN ile {len(faces)} yüz tespit edildi")
            
            if len(faces) == 0:
                print("MTCNN yüz bulamadı, Haar Cascade deneniyor...")
                faces = self.detect_faces_with_haar_cascade(image_array)
                
                converted_faces = []
                for face in faces:
                    x, y, w, h = face
                    landmarks = {
                        'left_eye': (x + w//4, y + h//3),
                        'right_eye': (x + 3*w//4, y + h//3),
                        'nose': (x + w//2, y + h//2),
                        'mouth_left': (x + w//4, y + 2*h//3),
                        'mouth_right': (x + 3*w//4, y + 2*h//3)
                    }
                    converted_faces.append({
                        'box': [x, y, w, h],
                        'keypoints': landmarks,
                        'confidence': 0.8
                    })
                faces = converted_faces
                print(f"Haar Cascade ile {len(faces)} yüz tespit edildi")

            return faces
        except Exception as e:
            print(f"Yüz tespit hatası: {e}")
            return []

    def detect_faces_with_haar_cascade(self, image_array):
        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            print(f"Haar Cascade ile {len(faces)} yüz tespit edildi")
            return faces
            
        except Exception as e:
            print(f"Haar Cascade hatası: {e}")
            return []

    def align_face(self, image_array, landmarks):
        try:
            if isinstance(landmarks, dict):
                left_eye = landmarks.get('left_eye', landmarks.get('leftEye'))
                right_eye = landmarks.get('right_eye', landmarks.get('rightEye'))

                if left_eye is None or right_eye is None:
                    print("Göz landmark'ları bulunamadı, orijinal görüntü döndürülüyor")
                    return image_array

                eye_angle = np.degrees(np.arctan2(
                    right_eye[1] - left_eye[1],
                    right_eye[0] - left_eye[0]
                ))

                if abs(eye_angle) < 5:
                    print("Açı çok küçük, hizalama yapılmıyor")
                    return image_array

                eye_center = (
                    int((left_eye[0] + right_eye[0]) / 2),
                    int((left_eye[1] + right_eye[1]) / 2)
                )

                height, width = image_array.shape[:2]
                diagonal = int(np.sqrt(width**2 + height**2))

                new_width = diagonal
                new_height = diagonal

                new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

                x_offset = (new_width - width) // 2
                y_offset = (new_height - height) // 2
                new_image[y_offset:y_offset+height, x_offset:x_offset+width] = image_array

                new_center = (x_offset + eye_center[0], y_offset + eye_center[1])

                rotation_matrix = cv2.getRotationMatrix2D(new_center, eye_angle, 1.0)

                aligned_face = cv2.warpAffine(new_image, rotation_matrix, (new_width, new_height))

                aligned_face = self.remove_black_borders(aligned_face)

                return aligned_face
            else:
                print("Landmark formatı tanınmadı, orijinal görüntü döndürülüyor")
                return image_array

        except Exception as e:
            print(f"Yüz hizalama hatası: {e}")
            return image_array

    def remove_black_borders(self, image):
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            kernel = np.ones((5,5), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                min_size = 30
                if w < min_size or h < min_size:
                    return image

                margin = 2
                x = max(0, x - margin)
                y = max(0, y - margin)
                w = min(image.shape[1] - x, w + 2 * margin)
                h = min(image.shape[0] - y, h + 2 * margin)

                cropped = image[y:y+h, x:x+w]

                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    return cropped

            return image
        except Exception as e:
            print(f"Siyah kenar kaldırma hatası: {e}")
            return image 