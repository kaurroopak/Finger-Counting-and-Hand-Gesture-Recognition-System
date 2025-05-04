import cv2
import numpy as np
import mediapipe as mp
import joblib
from skimage.feature import hog

# Load model and HOG parameters
model_data = joblib.load('gesture_rf_hog_model.pkl')
model = model_data['model']
hog_params = model_data['hog_params']
gesture_names = model_data['classes']

# Image settings
image_size = (64, 64)

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Preprocessing and feature extraction
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

def extract_features(img):
    processed = preprocess_image(img)
    return hog(processed, **hog_params)

# Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min, x_max, y_max = w, h, 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            margin = 20
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            hand_img = frame[y_min:y_max, x_min:x_max]

            try:
                features = extract_features(hand_img)
                prediction = model.predict([features])[0]
                label = gesture_names[prediction]

                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print("Feature extraction error:", e)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# import cv2
# import pickle
# import numpy as np
# from skimage.feature import hog
# import HandTrackingModule as htm
#
#
# class GestureRecognizer:
#     def __init__(self, model_path=r'C:\Users\DELL\PycharmProjects\Finger-Counting-System\gesture_model.pkl'):
#         # Load trained model
#         try:
#             with open(model_path, 'rb') as f:
#                 self.model_data = pickle.load(f)
#                 self.model = self.model_data['model']
#                 self.gesture_classes = self.model_data['gesture_classes']
#                 print(f"Loaded model with gesture classes: {self.gesture_classes}")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#             self.model = None
#             self.gesture_classes = []
#             self.model_data = {}
#
#     def extract_hand_region(self, img, landmarks):
#         """Extract hand region with better padding"""
#         if not landmarks:
#             return None
#
#         x_coords = [lm[1] for lm in landmarks]
#         y_coords = [lm[2] for lm in landmarks]
#
#         min_x, max_x = min(x_coords), max(x_coords)
#         min_y, max_y = min(y_coords), max(y_coords)
#
#         # Increased padding to 25%
#         padding_x = int((max_x - min_x) * 0.25)
#         padding_y = int((max_y - min_y) * 0.25)
#
#         height, width = img.shape[:2]
#         x1 = max(0, min_x - padding_x)
#         y1 = max(0, min_y - padding_y)
#         x2 = min(width, max_x + padding_x)
#         y2 = min(height, max_y + padding_y)
#
#         hand_img = img[y1:y2, x1:x2]
#         if hand_img.size == 0:
#             return None
#
#         return hand_img
#
#     def preprocess_hand_image(self, hand_img):
#         """Preprocess hand image with standardization"""
#         gray = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
#         gray = cv2.resize(gray, (96, 96))  # Smaller size
#
#         # Normalize
#         gray = gray / 255.0
#
#         # Histogram equalization to standardize brightness
#         gray = (gray * 255).astype(np.uint8)
#         gray = cv2.equalizeHist(gray)
#         gray = gray / 255.0
#
#         return gray
#
#     def predict(self, img, landmarks):
#         """Predict gesture using HOG features and SVM"""
#         if not landmarks:
#             return "Unknown"
#
#         # Extract hand region
#         hand_img = self.extract_hand_region(img, landmarks)
#         if hand_img is None:
#             return "Unknown"
#
#         # Preprocess hand image
#         gray = self.preprocess_hand_image(hand_img)
#
#         # HOG Feature extraction
#         hog_params = {
#             'orientations': 9,
#             'pixels_per_cell': (8, 8),  # Smaller cells
#             'cells_per_block': (2, 2),
#             'block_norm': 'L2-Hys'
#         }
#         features = hog((gray * 255).astype('uint8'), **hog_params)
#
#         # Apply PCA if used
#         if self.model_data.get('use_pca', False) and self.model_data.get('pca') is not None:
#             features = self.model_data['pca'].transform([features])
#         else:
#             features = [features]
#
#         # Predict gesture
#         prediction = self.model.predict(features)[0]
#         gesture = self.gesture_classes[prediction]
#
#         return gesture
#
#
# def run_gesture_recognition():
#     """Run standalone gesture recognition"""
#     cap = cv2.VideoCapture(0)
#     detector = htm.handDetector(detectionCon=0.75)
#     recognizer = GestureRecognizer()
#
#     while True:
#         success, img = cap.read()
#         if not success:
#             break
#
#         img = cv2.flip(img, 1)  # Mirror image
#         img = detector.findHands(img)
#         lmList = detector.findPosition(img, draw=False)
#
#         gesture_text = "No hand detected"
#         if lmList:
#             gesture_text = recognizer.predict(img, lmList)
#
#         # Draw hand landmarks
#         if lmList:
#             for lm in lmList:
#                 cv2.circle(img, (lm[1], lm[2]), 5, (0, 255, 0), cv2.FILLED)
#
#         # Display recognized gesture
#         cv2.rectangle(img, (20, 20), (350, 80), (255, 0, 0), cv2.FILLED)
#         cv2.putText(img, f"Gesture: {gesture_text}", (30, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#
#         cv2.imshow("Gesture Recognition", img)
#
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
#     cv2.waitKey(1)
#
#
# if __name__ == "__main__":
#     run_gesture_recognition()
#
