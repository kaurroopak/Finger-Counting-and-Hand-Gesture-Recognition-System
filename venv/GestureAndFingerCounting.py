import cv2
import numpy as np
import joblib
from skimage.feature import hog
import HandTrackingModule as htm


class GestureRecognizer:
    def __init__(self, model_path='gesture_rf_hog_model.pkl'):
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.gesture_classes = model_data['classes']
            self.hog_params = model_data['hog_params']
            print(f"Loaded model with gestures: {self.gesture_classes}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            self.gesture_classes = []
            self.hog_params = {}

        self.image_size = (64, 64)

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.image_size)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return blurred

    def extract_hand_region(self, img, landmarks):
        if not landmarks:
            return None

        x_coords = [lm[1] for lm in landmarks]
        y_coords = [lm[2] for lm in landmarks]

        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)

        padding_x = int((max_x - min_x) * 0.2)
        padding_y = int((max_y - min_y) * 0.2)

        height, width = img.shape[:2]
        x1 = max(0, min_x - padding_x)
        y1 = max(0, min_y - padding_y)
        x2 = min(width, max_x + padding_x)
        y2 = min(height, max_y + padding_y)

        hand_img = img[y1:y2, x1:x2]
        if hand_img.size == 0:
            return None

        return hand_img

    def predict(self, img, landmarks):
        if not landmarks:
            return "Unknown"

        hand_img = self.extract_hand_region(img, landmarks)
        if hand_img is None:
            return "Unknown"

        processed = self.preprocess_image(hand_img)
        features = hog(processed, **self.hog_params)
        prediction = self.model.predict([features])[0]
        return self.gesture_classes[prediction]


def main():
    cap = cv2.VideoCapture(0)
    detector = htm.handDetector(detectionCon=0.75)
    recognizer = GestureRecognizer()
    tipIds = [4, 8, 12, 16, 20]

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=False)

        if lmList:
            fingers = []

            # Thumb
            if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)

            # Fingers
            for id in range(1, 5):
                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

            totalFingers = fingers.count(1)
            gesture = recognizer.predict(img, lmList)

            # Display count
            cv2.putText(img, f"Fingers: {totalFingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)  # black outline
            cv2.putText(img, f"Fingers: {totalFingers}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)  # white text

            # Display gesture
            cv2.putText(img, f"Gesture: {gesture}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3)  # black outline
            cv2.putText(img, f"Gesture: {gesture}", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)  # white text

        cv2.imshow("Finger + Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

