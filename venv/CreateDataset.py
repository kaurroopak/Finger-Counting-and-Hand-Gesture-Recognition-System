import cv2
import mediapipe as mp
import os
import time

# List of gestures to capture
gestures = ['call_me', 'fingers_crossed', 'fist', 'okay', 'peace', 'rock', 'stop', 'thumbs_up', 'up']
dataset_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\captured_dataset"
os.makedirs(dataset_path, exist_ok=True)

# Create folders for each gesture
for gesture in gestures:
    os.makedirs(os.path.join(dataset_path, gesture), exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
target_count = 200  # Images per gesture

# Loop through each gesture
for gesture in gestures:
    print(f"\nPreparing to capture '{gesture}' gesture.")
    print("Press 's' to start capturing...")

    # Wait for user to press 's'
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"Ready for: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 's' to start", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Hand Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    print(f"Capturing '{gesture}' gesture. Hold steady...")
    count = 0
    save_dir = os.path.join(dataset_path, gesture)

    # Start capturing images
    while cap.isOpened() and count < target_count:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Find bounding box
                x_min, y_min, x_max, y_max = w, h, 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min = min(x, x_min)
                    y_min = min(y, y_min)
                    x_max = max(x, x_max)
                    y_max = max(y, y_max)

                margin = 20
                x_min = max(x_min - margin, 0)
                y_min = max(y_min - margin, 0)
                x_max = min(x_max + margin, w)
                y_max = min(y_max + margin, h)

                # Crop and save image
                hand_img = frame[y_min:y_max, x_min:x_max]
                hand_img = cv2.resize(hand_img, (64, 64))
                save_path = os.path.join(save_dir, f'{count}.jpg')
                cv2.imwrite(save_path, hand_img)
                count += 1
                time.sleep(0.05)  # Delay to reduce duplicates

                # Draw landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Show capture progress
        cv2.putText(frame, f"Captured: {count}/{target_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture Hand Gesture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= target_count:
            break

print("\nDataset capture complete.")
cap.release()
cv2.destroyAllWindows()



# # GestureDataProcessor.py
# import cv2
# import os
# import numpy as np
# import HandTrackingModule as htm
#
#
# class GestureDatasetProcessor:
#     def __init__(self, dataset_path, filtered_path=None):
#         self.dataset_path = dataset_path
#         self.filtered_path = filtered_path
#         # Use lower detection confidence and static image mode
#         self.detector = htm.handDetector(detectionCon=0.1, mode=True)
#         self.gesture_classes = [d for d in os.listdir(dataset_path)
#                                 if os.path.isdir(os.path.join(dataset_path, d))]
#
#     def preprocess_image(self, img):
#         if img is None:
#             return None
#
#         # Resize to make hands appear larger
#         img = cv2.resize(img, (640, 480))
#
#         # Apply stronger contrast enhancement
#         img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
#
#         # Try adaptive histogram equalization
#         lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
#         l, a, b = cv2.split(lab)
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         cl = clahe.apply(l)
#         limg = cv2.merge((cl, a, b))
#         img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
#
#         return img
#
#     def extract_features(self, landmarks, img_shape):
#         """Extract normalized features from landmarks"""
#         if not landmarks or len(landmarks) < 21:
#             return None
#
#         # Use wrist as reference point
#         base_x, base_y = landmarks[0][1], landmarks[0][2]
#
#         # Extract features
#         features = []
#         for lm in landmarks:
#             # Normalize by image dimensions
#             x_rel = (lm[1] - base_x) / img_shape[1]
#             y_rel = (lm[2] - base_y) / img_shape[0]
#             features.extend([x_rel, y_rel])
#
#         return features
#
#     def create_dataset(self):
#         """Process images and create dataset for training"""
#         data = []
#         labels = []
#         successful_images = 0
#         total_images = 0
#
#         print("Creating gesture recognition dataset...")
#
#         # Create filtered directory if specified
#         if self.filtered_path:
#             os.makedirs(self.filtered_path, exist_ok=True)
#
#         # Process each gesture class
#         for gesture_idx, gesture_class in enumerate(self.gesture_classes):
#             gesture_dir = os.path.join(self.dataset_path, gesture_class)
#
#             # Create filtered subdirectory
#             if self.filtered_path:
#                 filtered_gesture_dir = os.path.join(self.filtered_path, gesture_class)
#                 os.makedirs(filtered_gesture_dir, exist_ok=True)
#
#             print(f"Processing {gesture_class} images...")
#
#             # Get all images
#             image_files = [f for f in os.listdir(gesture_dir)
#                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
#
#             for img_file in image_files:
#                 total_images += 1
#                 img_path = os.path.join(gesture_dir, img_file)
#
#                 # Load and preprocess
#                 img = cv2.imread(img_path)
#                 img = self.preprocess_image(img)
#
#                 if img is None:
#                     continue
#
#                 # Detect hand and get landmarks
#                 img = self.detector.findHands(img)
#                 landmarks = self.detector.findPosition(img, draw=False)
#
#                 if landmarks and len(landmarks) >= 21:
#                     # Extract features
#                     features = self.extract_features(landmarks, img.shape)
#
#                     if features:
#                         data.append(features)
#                         labels.append(gesture_idx)
#                         successful_images += 1
#
#                         # Save successful detection to filtered directory
#                         if self.filtered_path:
#                             filtered_img_path = os.path.join(
#                                 self.filtered_path, gesture_class, img_file
#                             )
#                             cv2.imwrite(filtered_img_path, img)
#
#                 # Show progress
#                 if total_images % 10 == 0:
#                     print(
#                         f"Processed {total_images} images. Success rate: {successful_images / max(1, total_images):.2%}")
#
#         print(f"Dataset creation complete. Successfully processed {successful_images}/{total_images} images.")
#
#         if successful_images == 0:
#             print("WARNING: No hands were detected in any images. Check your dataset or adjust detection parameters.")
#             return None, None, self.gesture_classes
#
#         return data, labels, self.gesture_classes
#
#
# # Main execution if run directly
# if __name__ == "__main__":
#     dataset_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\gesture_dataset"
#     filtered_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\filtered_gesture_dataset"
#
#     processor = GestureDatasetProcessor(dataset_path, filtered_path)
#     data, labels, gestures = processor.create_dataset()
#
#     # Save the processed data for training
#     if data and labels:
#         import pickle
#
#         with open('processed_data.pickle', 'wb') as f:
#             pickle.dump({
#                 'data': data,
#                 'labels': labels,
#                 'gestures': gestures
#             }, f)
#         print("Processed data saved to 'processed_data.pickle'")
