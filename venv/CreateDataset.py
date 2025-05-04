import cv2
import mediapipe as mp
import os
import time

# List of gestures to capture
gestures = ['call_me', 'fingers_crossed', 'fist', 'okay', 'peace', 'rock', 'stop', 'thumbs_up', 'up']
dataset_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-and-Hand-Gesture-Recognition-System\captured_dataset"
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

for gesture in gestures:
    print(f"\nPreparing to capture '{gesture}' gesture.")
    print("Press 's' to start capturing...")

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
                time.sleep(0.05)

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