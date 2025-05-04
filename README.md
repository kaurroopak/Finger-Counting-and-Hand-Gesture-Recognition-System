# ✋ Hand Gesture + Finger Counting Recognition System  
This project combines real-time hand gesture recognition and finger counting using a webcam, powered by **MediaPipe**, **OpenCV**, and a **scikit-learn** gesture classifier (trained on HOG features). It's built in Python and works efficiently for both tracking finger positions and identifying custom hand gestures.

✅ The gesture recognition model achieves **~98% accuracy** using a **RandomForestClassifier** from the `scikit-learn` library, trained on HOG (Histogram of Oriented Gradients) features.

## 🔧 Features  
✋ **Accurate Finger Counting** – Detects how many fingers are held up in real-time.  
🤙 **Gesture Recognition** – Recognizes custom hand gestures like `thumbs_up`, `call_me`, `fingers_crossed`, etc.  
🎥 **Live Webcam Input** – Processes video input from your webcam.  
🧠 **Machine Learning Classifier** – Uses HOG + PCA + **Random Forest (scikit-learn)** for gesture recognition.  
🎯 **Clean UI Overlay** – Neat, non-blocking overlay of finger count and gesture name.


## 📦 Dependencies
Install all dependencies via pip:

     pip install opencv-python mediapipe scikit-learn scikit-image numpy


Also, make sure you have:
- Python 3.8+
- Trained model: gesture_model.pkl
- Gesture class labels embedded in the pickle file
- A module file: HandTrackingModule.py (based on MediaPipe)


## 🚀 How to Run

     python GestureAndFingerCounting.py

Press q to exit the webcam window.


## 📁 Project Structure
.

├── GestureAndFingerCounting.py     (Main file to run the system)

├── HandTrackingModule.py           (Contains MediaPipe-based hand detector)

├── gesture_model.pkl               (Pre-trained gesture recognition model)

├── ProcessDataset.py               (Script to generate features for training)

├── TrainGestureModel.py            (Model training script (HOG + PCA + Classifier))

├── gesture_data/                   ((Optional) Folder containing gesture dataset)

└── README.md                       (Project description)


## 💡 How it Works
**Hand Tracking:** MediaPipe detects hand landmarks (21 points).

**Finger Counting:** Uses landmark indices to count raised fingers.

**Gesture Recognition:** Extracts hand region, computes HOG features, and classifies using the trained model.

**Output Display:** Finger count and gesture label are rendered on screen in real-time.


## 📸 Example Output
![image](https://github.com/user-attachments/assets/18be22df-2218-40a6-8ccb-9bd6ecb692f7)


## 🙌 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change or improve.

## 📄 License
This  project is licensed under the MIT License.

