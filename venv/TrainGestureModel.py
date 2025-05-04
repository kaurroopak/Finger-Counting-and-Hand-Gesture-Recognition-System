import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder  # 🔧 Added
import joblib

# Configuration
dataset_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\augmented_dataset"
image_size = (64, 64)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys'
}

def preprocess_image(img):
    """Match inference preprocessing exactly"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, image_size)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(resized)
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    return blurred

def extract_features(img):
    processed = preprocess_image(img)
    return hog(processed, **hog_params)

# Load data
gesture_names = [g for g in os.listdir(dataset_path)
                 if os.path.isdir(os.path.join(dataset_path, g))]

gesture_names.sort()  # 🔧 Ensure consistent class indexing
X, y = [], []

for gesture in gesture_names:
    gesture_folder = os.path.join(dataset_path, gesture)
    print(f"Processing {gesture}...")

    for filename in os.listdir(gesture_folder):
        img_path = os.path.join(gesture_folder, filename)
        img = cv2.imread(img_path)
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:  # 🔧 Added check
            print(f"Warning: Skipped invalid image {img_path}")
            continue

        features = extract_features(img)
        X.append(features)
        y.append(gesture)  # 🔧 Use gesture name, not index

# Encode labels
le = LabelEncoder()  # 🔧 Added label encoder
y = le.fit_transform(y)

X, y = np.array(X), np.array(y)

print(f"Feature vector shape: {X[0].shape}")
from collections import Counter
print("Class distribution:", Counter(y))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42  # 🔧 Ensures reproducibility
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:\n", classification_report(
    y_test, y_pred, target_names=le.classes_))  # 🔧 Use decoded class names

# Save model
joblib.dump({
    'model': model,
    'hog_params': hog_params,
    'classes': le.classes_,  # 🔧 Save actual gesture names
    'label_encoder': le
}, 'gesture_rf_hog_model.pkl')







# import os
# import cv2
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# from skimage.feature import hog
# import joblib
#
# # Set dataset path
# dataset_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\augmented_dataset"
# image_size = (64, 64)  # Must match your capture/augmentation size
#
# # HOG Parameters (add this section)
# hog_params = {
#     'orientations': 9,
#     'pixels_per_cell': (8, 8),
#     'cells_per_block': (2, 2),
#     'block_norm': 'L2-Hys',
# }
#
#
# def extract_features(img):
#     """Extract HOG features from image"""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     resized = cv2.resize(gray, image_size)
#     features = hog(resized, **hog_params)
#     return features
#
#
# # Step 1: Load Data with HOG features
# X = []  # HOG features
# y = []  # labels (gesture names)
#
# gesture_names = os.listdir(dataset_path)
# gesture_names = [g for g in gesture_names if os.path.isdir(os.path.join(dataset_path, g))]
#
# for idx, gesture in enumerate(gesture_names):
#     gesture_folder = os.path.join(dataset_path, gesture)
#     print(f"Processing {gesture}...")
#
#     for filename in os.listdir(gesture_folder):
#         img_path = os.path.join(gesture_folder, filename)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#
#         # Extract HOG features instead of raw pixels
#         features = extract_features(img)
#         X.append(features)
#         y.append(idx)
#
# X = np.array(X)
# y = np.array(y)
#
# # Step 2: Split Dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Step 3: Train RandomForest
# model = RandomForestClassifier(n_estimators=150,
#                                max_depth=15,
#                                random_state=42,
#                                class_weight='balanced',
#                                n_jobs=-1)
# model.fit(X_train, y_train)
#
# # Step 4: Evaluate
# y_pred = model.predict(X_test)
# print("\nAccuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=gesture_names))
#
# # Step 5: Save Model, HOG params, and Labels
# joblib.dump({
#     'model': model,
#     'hog_params': hog_params,
#     'classes': gesture_names
# }, 'gesture_rf_hog_model.pkl')
#
# print("\nModel, HOG parameters, and labels saved successfully!")









# import os
# import cv2
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, accuracy_score
# import joblib  # for saving the model
#
# # Set dataset path
# dataset_path = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\augmented_dataset"
# image_size = (64, 64)  # must match your capture/augmentation size
#
# # Step 1: Load Data
# X = []  # images
# y = []  # labels (gesture names)
#
# gesture_names = os.listdir(dataset_path)
# gesture_names = [g for g in gesture_names if os.path.isdir(os.path.join(dataset_path, g))]
#
# for idx, gesture in enumerate(gesture_names):
#     gesture_folder = os.path.join(dataset_path, gesture)
#     for filename in os.listdir(gesture_folder):
#         img_path = os.path.join(gesture_folder, filename)
#         img = cv2.imread(img_path)
#         if img is None:
#             continue
#         img = cv2.resize(img, image_size)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Optional: grayscale to reduce dimensionality
#         X.append(img_gray.flatten())  # Flatten image to 1D vector
#         y.append(idx)  # Label as index
#
# X = np.array(X)
# y = np.array(y)
#
# # Step 2: Split Dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#
# # Step 3: Train RandomForest
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)
#
# # Step 4: Evaluate
# y_pred = model.predict(X_test)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=gesture_names))
#
# # Step 5: Save Model and Labels
# joblib.dump(model, 'gesture_rf_model.pkl')
# np.save('gesture_labels.npy', np.array(gesture_names))
#
# print("\nModel and labels saved successfully!")






