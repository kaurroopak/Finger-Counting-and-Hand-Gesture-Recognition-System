import cv2
import os
import numpy as np
import random

input_root = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\captured_dataset"
output_root = r"C:\Users\DELL\PycharmProjects\Finger-Counting-System\augmented_dataset"

# --- Augmentation Functions ---
def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

def adjust_brightness(img, factor):
    return cv2.convertScaleAbs(img, alpha=factor, beta=0)

def zoom_image(img, zoom_factor):
    h, w = img.shape[:2]
    new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
    resized = cv2.resize(img, (new_w, new_h))
    if zoom_factor < 1:
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        return cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_CONSTANT)
    else:
        crop_h = (new_h - h) // 2
        crop_w = (new_w - w) // 2
        return resized[crop_h:crop_h+h, crop_w:crop_w+w]

def add_gaussian_noise(img):
    row, col, ch = img.shape
    mean = 0
    sigma = 15
    gauss = np.random.normal(mean, sigma, (row, col, ch)).astype('uint8')
    noisy = cv2.add(img, gauss)
    return noisy

def blur_image(img, k=3):
    return cv2.GaussianBlur(img, (k, k), 0)

def increase_contrast(img, factor=1.5):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8,8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def sharpen_image(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

# --- Process Each Gesture Folder ---
for gesture in os.listdir(input_root):
    input_folder = os.path.join(input_root, gesture)
    output_folder = os.path.join(output_root, gesture)
    os.makedirs(output_folder, exist_ok=True)

    count = 0
    for filename in os.listdir(input_folder):
        filepath = os.path.join(input_folder, filename)
        img = cv2.imread(filepath)

        if img is None:
            continue

        augmentations = [
            img,
            cv2.flip(img, 1),
            rotate_image(img, -15),
            rotate_image(img, 15),
            adjust_brightness(img, 1.2),
            adjust_brightness(img, 0.7),
            zoom_image(img, 1.2),
            zoom_image(img, 0.8),
            blur_image(img),
            add_gaussian_noise(img),
            increase_contrast(img),
            sharpen_image(img),
        ]

        for aug_img in augmentations:
            save_path = os.path.join(output_folder, f'{count}.jpg')
            cv2.imwrite(save_path, aug_img)
            count += 1

    print(f"[{gesture}] Augmentation complete: {count} images saved in '{output_folder}'.")

print("✅ All gestures augmented with rich transformations.")
