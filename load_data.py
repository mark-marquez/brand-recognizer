import pandas as pd
import cv2
import os
import imgaug.augmenters as iaa
import numpy as np

IMAGE_DIR = 'data/images'
LABEL_FILE = 'data/labels.csv'
IMG_SIZE = (512, 512)

# Load labels
df = pd.read_csv(LABEL_FILE)

# Augmentation pipeline
augmenters = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Affine(rotate=(-15, 15), scale=(0.9, 1.1)),
    iaa.Multiply((0.8, 1.2)),
    iaa.AdditiveGaussianNoise(scale=(5, 15))
])

def pad_to_square(img, color=(128, 128, 128)):
    h, w = img.shape[:2]
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

data = []

for _, row in df.iterrows():
    path = os.path.join(IMAGE_DIR, row['filename'])
    img = cv2.imread(path)
    if img is None:
        continue

    img = pad_to_square(img)
    img = cv2.resize(img, IMG_SIZE)
    label = row['brand'].strip().lower()

    # Add original
    data.append((img, label))

    # Add 4 augmentations
    for _ in range(4):
        aug_img = augmenters(image=img)
        data.append((aug_img, label))

print(f"Loaded {len(data)} total samples (original + augmented).")