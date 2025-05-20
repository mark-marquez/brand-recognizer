import cv2
import os
import numpy as np
import sys
import easyocr

# Add CRAFT utility path
sys.path.append('CRAFT-pytorch')

RESULT_DIR = 'CRAFT-pytorch/result'

# Initialize EasyOCR reader (CPU only)
reader = easyocr.Reader(['en'], gpu=False)

# Test a single image
fname = 'res_img9.txt'
img_name = fname.replace('.txt', '.jpg')
img_path = os.path.join(RESULT_DIR, img_name)
txt_path = os.path.join(RESULT_DIR, fname)

image = cv2.imread(img_path)
if image is None:
    print(f"Could not load {img_path}")
    exit()

# Parse the bounding boxes
boxes = []
with open(txt_path, 'r') as f:
    for line in f:
        coords = line.strip().split(',')
        if len(coords) < 8:
            continue
        box = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
        boxes.append(box)

print(f"\n{img_name}")
for i, box in enumerate(boxes):
    poly = np.array(box).astype(np.int32)
    x, y, w, h = cv2.boundingRect(poly)

    # Clamp and filter
    h_img, w_img = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    if w < 30 or h < 30:
        continue

    crop = image[y:y+h, x:x+w]
    if crop.size == 0:
        continue

    # Auto-rotate tall crops
    if h > 1.2 * w:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

    # Run EasyOCR
    result = reader.readtext(crop, detail=0)
    if result:
        text = result[0].strip()
        print(f"  ✅ Box {i+1}: {text}")
    else:
        print(f"  Box {i+1}: [no text]")