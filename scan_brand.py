from picamera2 import Picamera2
import cv2
import easyocr
import os
import numpy as np
import sys
import time

# Add CRAFT utility path
sys.path.append('CRAFT-pytorch')
RESULT_DIR = 'CRAFT-pytorch/result'
CAPTURE_DIR = 'CRAFT-pytorch/capture_only'
CRAFT_CMD = 'python3 test.py --trained_model=weights/craft_mlt_25k.pth --test_folder=capture_only --cuda=False'

# Initialize camera and EasyOCR
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

reader = easyocr.Reader(['en'], gpu=False)

# Detection interval setup
last_detect_time = 0
DETECTION_INTERVAL = 10  # seconds

print("📷 Running real-time brand detection every 10 seconds with CRAFT + EasyOCR (Ctrl+C to stop)\n")

try:
    while True:
        frame = picam2.capture_array()
        now = time.time()

        if now - last_detect_time < DETECTION_INTERVAL:
            continue  # Skip if not enough time has passed
        last_detect_time = now

        # Save frame
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        capture_path = os.path.join(CAPTURE_DIR, 'capture.jpg')
        cv2.imwrite(capture_path, frame)

        # Run CRAFT on the latest frame
        os.chdir('CRAFT-pytorch')
        os.system(CRAFT_CMD)
        os.chdir('..')

        # Load CRAFT result
        result_img_path = f'{RESULT_DIR}/res_capture.jpg'
        result_txt_path = f'{RESULT_DIR}/res_capture.txt'
        img = cv2.imread(result_img_path)

        if not os.path.exists(result_img_path) or not os.path.exists(result_txt_path) or img is None:
            print("❌ Detection failed or incomplete output.")
            continue

        with open(result_txt_path, 'r') as f:
            for i, line in enumerate(f):
                coords = line.strip().split(',')
                if len(coords) < 8:
                    continue
                box = [(int(coords[i]), int(coords[i+1])) for i in range(0, len(coords), 2)]
                x, y, w, h = cv2.boundingRect(np.array(box))
                x, y = max(0, x), max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                if w < 30 or h < 30:
                    continue
                crop = img[y:y+h, x:x+w]
                if h > 1.2 * w:
                    crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

                result = reader.readtext(crop, detail=0)
                if result:
                    text = result[0].strip()
                    if text.isupper() and len(text) >= 4:
                        print(f"✅ {text}")

except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
finally:
    picam2.close()