import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np
import threading
import time

reader = easyocr.Reader(['en'], gpu=False, quantize=True)
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

# Target size for faster OCR (smaller than full frame)
target_size = (320, 240)

# Shared OCR results
scaled_results = []
ocr_lock = threading.Lock()
ocr_interval = 3  # seconds between OCR runs

def ocr_loop():
    global scaled_results
    while True:
        frame = cam.capture_array()
        resized = cv2.resize(frame, target_size)

        # Convert to grayscale for faster, more accurate OCR
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Run OCR with multiprocessing
        results = reader.readtext(gray, batch_size=1, workers=2)

        # Calculate scale factors to map boxes back to original frame
        scale_x = frame.shape[1] / target_size[0]
        scale_y = frame.shape[0] / target_size[1]

        temp_results = []
        for bbox, text, conf in results:
            scaled_pts = [(int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in bbox]
            temp_results.append((scaled_pts, text))

        with ocr_lock:
            scaled_results = temp_results

        time.sleep(ocr_interval)

# Start OCR thread
threading.Thread(target=ocr_loop, daemon=True).start()

# Main display loop
while True:
    frame = cam.capture_array()

    with ocr_lock:
        current_results = list(scaled_results)

    # Draw results on video
    for pts, text in current_results:
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Create black strip for text display
    strip_height = 60
    text_panel = np.zeros((strip_height, frame.shape[1], 3), dtype=np.uint8)

    # Concatenate all recognized text into one line
    recognized_texts = [text for _, text in current_results]
    full_line = "BRAND NAME: " + " ".join(recognized_texts)

    # Draw the full line
    cv2.putText(text_panel, full_line, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Stack frame and text panel vertically
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    combined = np.vstack((frame, text_panel))

    cv2.imshow("OCR Feed", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()