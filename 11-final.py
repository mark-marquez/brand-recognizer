# OPTIMIZATIONS
# 1. images resized
# 2. ocr throttled
# 3. double threaded 
# 4. display brand 
# 5. images grayscaled
# 6. bluebox only
# 7. easyocr recognition model quantized
# 8. easyocr allowlist
# 9. YOLO input resolution reduced
# 10. YOLO detector throttled
# 11. YOLO model quantized
# 12. Cascading - first YOLO logo detection, then OCR


import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from queue import Queue

# Initialize models
yolo_model = YOLO('best-june-08_int8.tflite', task='detect')
# yolo_model = YOLO('best-june-08.pt', task='detect')
reader = easyocr.Reader(['en'], gpu=False, quantize=True)

# Initialize camera
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

# Resize parameters
YOLO_WIDTH = 320   # Target width for YOLO processing (faster)
OCR_WIDTH = 320    # Target width for OCR processing

# --- STEP 1: Add variables for YOLO throttling ---
YOLO_INTERVAL = 3  # Run YOLO detection only every 3 frames
OCR_INTERVAL = 5   # Only send frames to OCR every 5 frames (must be >= YOLO_INTERVAL)
last_yolo_boxes = [] # To store the boxes from the last detection

# Shared OCR results
ocr_queue = Queue(maxsize=10)
current_ocr_results = []  # List of (pts, text, conf) for current frame
ocr_lock = threading.Lock()

def ocr_loop():
    """Background thread for OCR processing"""
    while True:
        try:
            # Get next region to process (blocks if queue is empty)
            roi, x1, y1 = ocr_queue.get(timeout=1)
            
            if roi.size > 0:
                # Resize ROI for faster OCR
                roi_height, roi_width = roi.shape[:2]
                scale = OCR_WIDTH / roi_width
                new_height = int(roi_height * scale)
                roi_resized = cv2.resize(roi, (OCR_WIDTH, new_height))
                roi_resized = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                
                # Run EasyOCR on resized region
                allow_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
                results = reader.readtext(roi_resized, batch_size=1, workers=1, allowlist=allow_list)
                
                # Process results
                temp_results = []
                for bbox, text, conf in results:
                    # Adjust bbox coordinates back to frame
                    scaled_pts = []
                    for pt in bbox:
                        orig_x = int(pt[0] / scale)
                        orig_y = int(pt[1] / scale)
                        scaled_pts.append((orig_x + x1, orig_y + y1))
                    temp_results.append((scaled_pts, text, conf))
                
                # Replace current results with new ones
                with ocr_lock:
                    current_ocr_results.clear()
                    current_ocr_results.extend(temp_results)
                    
        except:
            continue  # Handle timeout or other errors

# Start OCR thread
threading.Thread(target=ocr_loop, daemon=True).start()

frame_count = 0

# Main display loop
while True:
    frame = cam.capture_array()
    frame_count += 1
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    
    # --- STEP 2: Gate the YOLO detection and OCR queuing ---
    if frame_count % YOLO_INTERVAL == 0:
        # Resize frame before running YOLO
        original_h, original_w = frame.shape[:2]
        yolo_h = int(original_h * (YOLO_WIDTH / original_w))
        frame_resized_for_yolo = cv2.resize(frame, (YOLO_WIDTH, yolo_h))

        # Use YOLO to detect regions on the smaller frame
        yolo_results = yolo_model(frame_resized_for_yolo, conf=0.25)
        
        # Clear the list of last known boxes
        last_yolo_boxes.clear()

        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
                    # Scale bounding box coordinates back to original frame size
                    x1_small, y1_small, x2_small, y2_small = box.xyxy[0].cpu().numpy()
                    x_scale = original_w / YOLO_WIDTH
                    y_scale = original_h / yolo_h
                    x1 = int(x1_small * x_scale)
                    y1 = int(y1_small * y_scale)
                    x2 = int(x2_small * x_scale)
                    y2 = int(y2_small * y_scale)

                    # Add the scaled box to our list for drawing
                    last_yolo_boxes.append((x1, y1, x2, y2))
                    
                    # Add region to OCR queue (gated by its own interval)
                    if frame_count % OCR_INTERVAL == 0 and not ocr_queue.full():
                        roi = frame[y1:y2, x1:x2]
                        try:
                            ocr_queue.put_nowait((roi, x1, y1))
                        except:
                            pass

            # If detection ran but found no boxes, clear stale data.
            if not last_yolo_boxes:
                # Clear the results currently being displayed
                with ocr_lock:
                    current_ocr_results.clear()
                
                # Empty the queue of any pending (now irrelevant) work
                while not ocr_queue.empty():
                    try:
                        ocr_queue.get_nowait()
                    except Empty:
                        continue
    
    # --- STEP 3: Draw the last known boxes on EVERY frame ---
    for (x1, y1, x2, y2) in last_yolo_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw current OCR results
    with ocr_lock:
        results_to_draw = list(current_ocr_results)
    
    # Create black strip for text display
    strip_height = 60
    text_panel = np.zeros((strip_height, frame.shape[1], 3), dtype=np.uint8)

    # Compose line of text
    full_line = "BRAND NAME: " + " ".join([text for _, text, _ in results_to_draw])
    cv2.putText(text_panel, full_line, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Stack frame and text panel
    combined = np.vstack((frame, text_panel))
    
    cv2.imshow("OCR Feed", combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.stop()