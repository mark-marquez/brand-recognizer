import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
from queue import Queue

# Initialize models
yolo_model = YOLO('best-june-08.pt')
reader = easyocr.Reader(['en'], gpu=False, quantize=True)

# Initialize camera
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

# Resize parameters for OCR
OCR_WIDTH = 320  # Target width for OCR processing

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
                
                # Run EasyOCR on resized region
                results = reader.readtext(roi_resized)
                
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
OCR_INTERVAL = 5  # Only send frames to OCR every 5 frames

# Main display loop
while True:
    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    
    frame_count += 1

    # Use YOLO to detect regions
    yolo_results = yolo_model(frame)
    
    for r in yolo_results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Add region to OCR queue if not full
                if frame_count % OCR_INTERVAL == 0 and not ocr_queue.full():
                    roi = frame[y1:y2, x1:x2]
                    try:
                        ocr_queue.put_nowait((roi, x1, y1))
                    except:
                        pass  # Queue full, skip this frame
                
                # Draw YOLO box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    # Draw current OCR results
    with ocr_lock:
        results_to_draw = list(current_ocr_results)
        
    for pts, text, conf in results_to_draw:
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow("OCR Feed", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.stop()