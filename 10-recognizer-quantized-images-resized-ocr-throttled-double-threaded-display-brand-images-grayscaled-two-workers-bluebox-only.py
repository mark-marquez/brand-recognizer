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
cam.set_controls({"AfMode": 0, "LensPosition": 3.2})  # Manual focus ~2 feet
cam.start()

# Resize parameters for OCR
OCR_WIDTH = 320  # Target width for OCR processing

# Shared OCR results
ocr_queue = Queue(maxsize=10)
current_ocr_results = []  # List of (pts, text, conf) for current frame
ocr_lock = threading.Lock()

latest_frame = None
latest_yolo_frame = None
yolo_lock = threading.Lock()

latest_boxes = []

# Helper function to check if an image is blurry
def is_blurry(image, threshold=100.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var < threshold

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
                results = reader.readtext(roi_resized, batch_size=1, workers=1)
                
                # Process results
                temp_results = []
                grouped_text = []
                for bbox, text, conf in results:
                    if conf > 0.5:
                        grouped_text.append((bbox, text, conf))

                if grouped_text:
                    all_text = " ".join(text for _, text, _ in grouped_text)
                    highest_conf = max(conf for _, _, conf in grouped_text)
                    scaled_pts = []
                    for pt in grouped_text[0][0]:  # use first bbox for positioning
                        orig_x = int(pt[0] / scale)
                        orig_y = int(pt[1] / scale)
                        scaled_pts.append((orig_x + x1, orig_y + y1))
                    temp_results.append((scaled_pts, all_text, highest_conf))
                
                # Replace current results with new ones
                with ocr_lock:
                    current_ocr_results.clear()
                    current_ocr_results.extend(temp_results)
                    
        except:
            continue  # Handle timeout or other errors

def yolo_loop():
    global latest_frame
    global latest_yolo_frame
    global latest_boxes
    frame_count = 0
    while True:
        if latest_frame is not None:
            frame_copy = latest_frame
            yolo_results = yolo_model(frame_copy) if (frame_count % 5 == 0) else [type('obj', (object,), {'boxes': None})()]
            if frame_count % 5 == 0:
                for r in yolo_results:
                    if r.boxes is not None:
                        for box in r.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 0, 0), 2)
            frame_count += 1
            
            new_boxes = []
            for r in yolo_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        new_boxes.append((x1, y1, x2, y2))
                        print(f"YOLO Detected box: ({x1}, {y1}), ({x2}, {y2})")
                        roi = frame_copy[y1:y2, x1:x2]
                        if not ocr_queue.full() and not is_blurry(roi):
                            try:
                                ocr_queue.put_nowait((roi, x1, y1))
                            except:
                                pass
            with yolo_lock:
                latest_boxes.clear()
                latest_boxes.extend(new_boxes)
                print("Updated latest_boxes with", len(new_boxes), "boxes")

# Start OCR thread
threading.Thread(target=ocr_loop, daemon=True).start()


latest_yolo_frame = np.zeros((480, 640, 3), dtype=np.uint8)
threading.Thread(target=yolo_loop, daemon=True).start()

# Main display loop
boxes_to_draw = []
last_box_time = 0

while True:
    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    latest_frame = frame

    with yolo_lock:
        if latest_boxes:
            last_box_time = time.time()
            boxes_to_draw = latest_boxes.copy()
        elif time.time() - last_box_time < 3:
            boxes_to_draw = boxes_to_draw  # reuse previous
        else:
            boxes_to_draw = []

    print("Drawing", len(boxes_to_draw), "boxes")

    display_frame = latest_frame.copy() if latest_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    # Box drawing now handled in yolo_loop; do not draw here

    # Draw current OCR results
    with ocr_lock:
        results_to_draw = list(current_ocr_results)

    # Create black strip for text display
    strip_height = 60
    text_panel = np.zeros((strip_height, display_frame.shape[1], 3), dtype=np.uint8)

    # Compose line of text
    if results_to_draw:
        best_result = max(results_to_draw, key=lambda x: x[2])
        full_line = "BRAND NAME: " + best_result[1]
    else:
        full_line = "BRAND NAME: "
    cv2.putText(text_panel, full_line, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Stack frame and text panel
    combined = np.vstack((display_frame, text_panel))
    cv2.imshow("OCR Feed", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.stop()