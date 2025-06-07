import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO

# Add YOLO model
yolo_model = YOLO('best.pt')  # Update path if needed

reader = easyocr.Reader(['en'], gpu=False, quantize=True)
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

# Resize parameters for OCR
OCR_WIDTH = 320  # Target width for OCR processing

# Frame throttling for OCR
frame_count = 0
OCR_INTERVAL = 5  # Run OCR every 5 frames
current_ocr_results = []  # Store current frame's OCR results

while True:
    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)  # Convert from 4-channel BGRA to 3-channel RGB
    
    # Use YOLO to detect regions
    yolo_results = yolo_model(frame)
    
    # Clear results on OCR frames
    if frame_count % OCR_INTERVAL == 0:
        current_ocr_results = []
    
    for r in yolo_results:
        if r.boxes is not None:
            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Draw YOLO box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Run OCR only every OCR_INTERVAL frames
                if frame_count % OCR_INTERVAL == 0:
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Resize ROI for faster OCR
                        roi_height, roi_width = roi.shape[:2]
                        scale = OCR_WIDTH / roi_width
                        new_height = int(roi_height * scale)
                        roi_resized = cv2.resize(roi, (OCR_WIDTH, new_height))
                        
                        # Run EasyOCR on resized region
                        results = reader.readtext(roi_resized)
                        
                        # Process results for this box
                        for bbox, text, conf in results:
                            # Adjust bbox coordinates back to original size then to frame
                            pts = []
                            for pt in bbox:
                                orig_x = int(pt[0] / scale)
                                orig_y = int(pt[1] / scale)
                                pts.append((orig_x + x1, orig_y + y1))
                            current_ocr_results.append((pts, text, conf))
    
    # Draw current OCR results
    for pts, text, conf in current_ocr_results:
        # Draw bounding box
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("OCR Feed", frame)
    
    frame_count += 1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()