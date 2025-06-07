import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO

# Add YOLO model
yolo_model = YOLO('best.pt') 

reader = easyocr.Reader(['en'], gpu=False, quantize=False)
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

while True:
    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)  # Convert from 4-channel BGRA to 3-channel RGB
    
    # Use YOLO to detect regions
    yolo_results = yolo_model(frame)
    
    for r in yolo_results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                roi = frame[y1:y2, x1:x2]
                
                if roi.size > 0:
                    # Run EasyOCR on detected region
                    results = reader.readtext(roi)
                    
                    for bbox, text, conf in results:
                        # Adjust bbox coordinates to frame
                        pts = []
                        for pt in bbox:
                            pts.append((int(pt[0] + x1), int(pt[1] + y1)))
                        
                        # Draw bounding box
                        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
                        cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    cv2.imshow("OCR Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.stop()