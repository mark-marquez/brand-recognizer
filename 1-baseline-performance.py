import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np
from ultralytics import YOLO
import time
import psutil

# Add YOLO model
yolo_model = YOLO('best-june-08.pt') 

reader = easyocr.Reader(['en'], gpu=False, quantize=False)
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

prev_time = 0
fps_readings = []
cpu_readings = []
process = psutil.Process()
run_start_time = time.time()

while True:
    # Programmatically end the script after 30 seconds
    if time.time() - run_start_time > 30:
        print("30-second test duration complete. Exiting...")
        break

    frame = cam.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    
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

    # --- Performance Calculation ---
    current_time = time.time()
    if prev_time > 0:
        fps = 1 / (current_time - prev_time)
        fps_readings.append(fps)
        cpu_readings.append(process.cpu_percent())
    prev_time = current_time
    # --- End of Performance Calculation ---
    
    cv2.imshow("OCR Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Calculate and Save Performance Metrics ---
if fps_readings:
    average_fps = sum(fps_readings) / len(fps_readings)
    # The first CPU reading can be 0.0, so we skip it for a more accurate average
    average_cpu = sum(cpu_readings[1:]) / len(cpu_readings[1:]) if len(cpu_readings) > 1 else 0

    print(f"Baseline Average FPS: {average_fps:.2f}")
    print(f"Baseline Average CPU Utilization: {average_cpu:.2f}%")
    
    with open("baseline_performance.txt", "w") as f:
        f.write(f"Average FPS: {average_fps:.2f}\n")
        f.write(f"Average CPU Utilization: {average_cpu:.2f}%\n")
# --- End of metric saving ---

cv2.destroyAllWindows()
cam.stop()