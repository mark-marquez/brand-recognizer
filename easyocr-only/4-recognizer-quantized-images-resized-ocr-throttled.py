import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np

reader = easyocr.Reader(['en'], gpu=False, quantize=True)
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

# Target size for faster OCR (smaller than full frame)
target_size = (320, 240)

# Run OCR every N frames
ocr_interval = 5
frame_count = 0
scaled_results = []

while True:
    frame = cam.capture_array()
    frame_count += 1

    if frame_count % ocr_interval == 0:
        resized = cv2.resize(frame, target_size)
        results = reader.readtext(resized, batch_size=1)
        
        # Calculate scale factors to map boxes back to original frame
        scale_x = frame.shape[1] / target_size[0]
        scale_y = frame.shape[0] / target_size[1]
        
        scaled_results = []
        for bbox, text, conf in results:
            scaled_pts = [(int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in bbox]
            scaled_results.append((scaled_pts, text))

    # Draw most recent results on current frame
    for pts, text in scaled_results:
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("OCR Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()