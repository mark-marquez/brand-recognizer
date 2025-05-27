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

while True:
    frame = cam.capture_array()
    
    # Resize for OCR processing
    resized = cv2.resize(frame, target_size)

    # Run EasyOCR on resized image
    results = reader.readtext(resized)

    # Calculate scale factors to map boxes back to original frame
    scale_x = frame.shape[1] / target_size[0]
    scale_y = frame.shape[0] / target_size[1]

    for bbox, text, conf in results:
        # Scale box coordinates back to original frame size
        scaled_pts = [
            (int(pt[0] * scale_x), int(pt[1] * scale_y)) for pt in bbox
        ]
        cv2.polylines(frame, [np.array(scaled_pts)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, text, scaled_pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("OCR Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()