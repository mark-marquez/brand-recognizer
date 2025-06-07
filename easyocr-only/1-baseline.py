import easyocr
from picamera2 import Picamera2
import cv2
import numpy as np

reader = easyocr.Reader(['en'], gpu=False, quantize=False)
cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (640, 480)}))
cam.start()

while True:
    frame = cam.capture_array()
    results = reader.readtext(frame)

    for bbox, text, conf in results:
        # Draw bounding box
        pts = [tuple(map(int, pt)) for pt in bbox]
        cv2.polylines(frame, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(frame, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("OCR Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()