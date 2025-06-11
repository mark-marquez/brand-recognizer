# --- OPTIMIZATIONS ---
#
# 1.  Cascading Pipeline: A fast YOLO model first finds the general logo area.
# 2.  Region-Specific OCR: The slower OCR process only runs on the small image region detected by YOLO.
# 3.  Multi-Threading: A secondary thread handles all OCR work so it doesn't block or slow down the main camera feed.
# 4.  YOLO Input Resizing: Full camera frames are downscaled before YOLO processing to speed up inference.
# 5.  YOLO Model Quantization: Uses a quantized .tflite model for faster, more efficient execution on edge devices.
# 6.  YOLO Throttling: YOLO detection is skipped on most frames and only runs intermittently (e.g., every 3rd frame).
# 7.  OCR Input Resizing: The cropped logo region is resized again before being sent to EasyOCR.
# 8.  Grayscale Conversion: Images are converted to grayscale before OCR, a common and efficient preprocessing step.
# 9.  Contrast Enhancement (CLAHE): Applies CLAHE to the grayscale image to improve text visibility in varied lighting.
# 10. OCR Model Quantization: The EasyOCR recognition model itself is quantized for better performance.
# 11. OCR Character Filtering: Uses an `allowlist` to restrict OCR to only search for alphabetic characters, speeding up recognition.
# 12. OCR Throttling: Queuing new regions for the OCR thread is independently throttled (e.g., every 5th frame).
# 13. Stable Visuals: The last detected bounding box is drawn on every single frame, preventing a "flickering" effect.
# 14. Simplified Display: Only the main blue bounding box from YOLO is rendered, not the individual word boxes from OCR.
# 15. Efficient Data Clearing: When a logo is no longer detected, the pending OCR queue is cleared instantly.

import cv2
import numpy as np
import threading
from queue import Queue, Empty, Full
import time
import psutil

import easyocr
from picamera2 import Picamera2, controls
from ultralytics import YOLO

# --- Configuration ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_WIDTH = 320
OCR_WIDTH = 320

YOLO_INTERVAL = 3  # Run YOLO detection every N frames.
OCR_INTERVAL = 5   # Queue regions for OCR every N frames.
SHARPNESS_THRESHOLD = 100.0 # Tune this value; higher is stricter.

# --- Initialization ---
yolo_model = YOLO('best-june-08_int8.tflite', task='detect')
ocr_reader = easyocr.Reader(['en'], gpu=False, quantize=True)

cam = Picamera2()
cam.configure(cam.create_preview_configuration({"size": (FRAME_WIDTH, FRAME_HEIGHT)}))
cam.set_controls({"AfMode": 0, "LensPosition": 2})
cam.start()

# --- Shared Resources for Threading ---
ocr_queue = Queue(maxsize=10)
last_yolo_boxes = []
current_ocr_results = []
ocr_lock = threading.Lock()

def calculate_sharpness(image: np.ndarray) -> float:
    """Calculates a sharpness score for an image using the variance of the Laplacian."""
    if image.size == 0:
        return 0.0
    # Convert to grayscale and compute the variance of the Laplacian
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def ocr_processing_thread():
    """
    Background thread to run OCR on image regions from a queue.
    This function preprocesses images and safely updates shared results.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    allow_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

    while True:
        try:
            roi, x_offset, y_offset = ocr_queue.get(timeout=1)
        except Empty:
            continue

        if roi.size == 0:
            continue

        # Preprocess ROI for better OCR accuracy.
        roi_height, roi_width = roi.shape[:2]
        scale = OCR_WIDTH / roi_width
        new_height = int(roi_height * scale)
        roi_resized = cv2.resize(roi, (OCR_WIDTH, new_height))
        gray_roi = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
        enhanced_roi = clahe.apply(gray_roi)

        results = ocr_reader.readtext(
            enhanced_roi, batch_size=1, workers=1, allowlist=allow_list
        )

        # Scale recognized text bounding boxes back to original frame coordinates.
        processed_results = []
        for bbox, text, conf in results:
            scaled_pts = [
                (int(pt[0] / scale) + x_offset, int(pt[1] / scale) + y_offset)
                for pt in bbox
            ]
            processed_results.append((scaled_pts, text, conf))

        with ocr_lock:
            current_ocr_results.clear()
            current_ocr_results.extend(processed_results)

def main():
    """
    Main loop to capture frames, run object detection, and display results.
    """
    global last_yolo_boxes, current_ocr_results
    frame_count = 0

    # Pre-calculate scaling factors for resizing bounding boxes.
    yolo_h = int(FRAME_HEIGHT * (YOLO_WIDTH / FRAME_WIDTH))
    x_scale = FRAME_WIDTH / YOLO_WIDTH
    y_scale = FRAME_HEIGHT / yolo_h

    # Start the background OCR processing thread.
    threading.Thread(target=ocr_processing_thread, daemon=True).start()

    prev_time = 0
    fps_readings = []
    cpu_readings = []
    process = psutil.Process()
    run_start_time = time.time()

    while True:
        if time.time() - run_start_time > 30:
            print("30-second test duration complete. Exiting...")
            break

        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        frame_count += 1

        # Periodically run the heavier YOLO model for performance.
        if frame_count % YOLO_INTERVAL == 0:
            frame_resized_for_yolo = cv2.resize(frame, (YOLO_WIDTH, yolo_h))
            yolo_results = yolo_model(frame_resized_for_yolo, conf=0.25, verbose=False)

            new_boxes = []
            should_queue_for_ocr = (frame_count % OCR_INTERVAL == 0)

            for r in yolo_results:
                for box in r.boxes:
                    x1_s, y1_s, x2_s, y2_s = box.xyxy[0].cpu().numpy()
                    x1 = int(x1_s * x_scale)
                    y1 = int(y1_s * y_scale)
                    x2 = int(x2_s * x_scale)
                    y2 = int(y2_s * y_scale)
                    new_boxes.append((x1, y1, x2, y2))

                    if should_queue_for_ocr and not ocr_queue.full():
                        # Extract the region of interest (ROI)
                        roi = frame[y1:y2, x1:x2]
                        
                        # Calculate sharpness of the ROI
                        sharpness = calculate_sharpness(roi)
                        
                        # --- For Tuning: Uncomment the line below to see sharpness scores ---
                        # print(f"Detected sharpness: {sharpness:.2f}")

                        # Only queue the ROI if it's sharp enough
                        if sharpness > SHARPNESS_THRESHOLD:
                            try:
                                ocr_queue.put_nowait((roi, x1, y1))
                            except Full:
                                pass
            
            last_yolo_boxes = new_boxes

            # If YOLO finds no objects, clear previous (now stale) results.
            if not last_yolo_boxes:
                with ocr_queue.mutex:
                    ocr_queue.queue.clear()

        # Draw the last known bounding boxes on every frame for visual stability.
        for (x1, y1, x2, y2) in last_yolo_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Safely copy the latest OCR results for drawing.
        with ocr_lock:
            results_to_draw = list(current_ocr_results)

        # Create a display panel for the recognized text.
        text_panel = np.zeros((60, frame.shape[1], 3), dtype=np.uint8)
        recognized_text = " ".join([text for _, text, _ in results_to_draw])
        cv2.putText(text_panel, f"BRAND NAME: {recognized_text}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        current_time = time.time()
        if prev_time > 0:
            fps = 1 / (current_time - prev_time)
            fps_readings.append(fps)
            cpu_readings.append(process.cpu_percent())
        prev_time = current_time

        combined_view = np.vstack((frame, text_panel))
        cv2.imshow("Brand Recognizer", combined_view)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    if fps_readings:
        average_fps = sum(fps_readings) / len(fps_readings)
        average_cpu = sum(cpu_readings[1:]) / len(cpu_readings[1:]) if len(cpu_readings) > 1 else 0

        print(f"Optimized Average FPS: {average_fps:.2f}")
        print(f"Optimized Average CPU Utilization: {average_cpu:.2f}%")
    
        with open("optimized_performance.txt", "w") as f:
            f.write(f"Average FPS: {average_fps:.2f}\n")
            f.write(f"Average CPU Utilization: {average_cpu:.2f}%\n")

    cv2.destroyAllWindows()
    cam.stop()

if __name__ == "__main__":
    main()