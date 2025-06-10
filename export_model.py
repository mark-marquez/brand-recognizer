# export_model.py

from ultralytics import YOLO

# 1. Load your original trained model
print("Loading the original .pt model...")
model = YOLO('best-june-08.pt')

# 2. Export it to the TFLite format with INT8 quantization
print("Exporting to a quantized TFLite model...")
model.export(format='tflite', int8=True)

print("\nExport complete!")
print("A new quantized model has been created in the 'best-june-08_saved_model' directory.")