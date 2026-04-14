import os
import sys
import torch
import shutil
from ultralytics import YOLO, RTDETR

print("=== DIAGNOSTICS ===")
print("Python Executable Path:", sys.executable)
print("Torch Version:", torch.__version__)
print("GPU Available:", torch.cuda.is_available())
print("===================")

MODEL_DIR = "models"

def get_model(model_name, model_class):
    target_path = os.path.join(MODEL_DIR, model_name)
    
    if os.path.exists(target_path):
        print(f"Found {model_name} in {MODEL_DIR}/.")
        return model_class(target_path)
    
    print(f"Downloading {model_name}")
    model = model_class(model_name) 
    
    if os.path.exists(model_name):
        shutil.move(model_name, target_path)
        print(f"Moved {model_name} to {MODEL_DIR}/")
        
    return model

print("Acceleration")

yolo_pt = get_model("yolo26n.pt", YOLO)
detr_pt = get_model("rtdetr-l.pt", RTDETR)

print("Exporting to ONNX")
yolo_onnx_orig = yolo_pt.export(format="onnx")
detr_onnx_orig = detr_pt.export(format="onnx")

yolo_onnx_path = shutil.move(yolo_onnx_orig, os.path.join(MODEL_DIR, os.path.basename(yolo_onnx_orig)))
detr_onnx_path = shutil.move(detr_onnx_orig, os.path.join(MODEL_DIR, os.path.basename(detr_onnx_orig)))
print(f"ONNX models moved to: {MODEL_DIR}")

print("Compiling TensorRT Engines")
yolo_engine_orig = yolo_pt.export(format="engine", device=0, half=True)
detr_engine_orig = detr_pt.export(format="engine", device=0, half=True)

yolo_engine_path = shutil.move(yolo_engine_orig, os.path.join(MODEL_DIR, os.path.basename(yolo_engine_orig)))
detr_engine_path = shutil.move(detr_engine_orig, os.path.join(MODEL_DIR, os.path.basename(detr_engine_orig)))
print(f"TensorRT engines moved to: {MODEL_DIR}")