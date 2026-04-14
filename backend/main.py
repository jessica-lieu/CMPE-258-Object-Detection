import cv2
import os
import shutil
import uuid
import time
import subprocess
import zipfile
import yaml
import traceback
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ultralytics import YOLO, RTDETR
import ultralytics.models.yolo.detect.val

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Load models once
MODELS = {
    "yolo_onnx": YOLO("../models/yolo26n.onnx", task="detect"),
    "yolo_trt":  YOLO("../models/yolo26n.engine", task="detect"),
    "detr_onnx": RTDETR("../models/rtdetr-l.onnx"),
    "detr_trt":  RTDETR("../models/rtdetr-l.engine")
}

@app.post("/process")
async def process(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    engine: str = Query("yolo_trt"),
    annotations: Optional[UploadFile] = File(None)
):
    unique_id = str(uuid.uuid4())
    input_path = f"in_{unique_id}.mp4"
    output_path = f"out_raw_{unique_id}.mp4"
    final_path = f"final_{unique_id}.mp4"

    # 1. Save upload
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. Setup OpenCV
    cap = cv2.VideoCapture(input_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    model = MODELS[engine]
    start_time = time.time()

    # 3. Process every frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        results = model.predict(frame, verbose=False, device=0)
        # .plot() draws the boxes and labels for you!
        annotated_frame = results[0].plot() 
        out.write(annotated_frame)

    cap.release()
    out.release()
    total_latency = (time.time() - start_time) * 1000 # in ms

    map_score = None
    if annotations:
        dataset_id = str(uuid.uuid4())
        base_dir = f"temp_dataset_{dataset_id}"
        images_dir = os.path.join(base_dir, "images", "val")
        labels_dir = os.path.join(base_dir, "labels", "val")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # Unpack zip, flatten structure
        zip_path = f"{base_dir}_annos.zip"
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(annotations.file, buffer)
            
        custom_images = False
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for info in zip_ref.infolist():
                if info.filename.endswith('.txt') and not info.is_dir():
                    info.filename = os.path.basename(info.filename)
                    zip_ref.extract(info, labels_dir)
                elif info.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) and not info.is_dir():
                    info.filename = os.path.basename(info.filename)
                    zip_ref.extract(info, images_dir)
                    custom_images = True
        os.remove(zip_path)

        if not custom_images:
            label_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])
            
            cap2 = cv2.VideoCapture(input_path)
            frame_idx = 0
            while cap2.isOpened():
                ret, frame = cap2.read()
                if not ret:
                    break
                
                if frame_idx < len(label_files):
                    frame_name = label_files[frame_idx].replace('.txt', '.jpg')
                else:
                    frame_name = f"frame_{frame_idx:04d}.jpg"
                    
                cv2.imwrite(os.path.join(images_dir, frame_name), frame)
                frame_idx += 1
            cap2.release()

        yaml_path = os.path.join(base_dir, "dataset.yaml")
        yaml_content = {
            "path": os.path.abspath(base_dir),
            "train": "images/val",
            "val": "images/val",
            "names": model.names
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f)
            
        # Run evaluation natively on the requested accelerator
        try:
            _orig_iou = ultralytics.utils.metrics.box_iou
            
            # Execute IoU purely on CPU, averting all CUDA NVFuser JIT compilation bugs
            def cpu_iou(box1, box2, eps=1e-7):
                b1_cpu = box1.cpu().float()
                b2_cpu = box2.cpu().float()
                iou = _orig_iou(b1_cpu, b2_cpu, eps=eps)
                return iou.to(box1.device)
                
            ultralytics.models.yolo.detect.val.box_iou = cpu_iou
            
            results_val = model.val(data=yaml_path, device=0, split='val', plots=False, verbose=False)
            map_score = results_val.box.map50
        except Exception as e:
            import traceback
            print(f"Eval Error: {e}")
            with open("eval_error.txt", "w") as f:
                f.write(traceback.format_exc())
            
        # Cleanup temp dataset
        shutil.rmtree(base_dir)

    # 4. CRITICAL: Convert to H.264 for Browser playback
    # Browsers hate raw OpenCV mp4 files. FFmpeg fixes this.
    subprocess.run([
        'ffmpeg', 
        '-i', output_path, 
        '-vcodec', 'libx264', 
        '-pix_fmt', 'yuv420p', # CRITICAL: This makes it playable in browsers
        '-profile:v', 'baseline', # Extra compatibility
        '-level', '3.0', 
        '-f', 'mp4', 
        final_path, 
        '-y'
    ], check=True)

    # 5. Cleanup temp files
    os.remove(input_path)
    os.remove(output_path)

    # 6. Return the video file
    # We add the latency to the headers so the frontend can read it
    headers = {"X-Inference-Latency": str(round(total_latency, 2))}
    if map_score is not None:
        headers["X-Inference-mAP"] = f"{map_score * 100:.2f}"

    return FileResponse(
        final_path, 
        media_type="video/mp4",
        headers=headers
    )