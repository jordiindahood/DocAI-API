#!/usr/bin/env python3
"""
YOLOv8 training script optimized for GTX 1050 (4GB VRAM)
"""

from ultralytics import YOLO
import torch
from pathlib import Path

# ---------------------------
# GPU CHECK
# ---------------------------
assert torch.cuda.is_available(), "CUDA not available. GPU not detected."

device = "cuda:0"
print("Using GPU:", torch.cuda.get_device_name(0))

# ---------------------------
# Load model (smallest version)
# ---------------------------
model = YOLO("models/yolov8n.pt")

# ---------------------------
# Dataset YAML
# ---------------------------
data_yaml = Path("data/yolo_dataset/data.yaml")
if not data_yaml.exists():
    raise FileNotFoundError(f"{data_yaml} not found")

# ---------------------------
# Training parameters
# ---------------------------
model.train(
    data=str(data_yaml),
    epochs=50,
    imgsz=640,
    batch=4,
    workers=2,
    device=0,
    amp=True,                  # mixed precision (important)
    project="outputs/yolo_results",
    name="invoice_yolo",
    exist_ok=True,
    verbose=True
)

print("Training finished successfully")
