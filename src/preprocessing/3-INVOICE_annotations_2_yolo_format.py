#!/usr/bin/env python3

from PIL import Image
import json
from pathlib import Path

IMG_DIR = Path("data/processed/invoiceIMG")
ANN_DIR = Path("data/annotations/invoiceANNOTATIONS")
YOLO_LABELS = Path("data/yolo_labels")

YOLO_LABELS.mkdir(exist_ok=True)

CLASS_ID = 0  # text

for img_path in IMG_DIR.glob("*.png"):
    pdf_name = img_path.stem.split("_page")[0]
    page_idx = int(img_path.stem.split("page")[1]) - 1

    ann_file = ANN_DIR / f"{pdf_name}.json"
    if not ann_file.exists():
        print(f"Annotation file missing: {ann_file}")
        continue

    # Load image size
    img = Image.open(img_path)
    W, H = img.size

    # Load page annotations
    with open(ann_file) as f:
        data = json.load(f)

    if page_idx >= len(data):
        print(f"Page index {page_idx} out of range in {ann_file}")
        continue

    labels = []
    for item in data[page_idx]:
        x0, y0, x1, y1 = item["bbox"]

        # Normalize coordinates for YOLO
        xc = ((x0 + x1) / 2) / W
        yc = ((y0 + y1) / 2) / H
        w  = (x1 - x0) / W
        h  = (y1 - y0) / H

        # Optional: skip very tiny boxes
        if w <= 0 or h <= 0:
            continue

        labels.append(f"{CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # Write YOLO .txt label file
    out_file = YOLO_LABELS / f"{img_path.stem}.txt"
    out_file.write_text("\n".join(labels))

    print(f"Saved YOLO labels: {out_file} ({len(labels)} boxes)")
