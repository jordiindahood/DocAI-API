#!/usr/bin/env python3
"""
Split YOLO images and labels into train / validation sets.

Input:
- data/processed/invoiceIMG/*.png
- data/yolo_labels/*.txt

Output:
- data/yolo_dataset/
    ├── images/train
    ├── images/val
    ├── labels/train
    └── labels/val
"""

from pathlib import Path
import random
import shutil

# --------------------------------------------------
# PROJECT ROOT = two levels up from this file
# src/preprocessing/4-split_yolo_dataset.py
# --------------------------------------------------

DATA_DIR = Path("data") #   REMEMBER TO EXECUTE THIS SCRIPT IN THE ROOT DIR "python src/preprocessing/4*"

IMG_SRC = DATA_DIR / "processed" / "invoiceIMG"
LBL_SRC = DATA_DIR / "yolo_labels"

DATASET_DIR = DATA_DIR / "yolo_dataset"
TRAIN_IMG = DATASET_DIR / "images" / "train"
VAL_IMG = DATASET_DIR / "images" / "val"
TRAIN_LBL = DATASET_DIR / "labels" / "train"
VAL_LBL = DATASET_DIR / "labels" / "val"

# --------------------------------------------------
# Create directories
# --------------------------------------------------
for p in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load images
# --------------------------------------------------
images = sorted(IMG_SRC.glob("*.png"))
random.shuffle(images)

split_idx = int(0.8 * len(images))

train_count = 0
val_count = 0

for i, img_path in enumerate(images):
    label_path = LBL_SRC / f"{img_path.stem}.txt"

    # Skip images without labels
    if not label_path.exists():
        continue

    if i < split_idx:
        shutil.copy(img_path, TRAIN_IMG / img_path.name)
        shutil.copy(label_path, TRAIN_LBL / label_path.name)
        train_count += 1
    else:
        shutil.copy(img_path, VAL_IMG / img_path.name)
        shutil.copy(label_path, VAL_LBL / label_path.name)
        val_count += 1

print("YOLO dataset split completed")
print(f"Train samples: {train_count}")
print(f"Validation samples: {val_count}")
print(f"Dataset location: {DATASET_DIR}")
