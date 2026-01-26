# YOLO Training Pipeline

This folder contains all files related to training the YOLO layout detection model for document analysis.

## 📁 Folder Structure

```
train/
├── preprocessing/      # Data preparation pipeline
├── augmentation/       # Data augmentation
├── training/           # YOLO model training
├── scripts/            # Helper scripts
└── notebooks/          # Jupyter notebooks for experimentation
```

## 🔄 Training Pipeline Workflow

The complete training pipeline follows these steps:

### 1. PDF to Image Conversion
**File:** `preprocessing/1-pdf2img.py`

Converts PDF documents to PNG images at 300 DPI.

```bash
python preprocessing/1-pdf2img.py
```

**Input:** `data/invoice/*.pdf`  
**Output:** `data/processed/invoiceIMG/*.png`

---

### 2. Extract Annotations
**File:** `preprocessing/2-extract_txt_and_bounding_boxes.py`

Extracts text and bounding box annotations from PDFs using PyMuPDF.

```bash
python preprocessing/2-extract_txt_and_bounding_boxes.py
```

**Input:** PDFs from `data/invoice/`  
**Output:** `data/annotations/invoiceANNOTATIONS/*.json`

---

### 3. Convert to YOLO Format
**File:** `preprocessing/3-INVOICE_annotations_2_yolo_format.py`

Converts JSON annotations to YOLO label format.

```bash
python preprocessing/3-INVOICE_annotations_2_yolo_format.py
```

**Input:** 
- Images: `data/processed/invoiceIMG/*.png`
- Annotations: `data/annotations/invoiceANNOTATIONS/*.json`

**Output:** `data/yolo_labels/*.txt` (YOLO format)

---

### 4. Split Dataset
**File:** `preprocessing/4-split_yolo_dataset.py`

Splits images and labels into training (80%) and validation (20%) sets.

```bash
python preprocessing/4-split_yolo_dataset.py
```

**Input:**
- Images: `data/processed/invoiceIMG/*.png`
- Labels: `data/yolo_labels/*.txt`

**Output:**
```
data/yolo_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

### 5. Data Augmentation (Optional)
**File:** `augmentation/synthetic_phone_augmentation.py`

Applies synthetic augmentations to simulate real-world conditions:
- Perspective distortion
- Illumination gradients
- Gaussian blur
- Brightness/contrast jitter
- JPEG compression
- Gaussian noise

```bash
python augmentation/synthetic_phone_augmentation.py \
    data/yolo_dataset \
    data/yolo_dataset_augmented \
    0.6 \
    42
```

**Parameters:**
- `0.6` = 60% of images will be augmented
- `42` = Random seed for reproducibility

**Supporting Files:**
- `augmentation/augmentations.py` - Core augmentation functions
- `augmentation/augmentation_utils.py` - YOLO label parsing utilities

---

### 6. Train YOLO Model
**File:** `training/train_yolo.py`

Trains YOLOv8 model optimized for GTX 1050 (4GB VRAM).

```bash
python training/train_yolo.py
```

**Requirements:**
- `data/yolo_dataset/data.yaml` must exist
- Pre-trained weights: `models/yolov8n.pt`

**Training Parameters:**
- Epochs: 50
- Image size: 640
- Batch size: 4
- Mixed precision (AMP): Enabled
- Workers: 2

**Output:** `outputs/yolo_results/invoice_yolo/`

---

## 📓 Jupyter Notebooks

### PDF Converter Notebook
**File:** `notebooks/1-pdf2imgConverter.ipynb`

Interactive notebook for experimenting with PDF to image conversion.

### Data Augmentation Notebook
**File:** `notebooks/data-augmentation.ipynb`

Visual experimentation with data augmentation techniques.

### YOLO Test Notebook
**File:** `notebooks/yolov8-layout-test.ipynb`

Testing and visualizing YOLO model predictions.

---

## 🛠️ Helper Scripts

### Convert Annotations to OCR JSON
**File:** `scripts/convert_annotations_to_ocr_json.py`

Utility script to convert annotations to OCR-compatible JSON format.

---

## 📋 Quick Start

To run the complete training pipeline from scratch:

```bash
# 1. Convert PDFs to images
python train/preprocessing/1-pdf2img.py

# 2. Extract annotations
python train/preprocessing/2-extract_txt_and_bounding_boxes.py

# 3. Convert to YOLO format
python train/preprocessing/3-INVOICE_annotations_2_yolo_format.py

# 4. Split dataset
python train/preprocessing/4-split_yolo_dataset.py

# 5. (Optional) Apply augmentations
python train/augmentation/synthetic_phone_augmentation.py \
    data/yolo_dataset data/yolo_dataset_augmented 0.6 42

# 6. Train the model
python train/training/train_yolo.py
```

---

## 📊 Expected Data Flow

```
PDFs (data/invoice/)
    ↓
Images (data/processed/invoiceIMG/)
    ↓
Annotations (data/annotations/invoiceANNOTATIONS/)
    ↓
YOLO Labels (data/yolo_labels/)
    ↓
Split Dataset (data/yolo_dataset/)
    ↓
[Optional] Augmented Dataset (data/yolo_dataset_augmented/)
    ↓
Trained Model (outputs/yolo_results/invoice_yolo/)
```

---

## ⚙️ Requirements

- Python 3.8+
- pdf2image
- PyMuPDF (fitz)
- OpenCV (cv2)
- ultralytics (YOLOv8)
- PyTorch with CUDA support
- PIL/Pillow
- NumPy

---

## 📝 Notes

- All scripts assume execution from the project root directory
- Preprocessing scripts 1-4 are designed to run sequentially
- Augmentation is optional but recommended for better model generalization
- Training script is optimized for GTX 1050 (4GB VRAM) - adjust batch size for other GPUs
- Original files remain in `src/`, `scripts/`, and `notebooks/` - these are copies for convenience
