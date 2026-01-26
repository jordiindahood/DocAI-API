#!/usr/bin/env python3
"""
Convert existing invoice annotations to OCR JSON format
Reads from data/annotations/invoiceANNOTATIONS/ and converts to data/raw/invoices/ocr_json/
"""

import json
from pathlib import Path
from PIL import Image

# Input: Existing annotations
ANN_DIR = Path("data/annotations/invoiceANNOTATIONS")
IMG_DIR = Path("data/processed/invoiceIMG")

# Output: OCR JSON format
OCR_JSON_DIR = Path("data/raw/invoices/ocr_json")
OCR_JSON_DIR.mkdir(parents=True, exist_ok=True)

# Create images directory if needed
IMAGES_DIR = Path("data/raw/invoices/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def convert_annotation_to_ocr_json(ann_file: Path):
    """
    Convert annotation file to OCR JSON format
    
    Input format (data/annotations/invoiceANNOTATIONS/):
    [
      [  # Page 0
        {"text": "Invoice", "bbox": [x0, y0, x1, y1], "page": 0},
        ...
      ],
      [  # Page 1
        ...
      ]
    ]
    
    Output format (data/raw/invoices/ocr_json/):
    {
      "image_path": "data/raw/invoices/images/0001_page1.png",
      "width": 2480,
      "height": 3508,
      "words": [
        {"text": "Invoice", "bbox": [x0, y0, x1, y1]}
      ]
    }
    """
    pdf_name = ann_file.stem
    
    try:
        with open(ann_file, 'r', encoding='utf-8') as f:
            pages = json.load(f)
    except Exception as e:
        print(f"Error loading {ann_file}: {e}")
        return 0
    
    converted_count = 0
    
    # Process each page
    for page_idx, page_data in enumerate(pages):
        # Find corresponding image
        img_file = IMG_DIR / f"{pdf_name}_page{page_idx+1}.png"
        
        if not img_file.exists():
            print(f"Warning: Image not found: {img_file}, skipping page {page_idx+1}")
            continue
        
        # Load image to get dimensions
        try:
            img = Image.open(img_file)
            img_width, img_height = img.size
        except Exception as e:
            print(f"Error loading image {img_file}: {e}")
            continue
        
        # Copy image to images directory (or use symlink)
        target_img_path = IMAGES_DIR / img_file.name
        if not target_img_path.exists():
            # Copy image
            import shutil
            shutil.copy2(img_file, target_img_path)
        
        # Convert page data to OCR JSON format
        words = []
        for item in page_data:
            text = item.get("text", "").strip()
            bbox = item.get("bbox", [])
            
            if not text or len(bbox) < 4:
                continue
            
            words.append({
                "text": text,
                "bbox": bbox  # Already in image coordinates
            })
        
        if not words:
            continue
        
        # Create OCR JSON structure
        ocr_json = {
            "image_path": str(target_img_path),
            "width": img_width,
            "height": img_height,
            "words": words
        }
        
        # Save OCR JSON file (one per page)
        ocr_json_file = OCR_JSON_DIR / f"{pdf_name}_page{page_idx+1}.json"
        with open(ocr_json_file, 'w', encoding='utf-8') as f:
            json.dump(ocr_json, f, indent=2, ensure_ascii=False)
        
        converted_count += 1
        print(f"  ✓ Converted {pdf_name} page {page_idx+1} ({len(words)} words)")
    
    return converted_count


def main():
    """Convert all annotation files to OCR JSON format"""
    print("="*60)
    print("Converting Annotations to OCR JSON Format")
    print("="*60)
    print(f"Input: {ANN_DIR}")
    print(f"Output: {OCR_JSON_DIR}")
    print("="*60)
    print()
    
    if not ANN_DIR.exists():
        print(f"Error: Annotation directory not found: {ANN_DIR}")
        return
    
    # Find all annotation files
    ann_files = sorted(ANN_DIR.glob("*.json"))
    print(f"Found {len(ann_files)} annotation files")
    print()
    
    total_converted = 0
    for ann_file in ann_files:
        print(f"Converting: {ann_file.name}")
        count = convert_annotation_to_ocr_json(ann_file)
        total_converted += count
    
    print()
    print("="*60)
    print(f"Conversion complete!")
    print(f"Total pages converted: {total_converted}")
    print(f"OCR JSON files saved to: {OCR_JSON_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
