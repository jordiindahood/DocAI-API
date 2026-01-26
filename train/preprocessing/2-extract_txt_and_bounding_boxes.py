#!/usr/bin/env python3

import fitz  # PyMuPDF
import json
from pathlib import Path
from PIL import Image

PDF_DIR = Path("data/invoice")
IMG_DIR = Path("data/processed/invoiceIMG")
OUT_DIR = Path("data/annotations/invoiceANNOTATIONS")

OUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in PDF_DIR.glob("*.pdf"):
    doc = fitz.open(pdf_path)
    all_pages = []

    for page_num, page in enumerate(doc):
        words = page.get_text("words")
        # (x0, y0, x1, y1, text, block_no, line_no, word_no)

        # Get corresponding image size
        img_file = IMG_DIR / f"{pdf_path.stem}_page{page_num+1}.png"
        if not img_file.exists():
            print(f"Image not found: {img_file}, skipping page")
            continue
        img = Image.open(img_file)
        img_w, img_h = img.size

        # PDF page size
        pdf_w = page.rect.width
        pdf_h = page.rect.height

        scale_x = img_w / pdf_w
        scale_y = img_h / pdf_h

        page_data = []
        for w in words:
            x0, y0, x1, y1, text = w[:5]

            # Scale to image coordinates
            x0_img = x0 * scale_x
            y0_img = y0 * scale_y
            x1_img = x1 * scale_x
            y1_img = y1 * scale_y

            page_data.append({
                "text": text,
                "bbox": [x0_img, y0_img, x1_img, y1_img],
                "page": page_num
            })

        all_pages.append(page_data)

    out_file = OUT_DIR / f"{pdf_path.stem}.json"
    with open(out_file, "w") as f:
        json.dump(all_pages, f, indent=2)

    print(f"Saved: {out_file}")
