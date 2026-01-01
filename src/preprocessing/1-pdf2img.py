#!/usr/bin/env python3
from pdf2image import convert_from_path
from pathlib import Path

" 3300×2550 "

PDF_DIR = "data/invoice" #path
IMG_DIR = "data/processed/invoiceIMG" #path


Path(IMG_DIR).mkdir(parents=True, exist_ok=True)

for pdf in Path(PDF_DIR).glob("*.pdf"):
    pages = convert_from_path(pdf, dpi=300)
    for i, page in enumerate(pages):
        out = Path(IMG_DIR) / f"{pdf.stem}_page{i+1}.png"
        page.save(out, "PNG")
