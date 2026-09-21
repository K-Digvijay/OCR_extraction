# Scanned Engineering Drawing OCR & Title Box Extractor

Fast, 100% local, offline CLI pipeline to detect title blocks, locate the exact title box, and extract drawing titles from technical drawings in French, Portuguese, and English.

## Key Features
- **Ultra-Fast (1-2 seconds per drawing)**: Streamlined local processing with PyMuPDF, OpenCV, and local Tesseract with multilingual traineddata (`fra`, `por`, `eng`).
- **Multi-Format Support**: Reads **`.pdf`**, **`.tiff`**, **`.tif`**, **`.png`**, **`.jpg`**, **`.jpeg`**, **`.bmp`**, and **`.webp`**.
- **Complete Punctuation Preservation**: Strictly preserves all symbols (`-`, `/`, `\`, `&`, `.`, `_`, `()`, `#`, etc.) in the title.
- **Strict Title Validation**: If a drawing has no title written, it automatically leaves the red bounding box and `.txt` file empty (no false guessing).
- **Single-File Codebase**: Everything implemented cleanly in [`main.py`](file:///D:/app/OCR/main.py) following [`codestyle.md`](file:///D:/app/OCR/codestyle.md).

---

## Quick Start CLI Usage

### 1. Process a Single File (PDF, TIFF, TIF, PNG, JPG, JPEG):
```powershell
uv run main.py --input test_samples/vintage_french_drawing.pdf --output results --overwrite --lang fr
```
```powershell
uv run main.py --input test_samples/punct_test.tiff --output results --overwrite
```
```powershell
uv run main.py --input test_samples/punct_test.tif --output results --overwrite
```
```powershell
uv run main.py --input test_samples/punct_test.jpeg --output results --overwrite
```

### 2. Process an Entire Folder of Mixed Drawings (Batch Mode):
```powershell
uv run main.py --input test_samples --output results --overwrite --lang auto
```

---

## The 3 Output Files Generated Per Drawing
Inside your `--output` folder, every processed drawing produces:
1. `[filename]_crop.png`: High-resolution crop of the detected title block.
2. `[filename]_boxes.png`: OpenCV visualization showing the title block boundary in green, and the **exact title box highlighted in bold red with an `EXTRACTED TITLE` badge** (or left clean if no title is present).
3. `[filename]_extracted.txt`: Text file containing **ONLY the extracted title text** (or empty if no title was written).
