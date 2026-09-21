# Scanned Engineering Drawing OCR & Title Box Extractor

A 100% local, offline CLI pipeline to detect title blocks, locate the exact title box, and extract drawing titles from technical drawings in French, Portuguese, and English.

## Key Features
- **100% Local & Offline**: Powered by PaddleOCR and PyTesseract — zero external cloud/API calls.
- **Multi-Format Support**: Reads **`.pdf`**, **`.tiff`**, **`.tif`**, **`.png`**, **`.jpg`**, **`.jpeg`**, **`.bmp`**, and **`.webp`**.
- **Complete Punctuation Preservation**: Strictly preserves all symbols (`-`, `/`, `\`, `&`, `.`, `_`, `()`, `#`, etc.) in the title.
- **Single-File Codebase**: Everything implemented cleanly in [`main.py`](file:///D:/app/OCR/main.py) following [`codestyle.md`](file:///D:/app/OCR/codestyle.md).
- **Exact Single-Title Output**: The `.txt` file contains **ONLY the extracted drawing title**, with no extra headers or metadata.

---

## Quick Start CLI Usage

### 1. Process a Single File (PDF, TIFF, TIF, PNG, JPG, JPEG):
```powershell
uv run main.py --input test_samples/french_drawing.pdf --output results --overwrite --lang fr
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
2. `[filename]_boxes.png`: OpenCV visualization showing all candidate cells in blue, the title block boundary in green, and the **exact title box highlighted in bold red with an `EXTRACTED TITLE BOX` badge**.
3. `[filename]_extracted.txt`: Text file containing **ONLY the extracted title text** (with all punctuation like `-`, `/`, `\`, `&`, `.` strictly preserved).
