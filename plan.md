# Implementation Plan: 100% Local Scanned PDF Drawing OCR & Title Box Extraction

## 1. Goal & Privacy Contract
Build a 100% local, offline CLI extraction pipeline in a **single, simple file (`main.py`)** adhering strictly to [`codestyle.md`](file:///D:/app/OCR/codestyle.md):
- **100% Local & Offline**: Zero network calls, zero external LLMs.
- **Library-First Philosophy**: Rely on standard library calls in `fitz` (PyMuPDF), `cv2` (OpenCV), and `pytesseract` / `paddleocr`. Avoid custom wrapper classes or reinvented wheels.
- **Single File (`main.py`)**: All functions are written in a simple, flat, understandable way without unnecessary complexity.
- **Multilingual Support**: Portuguese (`TÍTULO`, `DESIGNAÇÃO`), French (`TITRE`, `DÉSIGNATION`), and English.

---

## 2. Mandatory 3 Output Types
For every drawing processed (e.g. `sheet1.pdf`), the pipeline produces exactly the **3 required outputs** inside the `--output` folder:

1. 🖼️ **Cropped Image (`<name>_crop.png`)**:
   - The cropped high-resolution image of the detected title box / title region.
2. 📦 **OpenCV Detected Boxes Visualizer (`<name>_boxes.png`)**:
   - Visual output created via `cv2.rectangle()` displaying all bounding boxes/cells detected by OpenCV across the drawing/title block.
   - Distinctly highlights the **exact box from which the title was extracted** (e.g., in prominent green/red with an OpenCV text label: `EXTRACTED TITLE`).
3. 📝 **Exact Extracted Text File (`<name>_extracted.txt`)**:
   - Plain `.txt` file containing the **exact text written in the PDF file / title box**, character-for-character as recognized by the local OCR, along with the isolated drawing title header.
   - Example format:
     ```text
     ============================================================
     IDENTIFIED DRAWING TITLE:
     PLAN DE TUYAUTERIE DU CIRCUIT DE REFROIDISSEMENT
     ============================================================
     
     EXACT TEXT EXTRACTED FROM TITLE BOX:
     ------------------------------------------------------------
     SOCIÉTÉ NATIONALE D'INGÉNIERIE
     PROJET: USINE CENTRALE HYDRO-ÉLECTRIQUE
     DÉSIGNATION: PLAN DE TUYAUTERIE DU CIRCUIT DE REFROIDISSEMENT
     N° DE PLAN: FR-CWS-2026-01
     INDICE: B
     DATE: 15/04/2026
     ÉCHELLE: 1:50
     DESSINÉ PAR: J. DUPONT
     APPROUVÉ PAR: M. MARTIN
     ------------------------------------------------------------
     ```

---

## 3. Implementation Details in `main.py`

All logic resides in `main.py` using simple, library-first functions:

```
main.py
  ├── load_image(file_path, dpi=300) -> np.ndarray [PyMuPDF fitz or cv2.imread]
  ├── detect_boxes(image) -> (title_box_rect, all_cell_rects) [cv2.morphologyEx + findContours]
  ├── run_ocr(image_crop, engine, lang) -> (raw_text, text_boxes) [pytesseract or paddleocr]
  ├── extract_title(text_boxes, raw_text, lang) -> (title_text, title_rect)
  ├── save_outputs(image, title_box_rect, all_cell_rects, title_rect, raw_text, title_text, output_dir, file_stem)
  └── main() [argparse CLI runner]
```

### Step 1: Load Image (`load_image`)
- Scanned PDF: `fitz.open(path)[0].get_pixmap(dpi=300)` converted to numpy image.
- Scanned Image (`.png`, `.jpg`, `.jpeg`, `.tiff`): `cv2.imread(path)`.

### Step 2: Detect Boxes via OpenCV (`detect_boxes`)
- Focuses on the standard lower-right region (bottom 25% height, right 35% width).
- Uses `cv2.getStructuringElement` with `cv2.morphologyEx` to detect horizontal and vertical table lines.
- Finds all internal grid cells using `cv2.findContours`.
- Returns both the outer title box coordinates and the individual cell rectangles.

### Step 3: Run Local OCR (`run_ocr`)
- Runs Tesseract 5.5 / PaddleOCR on the cropped title box.
- Returns the exact extracted text string and bounding boxes for each line/token.

### Step 4: Extract Title (`extract_title`)
- Scans tokens using the Portuguese/French/English dictionary (`TITRE`, `DÉSIGNATION`, `TÍTULO`, `DESIGNAÇÃO`, `TITLE`).
- Locates the exact cell/box containing the drawing title text.
- Fallback: Selects the text box with the largest font height in the title block.

### Step 5: Save the 3 Outputs (`save_outputs`)
- Writes `<name>_crop.png` using `cv2.imwrite`.
- Draws all candidate boxes on the drawing/ROI in blue/cyan using `cv2.rectangle`, and draws the specific title box in bold red/green with text `cv2.putText`, then saves `<name>_boxes.png`.
- Writes `<name>_extracted.txt` containing the exact extracted text and identified title.

---

## 4. CLI Specifications (`uv run`)

### Command:
```powershell
uv run main.py --input <path_to_pdf_or_image_or_folder> --output <folder_name> --overwrite --lang auto
```

### Options:
- `--input` / `-i` *(required)*: File path (`drawing.pdf`, `scan.png`) or directory of drawings.
- `--output` / `-o` *(required)*: Output folder name.
- `--overwrite`: Overwrite existing output files in the folder.
- `--lang` / `-l`: `auto` (default), `fr`, `pt`, `en`.
- `--engine` / `-e`: `hybrid` (default), `tesseract`, `paddleocr`.

---

## 5. Verification Plan

### Automated Tests:
1. Generate test French and Portuguese drawing PDFs locally with standard title boxes.
2. Run `uv run main.py --input test_drawing.pdf --output test_results --overwrite`.
3. Verify that all **3 output files** are produced:
   - `test_results/test_drawing_crop.png` exists and is a valid image.
   - `test_results/test_drawing_boxes.png` exists and contains visible OpenCV bounding boxes.
   - `test_results/test_drawing_extracted.txt` exists and contains the exact text and identified title.
4. Verify `--overwrite` skips or replaces cleanly.

### Manual Verification:
- User can run on their scanned Portuguese and French drawings and view the generated `.png` and `.txt` files.
