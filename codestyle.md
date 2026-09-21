# Coding Style & Architectural Guidelines

## 1. Core Philosophy: Simplicity & Library-First
- **Use Existing Libraries First**: Leverage built-in functions from `pymupdf` (fitz), `opencv-python` (`cv2`), `pytesseract`, `paddleocr`, and `pathlib` instead of writing custom algorithms or wheel-reinventions.
- **Do Not Write Custom Functions First**: Use direct library calls wherever possible. Custom helper functions should only be written as a last resort for minimal glue logic.
- **No Over-Engineering**: Avoid unnecessary abstract classes, factory patterns, or complex inheritance hierarchies. Write straightforward procedural/functional code.
- **Single-File Architecture**: Keep all execution and extraction logic in a single, easily readable file: `main.py`.

---

## 2. Code Structure & Readability Rules
1. **Clear, Readable Functions**:
   - Each function does one plain task clearly (e.g., `load_image`, `detect_boxes`, `run_ocr`, `extract_title_and_text`, `save_outputs`).
   - Function names and variable names must be descriptive and obvious.
2. **Minimal Complexity**:
   - Flat is better than nested.
   - Use standard library `argparse` for CLI commands.
   - Use standard `pathlib.Path` for file handling.
3. **Robust Fallbacks**:
   - If OpenCV contour detection doesn't find internal lines, fall back to standard bottom-right corner crop.
   - If PaddleOCR is unavailable or fails, fall back to Tesseract.

---

## 3. Mandatory 3 Output Types
For every processed drawing (`<filename>`), the output folder must contain:
1. **Cropped Image (`<filename>_crop.png`)**: High-resolution cropped image of the title block.
2. **OpenCV Detected Boxes Visualizer (`<filename>_boxes.png`)**: Drawing/ROI with all detected boxes/cells drawn with `cv2.rectangle`, clearly highlighting the exact box from which the title was extracted.
3. **Exact Extracted Text File (`<filename>_extracted.txt`)**: Plain text file containing the exact text as written in the PDF/drawing, along with the identified title.

---

## 4. CLI Command Contract
Execute cleanly using `uv run`:
```powershell
uv run main.py --input <path_to_drawing_or_folder> --output <output_folder> --overwrite --lang auto
```
- `--input` (`-i`): Single PDF, single image (PNG/JPG/TIFF), or a directory of drawings.
- `--output` (`-o`): Output folder name.
- `--overwrite`: Overwrites existing outputs if present.
- `--lang` (`-l`): Language selection (`auto`, `fr`, `pt`, `en`). Default: `auto`.
- `--engine` (`-e`): OCR engine (`hybrid`, `paddleocr`, `tesseract`). Default: `hybrid`.
