import os
from pathlib import Path

# Use local tessdata with fra, por, eng
LOCAL_TESSDATA = Path(__file__).parent / "tessdata"
if LOCAL_TESSDATA.exists():
    os.environ["TESSDATA_PREFIX"] = str(LOCAL_TESSDATA.resolve())

import argparse
import re
import sys
import unicodedata
import cv2
import numpy as np
import pymupdf  # PyMuPDF
import pytesseract
from PIL import Image

# Supported drawing extensions
SUPPORTED_EXTENSIONS = {".pdf", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".webp"}

# Multilingual Title Keywords (French, Portuguese, English)
TITLE_KEYWORDS = [
    # French
    "TITRE", "DESIGNATION", "INTITULE", "OBJET", "TITRE DU PLAN",
    # Portuguese
    "TITULO", "DESIGNACAO", "DESCRICAO", "DENOMINACAO", "NOME DO PROJETO",
    # English
    "TITLE", "DRAWING TITLE", "DWG TITLE", "DESCRIPTION", "PROJECT"
]


def normalize_for_match(s: str) -> str:
    """Normalize string ONLY for keyword matching. Never alters the extracted title text."""
    if not s:
        return ""
    nfkd = unicodedata.normalize('NFKD', s)
    no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = re.sub(r'[°º\.:_\-\[\]/\\()]', ' ', no_accents)
    return " ".join(clean.upper().split())


def load_image(file_path: Path, dpi: int = 300) -> np.ndarray:
    """
    Fast loader for PDF, TIFF, PNG, JPEG.
    Uses PyMuPDF for PDFs (fast 300 DPI rendering) and Pillow/OpenCV for images.
    """
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        doc = pymupdf.open(str(file_path))
        if len(doc) == 0:
            raise ValueError(f"PDF {file_path} is empty.")
        page = doc[0]
        pix = page.get_pixmap(dpi=dpi)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, pix.n))
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif pix.n == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        doc.close()
        return img
    elif suffix in (".tiff", ".tif"):
        pil_img = Image.open(str(file_path))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(str(file_path))
        if img is None:
            pil_img = Image.open(str(file_path)).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img


def get_title_block_roi(image: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Locates the Title Block at the bottom-right corner of the drawing sheet (ISO / ASME standard).
    Fast and robust: finds the bounding lines in the lower-right area.
    """
    h, w = image.shape[:2]

    # Search in bottom-right corner: lower 28% height, right 38% width
    tb_h = int(h * 0.28)
    tb_w = int(w * 0.38)
    x1 = w - tb_w - int(w * 0.015)
    y1 = h - tb_h - int(h * 0.015)

    # Refine boundary using OpenCV lines if prominent box borders exist
    corner_gray = cv2.cvtColor(image[y1:h, x1:w], cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(corner_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, tb_w // 20), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, tb_h // 15)))
    lines = cv2.bitwise_or(
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h),
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    )

    cnts, _ = cv2.findContours(lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_rect = None
    max_area = 0
    min_area = (tb_w * tb_h) * 0.15

    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if min_area < area < (tb_w * tb_h * 0.98) and area > max_area:
            max_area = area
            best_rect = (x1 + bx, y1 + by, bw, bh)

    if best_rect is not None:
        rx, ry, rw, rh = best_rect
    else:
        rx, ry, rw, rh = x1, y1, tb_w, tb_h

    crop = image[ry:ry+rh, rx:rx+rw]
    return (rx, ry, rw, rh), crop


def extract_title_and_box(
    crop_img: np.ndarray,
    tb_coords: tuple[int, int, int, int],
    lang: str = "auto"
) -> tuple[str, tuple[int, int, int, int] | None]:
    """
    Runs fast local OCR once on the title block crop.
    Finds the title text and its exact bounding box.
    If no title is found, returns ("", None) — leaves it empty as requested.
    Strictly preserves all punctuation (-, /, \, &, ., etc.).
    """
    rx, ry, rw, rh = tb_coords

    # Select best Tesseract language
    avail = pytesseract.get_languages()
    chosen_lang = "eng"
    if lang == "fr" and "fra" in avail:
        chosen_lang = "fra"
    elif lang == "pt" and "por" in avail:
        chosen_lang = "por"
    elif "fra" in avail and "por" in avail:
        chosen_lang = "fra+por+eng"
    elif "fra" in avail:
        chosen_lang = "fra+eng"
    elif "por" in avail:
        chosen_lang = "por+eng"

    # Fast OCR with bounding boxes in one pass
    data = pytesseract.image_to_data(
        crop_img,
        lang=chosen_lang,
        config="--psm 3",
        output_type=pytesseract.Output.DICT
    )

    # Group words into lines
    lines = {}
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        word = data['text'][i].strip()
        conf = float(data['conf'][i]) if data['conf'][i] != '-1' else 0.0
        if word:
            ln = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            lines.setdefault(ln, []).append({
                "text": word,
                "box": (x, y, w, h),
                "conf": conf
            })

    line_list = []
    for ln, words in lines.items():
        full_line = " ".join(w["text"] for w in words).strip()
        if full_line:
            min_x = min(w["box"][0] for w in words)
            min_y = min(w["box"][1] for w in words)
            max_x = max(w["box"][0] + w["box"][2] for w in words)
            max_y = max(w["box"][1] + w["box"][3] for w in words)
            line_list.append({
                "text": full_line,
                "box": (min_x, min_y, max_x - min_x, max_y - min_y),
                "words": words
            })

    extracted_title = ""
    title_box = None

    # Step 1: Search for explicit title keywords
    for idx, item in enumerate(line_list):
        norm = normalize_for_match(item["text"])
        for kw in TITLE_KEYWORDS:
            if kw in norm:
                # Try to extract inline title value
                # e.g., "DESIGNATION : PLAN DE MASSE / BAT-A \ 01"
                pattern = rf"\b{re.escape(kw)}\b\s*[:\-\—]?"
                match = re.split(pattern, item["text"], flags=re.IGNORECASE)
                val = match[1].strip() if len(match) > 1 else ""

                if val and len(val) > 2:
                    extracted_title = val
                    title_box = (rx + item["box"][0], ry + item["box"][1], item["box"][2], item["box"][3])
                    break
                elif idx + 1 < len(line_list):
                    # Value is on the next line immediately below the keyword
                    next_item = line_list[idx + 1]
                    next_text = next_item["text"].strip()
                    if len(next_text) > 3:
                        extracted_title = next_text
                        title_box = (rx + next_item["box"][0], ry + next_item["box"][1], next_item["box"][2], next_item["box"][3])
                        break
        if extracted_title:
            break

    # Step 2: Fallback - CAD convention: drawing title is the largest text height
    if not extracted_title and line_list:
        best_item = None
        max_h = 0
        for item in line_list:
            t = item["text"].strip()
            # Ignore company names with INC, SAS, LTDA, SA or very short codes
            norm = normalize_for_match(t)
            is_company = any(c in norm for c in ["SAS", "LTDA", "INC", "CORP", "COMPANY", "COMPANHIA", "SOCIETE", "BUREAU"])
            is_dwg_no = any(d in norm for d in ["DWG", "REF", "REV", "ECHELLE", "ESCALA", "DATE", "DATA"])
            if len(t) > 5 and not is_company and not is_dwg_no:
                h = item["box"][3]
                if h > max_h:
                    max_h = h
                    best_item = item

        if best_item is not None:
            extracted_title = best_item["text"].strip()
            title_box = (rx + best_item["box"][0], ry + best_item["box"][1], best_item["box"][2], best_item["box"][3])

    # If title is not written / not found, leave it as requested ("if the title is not written to take the bounding box leave it")
    if not extracted_title or len(extracted_title) < 3:
        return "", None

    # Clean typographical dashes to standard hyphen while strictly preserving all punctuation (-, /, \, &, .)
    extracted_title = re.sub(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D—–]', '-', extracted_title).strip()

    # Add a comfortable margin around the title bounding box
    if title_box is not None:
        bx, by, bw, bh = title_box
        margin_x = 10
        margin_y = 6
        title_box = (
            max(0, bx - margin_x),
            max(0, by - margin_y),
            bw + (margin_x * 2),
            bh + (margin_y * 2)
        )

    return extracted_title, title_box


def save_outputs(
    image: np.ndarray,
    tb_coords: tuple[int, int, int, int],
    title_box: tuple[int, int, int, int] | None,
    title_text: str,
    output_dir: Path,
    file_stem: str
):
    """
    Saves the 3 required outputs:
    1. <name>_crop.png: Cropped image of the title block.
    2. <name>_boxes.png: OpenCV visualizer showing the title box (or left clean if no title).
    3. <name>_extracted.txt: ONLY the extracted title written (or empty if not found).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rx, ry, rw, rh = tb_coords

    # Output 1: Crop of Title Block
    crop_path = output_dir / f"{file_stem}_crop.png"
    crop_img = image[ry:ry+rh, rx:rx+rw]
    cv2.imwrite(str(crop_path), crop_img)

    # Output 2: OpenCV Boxes Visualizer
    boxes_path = output_dir / f"{file_stem}_boxes.png"
    vis = image.copy()

    # Draw Title Block perimeter in green
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
    cv2.rectangle(vis, (rx, max(0, ry - 30)), (rx + 200, ry), (0, 200, 0), -1)
    cv2.putText(vis, "TITLE BLOCK", (rx + 5, max(18, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # If title was found, draw the prominent red title box
    if title_box is not None:
        tx, ty, tw, th = title_box
        cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 4)
        tag_y = max(24, ty - 6)
        cv2.rectangle(vis, (tx, tag_y - 24), (tx + 220, tag_y + 2), (0, 0, 220), -1)
        cv2.putText(vis, "EXTRACTED TITLE", (tx + 5, tag_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Zoomed ROI output with 15% margin
    margin = int(max(rw, rh) * 0.15)
    vy1 = max(0, ry - margin)
    vy2 = min(image.shape[0], ry + rh + margin)
    vx1 = max(0, rx - margin)
    vx2 = min(image.shape[1], rx + rw + margin)
    zoomed = vis[vy1:vy2, vx1:vx2]
    cv2.imwrite(str(boxes_path), zoomed)

    # Output 3: Only the title written in the txt file (empty if not found)
    txt_path = output_dir / f"{file_stem}_extracted.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        if title_text:
            f.write(title_text.strip() + "\n")


def process_file(file_path: Path, output_dir: Path, overwrite: bool = False, lang: str = "auto", dpi: int = 300) -> bool:
    """Fast processing of a single drawing file in under 2 seconds."""
    file_stem = file_path.stem
    txt_path = output_dir / f"{file_stem}_extracted.txt"

    if txt_path.exists() and not overwrite:
        print(f"[SKIP] {file_path.name} already processed.")
        return True

    print(f"\n[START] {file_path.name}")
    try:
        # 1. Load image (PDF, TIFF, PNG, JPEG)
        img = load_image(file_path, dpi=dpi)

        # 2. Get Title Block ROI (bottom-right)
        tb_coords, crop_img = get_title_block_roi(img)

        # 3. Extract Title and its exact bounding box
        title, title_box = extract_title_and_box(crop_img, tb_coords, lang=lang)

        if title:
            print(f"  -> Title Extracted: '{title}'")
        else:
            print(f"  -> [No Title Found] Leaving title and bounding box empty.")

        # 4. Save the 3 outputs
        save_outputs(
            image=img,
            tb_coords=tb_coords,
            title_box=title_box,
            title_text=title,
            output_dir=output_dir,
            file_stem=file_stem
        )

        print(f"[DONE] Saved: {file_stem}_crop.png, {file_stem}_boxes.png, {file_stem}_extracted.txt")
        return True

    except Exception as e:
        print(f"[ERROR] Failed {file_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Fast Local Scanned PDF/TIFF/PNG/JPEG Drawing Title Extractor")
    parser.add_argument("--input", "-i", required=True, help="Path to input drawing file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--lang", "-l", default="auto", choices=["auto", "fr", "pt", "en"], help="Language: auto, fr, pt, en")
    parser.add_argument("--dpi", type=int, default=300, help="PDF rendering DPI (default: 300)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        print(f"[ERROR] Path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    files = []
    if input_path.is_file():
        if input_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(input_path)
        else:
            print(f"[ERROR] Unsupported file: {input_path.suffix}", file=sys.stderr)
            sys.exit(1)
    elif input_path.is_dir():
        for p in input_path.iterdir():
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(p)

    print(f"Processing {len(files)} drawing(s)...")
    for f in files:
        process_file(f, output_dir=output_dir, overwrite=args.overwrite, lang=args.lang, dpi=args.dpi)


if __name__ == "__main__":
    main()
