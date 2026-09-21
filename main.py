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

# Multilingual Title Anchor Keywords (French, Portuguese, English)
TITLE_ANCHORS = [
    # French
    "TITRE", "DESIGNATION", "INTITULE", "OBJET", "SUJET", "LIBELLE",
    "TITRE DU PLAN", "NOM DU PLAN", "TITRE DU PROJET",
    # Portuguese
    "TITULO", "DESIGNACAO", "DESCRICAO", "DENOMINACAO", "NOME DO PROJETO",
    "TITULO DO DESENHO", "IDENTIFICACAO",
    # English
    "TITLE", "DRAWING TITLE", "DWG TITLE", "DESCRIPTION", "PROJECT"
]

ANCHOR_PREFIX_REGEX = (
    r"^(?:.*?)(?:TITRE|DESIGNATION|D[EÉ]SIGNATION|INTITUL[EÉ]|OBJET|SUJET|LIBELL[EÉ]|"
    r"T[IÍ]TULO|DESIGNA[CÇ][AÃ]O|DESCRI[CÇ][AÃ]O|DENOMINA[CÇ][AÃ]O|TITLE|DESCRIPTION|PROJECT)"
    r"(?:\s*(?:DO\s*DESENHO|DU\s*PLAN|DU\s*PROJET))?"
    r"(?:\s*/\s*(?:TITRE|DESIGNATION|D[EÉ]SIGNATION|T[IÍ]TULO|TITLE)(?:\s*(?:DO\s*DESENHO|DU\s*PLAN))?)*"
    r"\s*[:\-\—\.]*\s*"
)

NON_TITLE_FIELDS_REGEX = (
    r"^(?:DWG\s*NO|N[°º]?\s*D[EO]\s*(?:PLAN|DESENHO)|REF\s*/|REF\b|REV(?:ISAO)?\b|"
    r"INDICE\b|DATE\b|DATA\b|ECHELLE\b|ESCALA\b|SCALE\b|FORMAT\b|PAGE\b|SHEET\b|FOLHA\b)"
)


def normalize_for_match(s: str) -> str:
    """Normalize string ONLY for anchor keyword matching. Never alters the extracted title text."""
    if not s:
        return ""
    nfkd = unicodedata.normalize('NFKD', s)
    no_accents = "".join([c for c in nfkd if not unicodedata.combining(c)])
    clean = re.sub(r'[°º\.:_\-\[\]/\\()]', ' ', no_accents)
    return " ".join(clean.upper().split())


def load_image(file_path: Path, dpi: int = 300) -> np.ndarray:
    """
    Load any supported drawing format: PDF, TIFF, TIF, PNG, JPEG, BMP.
    Uses PyMuPDF for PDFs and Pillow/OpenCV for image files.
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


def get_title_block_and_cells(image: np.ndarray) -> tuple[tuple[int, int, int, int], list[tuple[int, int, int, int]], np.ndarray]:
    """
    Locates the Title Block and detects internal table cells in the bottom-right corner.
    Returns:
        tb_coords: (rx, ry, rw, rh) of the title block
        cells: list of (cx, cy, cw, ch) for all detected cells in global coordinates
        crop: the cropped title block image
    """
    h, w = image.shape[:2]

    # Search window: bottom-right corner (lower 32% height, right 42% width)
    roi_y1 = int(h * 0.68)
    roi_x1 = int(w * 0.58)
    roi = image[roi_y1:h, roi_x1:w]
    roi_h, roi_w = roi.shape[:2]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Detect horizontal and vertical grid lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, roi_w // 25), 1))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, roi_h // 20)))
    grid = cv2.bitwise_or(
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h),
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v)
    )

    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells = []
    min_area = (roi_w * roi_h) * 0.003
    max_area = (roi_w * roi_h) * 0.85

    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        touches_border = (bx <= 2 or by <= 2 or (bx + bw) >= roi_w - 2 or (by + bh) >= roi_h - 2)
        if min_area < area < max_area and not touches_border:
            cells.append((roi_x1 + bx, roi_y1 + by, bw, bh))

    if cells:
        all_x1 = min(c[0] for c in cells)
        all_y1 = min(c[1] for c in cells)
        all_x2 = max(c[0] + c[2] for c in cells)
        all_y2 = max(c[1] + c[3] for c in cells)
        tb_coords = (all_x1, all_y1, all_x2 - all_x1, all_y2 - all_y1)
    else:
        fb_w = int(w * 0.32)
        fb_h = int(h * 0.20)
        tb_coords = (w - fb_w - int(w * 0.015), h - fb_h - int(h * 0.015), fb_w, fb_h)
        cells = [tb_coords]

    rx, ry, rw, rh = tb_coords
    crop = image[ry:ry+rh, rx:rx+rw]
    return tb_coords, cells, crop


def extract_content_from_text(raw_text: str) -> str:
    """
    Extracts the drawing title content from raw text by stripping the anchor label prefix.
    Strictly preserves all punctuation (-, /, \\, &, ., (), :, etc.).
    """
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if not lines:
        return ""

    anchor_idx = -1
    for idx, ln in enumerate(lines):
        norm = normalize_for_match(ln)
        for anchor in TITLE_ANCHORS:
            if re.search(rf"\b{anchor}\b", norm):
                anchor_idx = idx
                break
        if anchor_idx >= 0:
            break

    if anchor_idx < 0:
        return ""

    anchor_line = lines[anchor_idx]
    cleaned_anchor_line = re.sub(ANCHOR_PREFIX_REGEX, "", anchor_line, flags=re.IGNORECASE).strip()

    content_lines = []
    if cleaned_anchor_line:
        content_lines.append(cleaned_anchor_line)

    for ln in lines[anchor_idx + 1:]:
        norm = normalize_for_match(ln)
        if re.search(NON_TITLE_FIELDS_REGEX, norm, flags=re.IGNORECASE):
            break
        content_lines.append(ln)

    content = " ".join(content_lines).strip()
    content = re.sub(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D—–]', '-', content)
    content = content.strip("[]| ")
    return content


NON_TITLE_WORDS = {
    "DWG", "NO", "REF", "REV", "REVISAO", "INDICE", "DATE", "DATA",
    "SCALE", "ECHELLE", "ESCALA", "SIZE", "PAGE", "SHEET", "FOLHA",
    "WEIGHT", "POIDS", "PESO", "DRAWN", "APPROVED", "APPR", "REVIEWED",
    "CHECKED", "DESSINE", "DESENHADO", "MAT", "MATERIAL", "FINISH",
    "CASE"
}

TITLE_KEYWORDS = {
    # English
    "PLAN", "SCHEME", "SCHEMA", "DIAGRAM", "WORKS", "INTAKE",
    "DIFFUSER", "SLAB", "REINFORCEMENT", "DETAILS", "SECTION", "ELEVATION",
    "LAYOUT", "ASSEMBLY", "CIRCUIT", "SYSTEM", "PLATE", "FOUNDATION",
    "PIPING", "STRUCTURAL", "PROJECT", "BUILDING", "CONSTRUCTION",
    "INSTALLATION", "GENERAL", "ARRANGEMENT", "SPECIFICATION",
    # French
    "TUYAUTERIE", "REFROIDISSEMENT", "POSTE", "ELECTRIQUE", "BATIMENT",
    "OUVRAGE", "COUPE", "FACADE", "CHAUFFAGE", "DISTRIBUTION",
    # Portuguese
    "TUBULACAO", "AGUA", "HIDRAULICO", "ELETRICO", "EDIFICIO", "OBRA",
    "DETALHES", "CORTE", "FACHADA", "ESTRUTURAL", "LOCALIZACAO", "SISTEMA"
}

NON_TITLE_PHRASES = [
    r"UNLESS\s+OTHERWISE\s+SPECIFIED",
    r"TOLERANCE",
    r"TOLERANCIAS",
    r"MINIMUM\s+LENGTHS",
    r"HORIZONTAL\s+BARS",
    r"BAR\s+SIZE",
    r"ALL\s+DIMENSION",
    r"DO\s+NOT\s+SCALE",
    r"CONSULTANT",
    r"ENGINEERING\s+CONSULTANT",
    r"SUB\s+CONTRACTOR",
    r"FOR\s+REVIEW",
    r"APPROVALS",
    r"APPROVAL",
    r"DRAWN\s+BY",
    r"CHECKED\s+BY",
]


def score_cell_for_title(text: str, w: int, h: int) -> float:
    norm = normalize_for_match(text)
    if not norm:
        return -100.0
    for pat in NON_TITLE_PHRASES:
        if re.search(pat, norm):
            return -100.0
    words = norm.split()
    if len(words) <= 1:
        return -50.0
    if len(words) == 2 and re.search(r"\b(REV|DATE|DATA|SCALE|ECHELLE|DWG)\b", norm):
        return -50.0

    score = 0.0
    score += min(len(words), 15) * 5.0
    score += (w * h) / 1000.0
    for kw in TITLE_KEYWORDS:
        if re.search(rf"\b{kw}\b", norm):
            score += 30.0
    return score


def extract_by_anchor(
    image: np.ndarray,
    tb_coords: tuple[int, int, int, int],
    cells: list[tuple[int, int, int, int]],
    lang: str = "auto"
) -> tuple[str, tuple[int, int, int, int] | None]:
    """
    Finds the ANCHOR title word (TITRE, DÉSIGNATION, TÍTULO, TITLE, etc.).
    The bounding box MUST be that anchor title box/cell.
    Extracts ONLY the title content written under or beside the anchor word.
    If no anchor title word is found, scans and scores title block cells to
    extract the primary title content without losing accuracy.
    """
    rx, ry, rw, rh = tb_coords
    h, w = image.shape[:2]

    # Select OCR languages
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

    # Step 1: Check detected table cells inside Title Block (excluding the whole block container)
    proper_cells = [c for c in cells if (c[2] * c[3]) < (rw * rh * 0.70)]
    for cell in proper_cells:
        cx, cy, cw, ch = cell
        cell_crop = image[cy:cy+ch, cx:cx+cw]
        txt = pytesseract.image_to_string(cell_crop, lang=chosen_lang, config="--psm 6").strip()
        cnt = extract_content_from_text(txt)
        if cnt:
            return cnt, cell

    # Step 2: Fallback to word-level anchor search in the bottom-right ROI
    roi_y1 = int(h * 0.60)
    roi_x1 = int(w * 0.45)
    roi = image[roi_y1:h, roi_x1:w]

    data = pytesseract.image_to_data(
        roi,
        lang=chosen_lang,
        config="--psm 3",
        output_type=pytesseract.Output.DICT
    )

    words = []
    for i in range(len(data['text'])):
        t = data['text'][i].strip()
        if t:
            words.append({
                "text": t,
                "x": data['left'][i],
                "y": data['top'][i],
                "w": data['width'][i],
                "h": data['height'][i]
            })

    # Locate Anchor Word
    anchor_word = None
    for idx, w_info in enumerate(words):
        norm = normalize_for_match(w_info["text"])
        for a in TITLE_ANCHORS:
            if re.search(rf"\b{a}\b", norm):
                anchor_word = w_info
                break
        if anchor_word:
            break

    if anchor_word:
        ax, ay, aw, ah = anchor_word["x"], anchor_word["y"], anchor_word["w"], anchor_word["h"]
        title_words = []

        # Check same-line words (right of anchor)
        same_line = []
        for w_info in words:
            if w_info is anchor_word:
                continue
            if abs(w_info["y"] - ay) < max(ah, 20) and w_info["x"] > (ax + aw * 0.5):
                if not re.match(r"^[:\-\—\.\/]+$", w_info["text"]):
                    same_line.append(w_info)

        if same_line:
            first_norm = normalize_for_match(same_line[0]["text"])
            if first_norm not in NON_TITLE_WORDS:
                title_words.extend(same_line)

        # Check lines below anchor word
        below_lines = {}
        for w_info in words:
            if (ay + ah * 0.6) <= w_info["y"] <= (ay + ah * 10):
                if (ax - 80) <= w_info["x"] <= (ax + max(aw * 10, 1000)):
                    y_bucket = round(w_info["y"] / 15) * 15
                    below_lines.setdefault(y_bucket, []).append(w_info)

        for y_b in sorted(below_lines.keys()):
            line_w = sorted(below_lines[y_b], key=lambda item: item["x"])
            line_text = " ".join(item["text"] for item in line_w)
            norm_line = normalize_for_match(line_text)

            words_norm = norm_line.split()
            if words_norm and words_norm[0] in NON_TITLE_WORDS:
                break
            if re.search(r"\b(REV|DATE|DATA|SCALE|ECHELLE|ESCALA|DWG|SIZE|SHEET|FOLHA|N\s*DE\s*PLAN|N\s*DO\s*DESENHO)\b", norm_line):
                break

            title_words.extend(line_w)
            if len(title_words) >= 15:
                break

        if title_words:
            all_title_boxes = [anchor_word] + title_words
            min_x = min(w["x"] for w in all_title_boxes)
            min_y = min(w["y"] for w in all_title_boxes)
            max_x = max(w["x"] + w["w"] for w in all_title_boxes)
            max_y = max(w["y"] + w["h"] for w in all_title_boxes)

            title_box = (
                max(0, roi_x1 + min_x - 12),
                max(0, roi_y1 + min_y - 10),
                min(w - (roi_x1 + min_x - 12), (max_x - min_x) + 24),
                min(h - (roi_y1 + min_y - 10), (max_y - min_y) + 20)
            )

            title_text = " ".join(w["text"] for w in title_words).strip()
            title_text = re.sub(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D—–]', '-', title_text)
            title_text = title_text.strip("[]|: ")
            return title_text, title_box

    # Step 3: If no anchor word was found, scan and score cells to extract title content
    best_score = 50.0
    best_cell = None
    best_cell_text = ""
    for cell in proper_cells:
        cx, cy, cw, ch = cell
        cell_crop = image[cy:cy+ch, cx:cx+cw]
        txt = pytesseract.image_to_string(cell_crop, lang=chosen_lang, config="--psm 6").strip()
        sc = score_cell_for_title(txt, cw, ch)
        if sc > best_score:
            best_score = sc
            best_cell = cell
            best_cell_text = txt

    if best_cell:
        lines = [ln.strip() for ln in best_cell_text.splitlines() if ln.strip()]
        clean_lines = []
        for ln in lines:
            norm = normalize_for_match(ln)
            if not re.search(NON_TITLE_FIELDS_REGEX, norm, flags=re.IGNORECASE):
                clean_lines.append(ln)
        content = " ".join(clean_lines).strip()
        content = re.sub(r'[\u2010-\u2015\u2212\uFE58\uFE63\uFF0D—–]', '-', content)
        content = content.strip("[]|: ")
        if content:
            return content, best_cell

    return "", None


def save_outputs(
    image: np.ndarray,
    tb_coords: tuple[int, int, int, int],
    title_box: tuple[int, int, int, int] | None,
    title_content: str,
    output_dir: Path,
    file_stem: str
):
    """
    Saves the 3 required outputs:
    1. <name>_crop.png: Cropped image of the detected Title Box (or Title Block if no box).
    2. <name>_boxes.png: OpenCV visualizer showing the title anchor box (or left clean if no title).
    3. <name>_extracted.txt: ONLY the extracted title content (or empty 0 bytes if not found).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rx, ry, rw, rh = tb_coords

    # Output 1: Crop of Title Box (from where the title is extracted)
    crop_path = output_dir / f"{file_stem}_crop.png"
    if title_box is not None:
        tx, ty, tw, th = title_box
        crop_img = image[ty:ty+th, tx:tx+tw]
    else:
        crop_img = image[ry:ry+rh, rx:rx+rw]
    cv2.imwrite(str(crop_path), crop_img)

    # Output 2: OpenCV Boxes Visualizer
    boxes_path = output_dir / f"{file_stem}_boxes.png"
    vis = image.copy()

    # Draw Title Block perimeter in green
    cv2.rectangle(vis, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 3)
    cv2.rectangle(vis, (rx, max(0, ry - 30)), (rx + 200, ry), (0, 200, 0), -1)
    cv2.putText(vis, "TITLE BLOCK", (rx + 5, max(18, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # If anchor title was found, draw THAT box in prominent bold red
    if title_box is not None:
        tx, ty, tw, th = title_box
        cv2.rectangle(vis, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 4)
        tag_y = max(24, ty - 6)
        cv2.rectangle(vis, (tx, tag_y - 24), (tx + 200, tag_y + 2), (0, 0, 220), -1)
        cv2.putText(vis, "TITLE BOX", (tx + 5, tag_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Zoomed ROI output with 15% margin
    margin = int(max(rw, rh) * 0.15)
    vy1 = max(0, ry - margin)
    vy2 = min(image.shape[0], ry + rh + margin)
    vx1 = max(0, rx - margin)
    vx2 = min(image.shape[1], rx + rw + margin)
    zoomed = vis[vy1:vy2, vx1:vx2]
    cv2.imwrite(str(boxes_path), zoomed)

    # Output 3: Only the title content written in the txt file (empty if not found)
    txt_path = output_dir / f"{file_stem}_extracted.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        if title_content:
            f.write(title_content.strip() + "\n")


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

        # 2. Get Title Block and Table Cells (bottom-right)
        tb_coords, cells, crop_img = get_title_block_and_cells(img)

        # 3. Locate Title Anchor word, take that box, and extract the content
        title_content, title_box = extract_by_anchor(img, tb_coords, cells, lang=lang)

        if title_content:
            print(f"  -> Title Content Extracted: '{title_content}'")
            print(f"  -> Title Anchor Box: x={title_box[0]}, y={title_box[1]}, w={title_box[2]}, h={title_box[3]}")
        else:
            print(f"  -> [No Title Anchor Found] Leaving title box and text empty.")

        # 4. Save the 3 outputs
        save_outputs(
            image=img,
            tb_coords=tb_coords,
            title_box=title_box,
            title_content=title_content,
            output_dir=output_dir,
            file_stem=file_stem
        )

        print(f"[DONE] Saved: {file_stem}_crop.png, {file_stem}_boxes.png, {file_stem}_extracted.txt")
        return True

    except Exception as e:
        print(f"[ERROR] Failed {file_path.name}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Fast Local Scanned PDF/TIFF/PNG/JPEG Title Anchor Extractor")
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
