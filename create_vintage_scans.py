"""
Generate authentic 20-year-old aged technical drawing scans (TIFF and PDF)
with realistic ink bleeding/spread, paper grain, background discoloration,
and PROPER Unicode rendering for French and Portuguese accents using PIL.ImageDraw.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf


def get_font(size: int):
    # Try standard Windows fonts with full Latin Unicode support
    for font_name in ["arial.ttf", "calibri.ttf", "tahoma.ttf", "cour.ttf"]:
        font_path = Path("C:/Windows/Fonts") / font_name
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size)
            except Exception:
                continue
    return ImageFont.load_default()


def simulate_vintage_scan(img: np.ndarray) -> np.ndarray:
    """
    Simulates a 20-year-old scanned engineering drawing:
    - Ink bleeding / spread
    - Paper aging, yellowing/graying background
    - Gaussian paper texture noise
    - Slight scan blur
    """
    h, w = img.shape[:2]

    # 1. Simulate ink bleed / spread:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink_mask = gray < 200

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    spread_ink = cv2.dilate((ink_mask.astype(np.uint8) * 255), kernel, iterations=1)
    spread_ink = cv2.GaussianBlur(spread_ink, (3, 3), 0.7)

    # 2. Aged paper background (slightly uneven lighting, grain, off-white / light sepia)
    y_grad = np.linspace(235, 248, h).reshape(h, 1)
    x_grad = np.linspace(238, 245, w).reshape(1, w)
    bg = ((y_grad + x_grad) / 2).astype(np.float32)

    noise = np.random.normal(0, 4.0, (h, w)).astype(np.float32)
    paper = np.clip(bg + noise, 180, 255).astype(np.uint8)
    paper_bgr = cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)

    # 3. Composite spread ink onto aged paper
    ink_factor = (spread_ink.astype(np.float32) / 255.0)[:, :, np.newaxis]
    ink_color = np.array([38, 36, 32], dtype=np.float32)

    aged = (paper_bgr.astype(np.float32) * (1.0 - ink_factor * 0.88) + ink_color * (ink_factor * 0.88)).astype(np.uint8)
    aged = cv2.GaussianBlur(aged, (3, 3), 0.4)
    return aged


def make_vintage_drawing(title_label: str, title_val: str, dwg_no: str, company: str):
    width, height = 2400, 1700
    pil_img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(pil_img)

    # Load fonts
    font_large = get_font(38)
    font_medium = get_font(32)
    font_small = get_font(26)

    # 1. Outer drawing border
    draw.rectangle([(60, 60), (width - 60, height - 60)], outline=(0, 0, 0), width=4)

    # 2. Schematic elements in drawing body
    draw.line([(250, 400), (1200, 400)], fill=(0, 0, 0), width=3)
    draw.line([(1200, 400), (1200, 950)], fill=(0, 0, 0), width=3)
    draw.ellipse([(650, 350), (750, 450)], outline=(0, 0, 0), width=3)
    draw.text((630, 310), "V-201 / BYPASS", fill=(0, 0, 0), font=font_small)

    # 3. Complex Title Block in bottom-right corner
    tb_w, tb_h = 880, 300
    tb_x = width - 60 - tb_w
    tb_y = height - 60 - tb_h

    # Title block perimeter
    draw.rectangle([(tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h)], outline=(0, 0, 0), width=4)

    # Dividers
    draw.line([(tb_x, tb_y + 60), (tb_x + tb_w, tb_y + 60)], fill=(0, 0, 0), width=2)
    draw.line([(tb_x, tb_y + 175), (tb_x + tb_w, tb_y + 175)], fill=(0, 0, 0), width=2)
    draw.line([(tb_x, tb_y + 235), (tb_x + tb_w, tb_y + 235)], fill=(0, 0, 0), width=2)

    # Bottom vertical dividers
    draw.line([(tb_x + 280, tb_y + 235), (tb_x + 280, tb_y + tb_h)], fill=(0, 0, 0), width=2)
    draw.line([(tb_x + 580, tb_y + 235), (tb_x + 580, tb_y + tb_h)], fill=(0, 0, 0), width=2)

    # Proper Unicode Text inside Title Block
    draw.text((tb_x + 25, tb_y + 12), company, fill=(0, 0, 0), font=font_medium)
    draw.text((tb_x + 25, tb_y + 70), f"{title_label} :", fill=(0, 0, 0), font=font_small)
    draw.text((tb_x + 25, tb_y + 115), title_val, fill=(0, 0, 0), font=font_large)
    draw.text((tb_x + 25, tb_y + 190), f"REF / DWG NO : {dwg_no}", fill=(0, 0, 0), font=font_small)
    draw.text((tb_x + 25, tb_y + 250), "REV : 02 / B", fill=(0, 0, 0), font=font_small)
    draw.text((tb_x + 305, tb_y + 250), "DATE : 14/06/2004", fill=(0, 0, 0), font=font_small)
    draw.text((tb_x + 605, tb_y + 250), "ÉCHELLE : 1/50", fill=(0, 0, 0), font=font_small)

    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return simulate_vintage_scan(img)


def main():
    test_dir = Path("test_samples")
    test_dir.mkdir(exist_ok=True)

    # 1. Vintage French PDF (2004 scan with ink bleed and accents)
    vintage_fr = make_vintage_drawing(
        title_label="DÉSIGNATION / TITRE",
        title_val="SCHEMA-ELECTRIQUE / POSTE-HTA \\ DEPART-01 & 02",
        dwg_no="FR-2004/ELEC-01\\REV.B",
        company="ELECTRICITE ET ENERGIE DE FRANCE - ANNEE 2004"
    )
    fr_tmp_png = test_dir / "vintage_french_drawing.png"
    cv2.imwrite(str(fr_tmp_png), vintage_fr)
    fr_doc = pymupdf.open()
    img_doc = pymupdf.open(str(fr_tmp_png))
    fr_doc.insert_pdf(pymupdf.open("pdf", img_doc.convert_to_pdf()))
    fr_pdf_path = test_dir / "vintage_french_drawing.pdf"
    fr_doc.save(str(fr_pdf_path))
    fr_doc.close()
    img_doc.close()
    fr_tmp_png.unlink()
    print(f"Generated: {fr_pdf_path}")

    # 2. Vintage Portuguese TIFF (2004 scan with ink bleed and accents)
    vintage_pt = make_vintage_drawing(
        title_label="DESIGNACAO / TITULO",
        title_val="PLANO-GERAL DE TUBULACAO \\ AREA-SUL & SETOR-04",
        dwg_no="BR-2004/TUB-04\\REV.A",
        company="COMPANHIA HIDROELETRICA NACIONAL - 2004"
    )
    pt_tiff_path = test_dir / "vintage_portuguese_drawing.tiff"
    pil_img = Image.fromarray(cv2.cvtColor(vintage_pt, cv2.COLOR_BGR2RGB))
    pil_img.save(str(pt_tiff_path), compression="tiff_deflate")
    print(f"Generated: {pt_tiff_path}")


if __name__ == "__main__":
    main()
