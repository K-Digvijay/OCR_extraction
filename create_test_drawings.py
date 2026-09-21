"""
Script to generate synthetic test engineering drawings in French and Portuguese.
Creates both a PDF drawing and an image drawing to test all pipeline capabilities.
"""
from pathlib import Path
import cv2
import numpy as np
import pymupdf


def draw_engineering_sheet(title_label: str, title_val: str, dwg_label: str, dwg_no: str, rev_label: str, rev_val: str, date_label: str, date_val: str, scale_label: str, scale_val: str, company: str):
    # Standard A3 sheet proportion in pixels (e.g. 2100 x 1485)
    width, height = 2100, 1485
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 1. Outer drawing border
    cv2.rectangle(img, (50, 50), (width - 50, height - 50), (0, 0, 0), 4)

    # 2. Some drawing schematic lines and elements in the main body
    cv2.line(img, (200, 300), (900, 300), (0, 0, 0), 3)
    cv2.line(img, (900, 300), (900, 800), (0, 0, 0), 3)
    cv2.circle(img, (550, 300), 40, (0, 0, 0), 2)
    cv2.putText(img, "V-101", (530, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.rectangle(img, (800, 800), (1000, 1100), (0, 0, 0), 3)
    cv2.putText(img, "PUMP P-102A", (810, 950), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # 3. Standard Title Block in the bottom-right corner (ISO 7200 / ASME style)
    tb_w = 600
    tb_h = 240
    tb_x = width - 50 - tb_w
    tb_y = height - 50 - tb_h

    # Title block outer border
    cv2.rectangle(img, (tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h), (0, 0, 0), 4)

    # Internal dividing lines
    # Horizontal divider 1 (Company row: height 50)
    cv2.line(img, (tb_x, tb_y + 50), (tb_x + tb_w, tb_y + 50), (0, 0, 0), 2)
    # Horizontal divider 2 (Title row: height 90)
    cv2.line(img, (tb_x, tb_y + 140), (tb_x + tb_w, tb_y + 140), (0, 0, 0), 2)
    # Horizontal divider 3 (Dwg no row: height 50)
    cv2.line(img, (tb_x, tb_y + 190), (tb_x + tb_w, tb_y + 190), (0, 0, 0), 2)

    # Vertical dividers in the bottom row (Rev, Date, Scale)
    cv2.line(img, (tb_x + 200, tb_y + 190), (tb_x + 200, tb_y + tb_h), (0, 0, 0), 2)
    cv2.line(img, (tb_x + 400, tb_y + 190), (tb_x + 400, tb_y + tb_h), (0, 0, 0), 2)

    # Text inside Title Block
    # Row 1: Company
    cv2.putText(img, company, (tb_x + 20, tb_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Row 2: Title Label and Value
    cv2.putText(img, f"{title_label} :", (tb_x + 20, tb_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(img, title_val, (tb_x + 20, tb_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)

    # Row 3: Drawing Number
    cv2.putText(img, f"{dwg_label} : {dwg_no}", (tb_x + 20, tb_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Row 4: Rev, Date, Scale
    cv2.putText(img, f"{rev_label} : {rev_val}", (tb_x + 20, tb_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(img, f"{date_label} : {date_val}", (tb_x + 220, tb_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(img, f"{scale_label} : {scale_val}", (tb_x + 420, tb_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return img


def main():
    test_dir = Path("test_samples")
    test_dir.mkdir(exist_ok=True)

    # 1. Generate French Drawing (Save as PDF)
    fr_img = draw_engineering_sheet(
        title_label="DESIGNATION",
        title_val="PLAN DE TUYAUTERIE DU CIRCUIT REFROIDISSEMENT",
        dwg_label="N DE PLAN",
        dwg_no="FR-2026-0042",
        rev_label="INDICE",
        rev_val="B",
        date_label="DATE",
        date_val="15/04/2026",
        scale_label="ECHELLE",
        scale_val="1:50",
        company="INGENIERIE INDUSTRIELLE FRANCE SAS"
    )

    # Convert to PDF via PyMuPDF
    fr_png_path = test_dir / "french_drawing.png"
    cv2.imwrite(str(fr_png_path), fr_img)

    fr_doc = pymupdf.open()
    img_doc = pymupdf.open(str(fr_png_path))
    pdf_bytes = img_doc.convert_to_pdf()
    fr_doc.insert_pdf(pymupdf.open("pdf", pdf_bytes))
    fr_pdf_path = test_dir / "french_drawing.pdf"
    fr_doc.save(str(fr_pdf_path))
    fr_doc.close()
    img_doc.close()
    print(f"Generated French PDF: {fr_pdf_path}")

    # 2. Generate Portuguese Drawing (Save as PNG image)
    pt_img = draw_engineering_sheet(
        title_label="TITULO",
        title_val="DIAGRAMA DE TUBULACAO DO SISTEMA DE AGUA",
        dwg_label="N DO DESENHO",
        dwg_no="PT-2026-0088",
        rev_label="REVISAO",
        rev_val="03",
        date_label="DATA",
        date_val="20/05/2026",
        scale_label="ESCALA",
        scale_val="1:100",
        company="COMPANHIA DE ENGENHARIA DO BRASIL"
    )

    pt_png_path = test_dir / "portuguese_drawing.png"
    cv2.imwrite(str(pt_png_path), pt_img)
    print(f"Generated Portuguese Image: {pt_png_path}")


if __name__ == "__main__":
    main()
