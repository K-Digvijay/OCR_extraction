"""
Generate test drawings with complex punctuation (- / \ & .) saved in TIFF, TIF, JPG, and JPEG.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

def make_drawing_with_punctuation(title_label: str, title_val: str, dwg_no: str, filename: str):
    width, height = 2100, 1485
    img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Outer border
    cv2.rectangle(img, (50, 50), (width - 50, height - 50), (0, 0, 0), 4)

    # Title block in bottom right
    tb_w, tb_h = 750, 240
    tb_x = width - 50 - tb_w
    tb_y = height - 50 - tb_h

    # Outer title block border
    cv2.rectangle(img, (tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h), (0, 0, 0), 4)

    # Dividers
    cv2.line(img, (tb_x, tb_y + 50), (tb_x + tb_w, tb_y + 50), (0, 0, 0), 2)
    cv2.line(img, (tb_x, tb_y + 140), (tb_x + tb_w, tb_y + 140), (0, 0, 0), 2)
    cv2.line(img, (tb_x, tb_y + 190), (tb_x + tb_w, tb_y + 190), (0, 0, 0), 2)

    # Text with punctuation
    cv2.putText(img, "BUREAU D'ETUDES TECHNIQUES - PARIS", (tb_x + 20, tb_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img, f"{title_label} :", (tb_x + 20, tb_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(img, title_val, (tb_x + 20, tb_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    cv2.putText(img, f"DWG NO : {dwg_no}", (tb_x + 20, tb_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, "REV: 01 / A    DATE: 22/09/2026    ECHELLE: 1/100", (tb_x + 20, tb_y + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    out_path = Path("test_samples") / filename
    ext = out_path.suffix.lower()

    if ext in (".tiff", ".tif"):
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_img.save(str(out_path))
    else:
        cv2.imwrite(str(out_path), img)

    print(f"Generated: {out_path}")

if __name__ == "__main__":
    # 1. TIFF with hyphens, slashes, backslashes, dots
    make_drawing_with_punctuation(
        title_label="DESIGNATION",
        title_val="CIRCUIT-EAU / ZONE-A \\ SEC.01 & 02",
        dwg_no="P-101/A\\01-REV.B",
        filename="punct_test.tiff"
    )

    # 2. TIF format
    make_drawing_with_punctuation(
        title_label="TITULO",
        title_val="SISTEMA-HIDRAULICO / SETOR-B \\ AREA-03",
        dwg_no="BR-2026/04-A",
        filename="punct_test.tif"
    )

    # 3. JPEG format
    make_drawing_with_punctuation(
        title_label="TITRE",
        title_val="PLAN-DE-MASSE / LOT-N.04 \\ BAT-C",
        dwg_no="FR-2026-LOT.04",
        filename="punct_test.jpeg"
    )
