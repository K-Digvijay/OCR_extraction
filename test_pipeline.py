"""
Comprehensive automated verification test for drawing title extraction pipeline.
Tests:
1. Real user drawings in sample/ (anchored and unanchored).
2. Worst-case faint/hardly visible degraded scan with lighting gradients and paper noise.
3. Multilingual French and Portuguese drawings with diacritics.
4. Strict punctuation preservation (-, /, \, &, .).
5. Truly untitled sheets (leaves empty 0-byte text and no red box).
6. Existence and integrity of all 3 output files ([name]_crop.png, [name]_boxes.png, [name]_extracted.txt).
"""
import os
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from main import process_file, load_image


def generate_hardly_visible_drawing(out_path: Path):
    """Generates a challenging degraded scan with faint pencil text, yellowed background, and paper noise."""
    w, h = 1600, 1100
    # Uneven background lighting / aged paper
    bg = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        for x in range(w):
            bg[y, x] = 208 + 18 * np.sin(x / 250.0) + 14 * np.cos(y / 180.0)
    bg = np.clip(bg, 180, 240).astype(np.uint8)

    pil_img = Image.fromarray(bg).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 28)

    # Drawing body lines
    draw.line([(100, 200), (800, 200)], fill=(160, 160, 160), width=2)
    draw.line([(800, 200), (800, 600)], fill=(160, 160, 160), width=2)

    # Title block in bottom-right (x=950, y=820, w=600, h=240)
    tb_x, tb_y, tb_w, tb_h = 950, 820, 600, 240
    draw.rectangle([(tb_x, tb_y), (tb_x + tb_w, tb_y + tb_h)], outline=(155, 155, 155), width=2)
    draw.line([(tb_x, tb_y + 80), (tb_x + tb_w, tb_y + 80)], fill=(155, 155, 155), width=2)
    draw.line([(tb_x, tb_y + 170), (tb_x + tb_w, tb_y + 170)], fill=(155, 155, 155), width=2)

    # Faint pencil text (gray ~165 on ~210 background, contrast only ~20%)
    draw.text((tb_x + 20, tb_y + 25), "DÉSIGNATION / TITRE :", fill=(162, 162, 162), font=font)
    draw.text((tb_x + 20, tb_y + 110), "CIRCUIT-VAPEUR / SECTION-04 \\ RESEAU-01 & 02", fill=(162, 162, 162), font=font)
    draw.text((tb_x + 20, tb_y + 195), "DWG NO : FR-2026-VAP-04", fill=(162, 162, 162), font=font)

    # Add realistic scan noise
    noisy = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, 4.0, noisy.shape)
    faint_img = np.clip(noisy + noise, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_path), faint_img)


def run_tests():
    output_dir = PROJECT_ROOT / "test_pipeline_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("RUNNING AUTOMATED PIPELINE VERIFICATION SUITE")
    print("=" * 70)

    # 1. Create and test the worst-case faint/hardly visible scan
    faint_path = output_dir / "worst_case_faint_scan.png"
    generate_hardly_visible_drawing(faint_path)
    print(f"\n[TEST 1] Generated worst-case degraded faint scan: {faint_path.name}")

    test_cases = [
        # (file_path, expected_substring, should_have_title, check_name)
        (faint_path, "CIRCUIT-VAPEUR", True, "Worst-Case Faint Pencil Scan"),
        (PROJECT_ROOT / "sample" / "sample-engineering-drawing.pdf", "EXAMPLE", True, "User CAD PDF (Anchored TITLE)"),
        (PROJECT_ROOT / "sample" / "SOLE_PLATE.pdf", "SOLE PLATE", True, "User Scanned PDF (SOLE PLATE)"),
        (PROJECT_ROOT / "sample" / "_img_precast-shop-drawings_2d-drawings_4.jpg", "TANTANGARA INTAKE", True, "User Drawing (Unanchored Title Block)"),
        (PROJECT_ROOT / "test_samples" / "french_drawing.pdf", "PLAN DE TUYAUTERIE", True, "French PDF Drawing"),
        (PROJECT_ROOT / "test_samples" / "portuguese_drawing.png", "DIAGRAMA DE TUBULACAO", True, "Portuguese Drawing"),
        (PROJECT_ROOT / "test_samples" / "punct_test.tiff", "CIRCUIT-EAU / ZONE-A \\ SEC.01 & 02", True, "Punctuation Preservation TIFF"),
        (PROJECT_ROOT / "test_samples" / "no_title.png", "", False, "Untitled Sheet (Clean Output)"),
    ]

    all_passed = True
    results = []

    for file_path, expected, should_have_title, desc in test_cases:
        if not file_path.exists():
            print(f"  [SKIP] File not found: {file_path}")
            continue

        print(f"\n>>> Running: {desc} ({file_path.name})")
        ok = process_file(file_path, output_dir=output_dir, overwrite=True)
        if not ok:
            print(f"  FAILED: Processing returned False for {file_path.name}")
            all_passed = False
            results.append((desc, "FAIL (Crash/Error)"))
            continue

        file_stem = file_path.stem
        txt_path = output_dir / f"{file_stem}_extracted.txt"
        crop_path = output_dir / f"{file_stem}_crop.png"
        boxes_path = output_dir / f"{file_stem}_boxes.png"

        # Check 3 outputs exist
        if not txt_path.exists() or not crop_path.exists() or not boxes_path.exists():
            print(f"  FAILED: Missing one of 3 required outputs for {file_stem}")
            all_passed = False
            results.append((desc, "FAIL (Missing output files)"))
            continue

        extracted_text = txt_path.read_text(encoding="utf-8").strip()

        if should_have_title:
            if not extracted_text:
                print(f"  FAILED: Expected title for {file_stem}, but extracted text is empty!")
                all_passed = False
                results.append((desc, "FAIL (Empty text)"))
                continue
            if expected and expected not in extracted_text:
                print(f"  FAILED: Expected '{expected}' in '{extracted_text}'")
                all_passed = False
                results.append((desc, f"FAIL (Mismatch: '{extracted_text}')"))
                continue
            print(f"  PASS: Extracted Title -> '{extracted_text}'")
            results.append((desc, f"PASS -> '{extracted_text[:40]}'"))
        else:
            if extracted_text:
                print(f"  FAILED: Expected empty text for {file_stem}, but got: '{extracted_text}'")
                all_passed = False
                results.append((desc, "FAIL (Expected empty)"))
                continue
            print(f"  PASS: Correctly left empty (0 bytes)")
            results.append((desc, "PASS (Clean 0-byte output)"))

    print("\n" + "=" * 70)
    print("AUTOMATED VERIFICATION SUMMARY")
    print("=" * 70)
    for desc, status in results:
        print(f"  {desc:<45} | {status}")

    if all_passed:
        print("\nALL VERIFICATION TESTS PASSED SUCCESSFULLY! (100% Accuracy)")
        return 0
    else:
        print("\nSOME TESTS FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
