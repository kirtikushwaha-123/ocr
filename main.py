#!/usr/bin/env python3
"""
main.py

PicWise - Ingredient and Nutrition Region Detection from Packaged Product
Images (Version 1: OCR + OpenCV + heuristics, NO trained YOLO model).

Usage:
    python main.py --image input/product.jpg
    python main.py --image input/product.jpg --output output/
    python main.py --image input/product.jpg --domain food
    python main.py --image input/product.jpg --domain personal_care
    python main.py --image input/product.jpg --test-mode

See README.md for full documentation of the pipeline and algorithm.
"""

import argparse
import json
import os
import sys
import time

import cv2

import config
from preprocessing.image_utils import (
    load_image,
    normalize_image,
    check_image_quality,
    safe_crop,
    pad_bbox,
    save_image,
)
from preprocessing.enhancement import preprocess_roi
from preprocessing.deskew import deskew
from preprocessing.perspective import correct_perspective

from detection.packet_region import detect_packet_region
from detection.ocr_detector import run_full_image_ocr
from detection.ingredient_region import detect_ingredient_region
from detection.nutrition_region import detect_nutrition_region
from detection.region_reconciliation import reconcile_regions

from ocr.ensemble import run_variant_ocr

from parsing.ingredient_parser import parse_ingredients
from parsing.nutrition_parser import parse_nutrition

from matching.knowledge_base import KnowledgeBase

from visualization.draw_regions import draw_regions


def classify_domain(all_text_lower):
    """
    STEP 20: simple heuristic domain classification (no neural network).
    Counts food vs personal-care signal keyword hits in the full-image
    OCR text and picks whichever has more hits. Ties (including 0-0)
    default to "food" since nutrition/ingredients panels are the more
    common case and food-domain output degrades gracefully (nutrition
    fields just end up empty) if the guess is wrong.
    """
    food_hits = sum(1 for kw in config.FOOD_SIGNALS if kw in all_text_lower)
    personal_care_hits = sum(1 for kw in config.PERSONAL_CARE_SIGNALS if kw in all_text_lower)

    if personal_care_hits > food_hits:
        return "personal_care", {"food_hits": food_hits, "personal_care_hits": personal_care_hits}
    return "food", {"food_hits": food_hits, "personal_care_hits": personal_care_hits}


def process_region(image, region_result, mode, save_prefix, output_dir, test_mode):
    """
    Shared post-detection pipeline for a detected region (ingredients or
    nutrition): pad+crop -> deskew -> optional perspective correction ->
    multi-variant preprocessing -> OCR ensemble -> best text.

    Returns (raw_crop, processed_best_image, ocr_ensemble_result) or
    (None, None, None) if region_result has no bbox.
    """
    bbox = region_result.get("bbox")
    if bbox is None:
        return None, None, None

    # The bbox has already been refined with a small margin in tight_roi.py
    crop = safe_crop(image, bbox)
    if crop is None:
        return None, None, None

    deskewed, angle = deskew(crop)
    corrected, applied_perspective = correct_perspective(deskewed)

    variants = preprocess_roi(corrected)

    if test_mode:
        stage_dir = os.path.join(output_dir, "stages", save_prefix)
        os.makedirs(stage_dir, exist_ok=True)
        save_image(crop, os.path.join(stage_dir, "01_raw_crop.jpg"))
        save_image(deskewed, os.path.join(stage_dir, "02_deskewed.jpg"))
        save_image(corrected, os.path.join(stage_dir, "03_perspective_corrected.jpg"))
        for name, img in variants.items():
            save_image(img, os.path.join(stage_dir, f"variant_{name}.jpg"))

    ensemble_result = run_variant_ocr(variants, mode=mode)

    best_variant_name = ensemble_result.get("best_variant")
    best_processed_image = variants.get(best_variant_name) if best_variant_name else corrected

    return crop, best_processed_image, ensemble_result


def run_pipeline(image_path, output_dir, domain_arg="auto", test_mode=False, kb=None):
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    log = []

    def _log(msg):
        print(msg)
        log.append(msg)

    _log(f"[1/9] Loading image: {image_path}")
    original = load_image(image_path)
    save_image(original, os.path.join(output_dir, "original.jpg"))

    quality = check_image_quality(original)
    _log(f"      Quality check: {quality}")

    _log("[2/9] Normalizing image size")
    normalized = normalize_image(original)

    _log("[3/9] Attempting packet/foreground isolation (OpenCV only, no YOLO)")
    packet_crop, packet_bbox, packet_confidence = detect_packet_region(normalized)
    _log(
        f"      Packet detection confidence={packet_confidence:.2f}, "
        f"bbox={packet_bbox}"
    )
    save_image(packet_crop, os.path.join(output_dir, "packet_crop.jpg"))

    # We run OCR/anchor detection on the packet crop when confidence is
    # reasonably high, otherwise on the full normalized image, per the
    # "don't force a crop that could damage the image" rule.
    working_image = packet_crop if packet_confidence >= config.PACKET_MIN_CONFIDENCE_TO_CROP else normalized

    _log("[4/9] Running full-image PaddleOCR")
    all_items = run_full_image_ocr(working_image)





    if test_mode and all_items:
        print("\n========== OCR POLYGON DEBUG ==========")

        for i, item in enumerate(all_items[:10]):

            print(f"\nOCR ITEM {i + 1}")

            print("Text:")
            print(item.get("text"))

            print("Confidence:")
            print(item.get("confidence"))

            print("Polygon:")
            print(item.get("polygon"))

            print("Rect:")
            print(item.get("rect"))

            print("Center:")
            print(item.get("center"))

            print("Width:")
            print(item.get("width"))

            print("Height:")
            print(item.get("height"))

        print("\n========================================\n")













    _log(f"      OCR found {len(all_items)} text boxes")

    all_text_lower = " ".join(it["norm_text"] for it in all_items)

    if domain_arg == "auto":
        domain, domain_debug = classify_domain(all_text_lower)
        _log(f"      Domain classification (heuristic): {domain} ({domain_debug})")
    else:
        domain = domain_arg
        _log(f"      Domain forced by CLI: {domain}")

    if kb is None:
        kb = KnowledgeBase()

    ingredient_vocab = kb.get_ingredient_names(domain=domain)

    _log("[5/9] Detecting Ingredients region")
    ingredient_result = detect_ingredient_region(
        all_items, working_image.shape, ingredient_vocab=ingredient_vocab, debug=test_mode
    )
    # Refine ROI to generate tight bbox with margin
    from detection.tight_roi import refine_ingredient_roi, refine_nutrition_roi
    ingredient_result["bbox"] = refine_ingredient_roi(ingredient_result.get("matched_items", []), working_image.shape)
    _log(
        f"      Ingredients bbox={ingredient_result['bbox']} "
        f"confidence={ingredient_result['confidence']} "
        f"anchor={ingredient_result['anchor']!r} method={ingredient_result['method']}"
    )

    nutrition_result = {
        "bbox": None, "confidence": 0.0, "anchor": None,
        "matched_items": [], "matched_terms": [], "method": "skipped_personal_care",
    }
    if domain == "food":
        _log("[6/9] Detecting Nutrition region")
        nutrition_result = detect_nutrition_region(
            all_items, working_image.shape, ingredient_vocab=ingredient_vocab, debug=test_mode
        )
        # Refine ROI to generate tight bbox with margin
        nutrition_result["bbox"] = refine_nutrition_roi(nutrition_result.get("matched_items", []), working_image.shape)
        _log(
            f"      Nutrition bbox={nutrition_result['bbox']} "
            f"confidence={nutrition_result['confidence']} "
            f"anchor={nutrition_result['anchor']!r} method={nutrition_result['method']}"
        )

        ingredient_result, nutrition_result = reconcile_regions(
            ingredient_result, nutrition_result, working_image.shape, ingredient_vocab=ingredient_vocab
        )
    else:
        _log("[6/9] Skipping Nutrition detection (domain=personal_care)")

    _log("[7/9] Cropping + preprocessing + re-OCR'ing Ingredients ROI")
    ing_raw_crop, ing_best_img, ing_ocr = process_region(
        working_image, ingredient_result, "ingredient", "ingredients", output_dir, test_mode
    )

    nut_raw_crop, nut_best_img, nut_ocr = (None, None, None)
    if domain == "food":
        _log("      Cropping + preprocessing + re-OCR'ing Nutrition ROI")
        nut_raw_crop, nut_best_img, nut_ocr = process_region(
            working_image, nutrition_result, "nutrition", "nutrition", output_dir, test_mode
        )

    # --- save region crop + processed outputs ---
    if ing_raw_crop is not None:
        save_image(ing_raw_crop, os.path.join(output_dir, "ingredients_region.jpg"))
    if ing_best_img is not None:
        save_image(ing_best_img, os.path.join(output_dir, "ingredients_processed.jpg"))

    if nut_raw_crop is not None:
        save_image(nut_raw_crop, os.path.join(output_dir, "nutrition_region.jpg"))
    if nut_best_img is not None:
        save_image(nut_best_img, os.path.join(output_dir, "nutrition_processed.jpg"))

    _log("[8/9] Parsing text + matching against knowledge base")
    ingredients_output = []
    best_ingredient_variant = None
    if ing_ocr:
        best_ingredient_variant = ing_ocr.get("best_variant")
        from nlp.ingredient_corrector import IngredientCorrector
        corrector = IngredientCorrector(kb=kb)
        ingredients_output = corrector.correct_and_match(ing_ocr.get("best_text", ""), domain=domain)

    nutrition_output = None
    best_nutrition_variant = None
    if domain == "food" and nut_ocr:
        best_nutrition_variant = nut_ocr.get("best_variant")
        nutrition_output = parse_nutrition(nut_ocr.get("best_text", ""))

    _log("[9/9] Drawing visualization + writing outputs")
    vis = draw_regions(
        working_image,
        ingredient_result=ingredient_result,
        nutrition_result=nutrition_result if domain == "food" else None,
        all_ocr_items=all_items,
    )
    save_image(vis, os.path.join(output_dir, "visualization.jpg"))

    # Debug info (STEP 23 in the region-detection spec): anchor used,
    # accepted/rejected lines, stop reason, column band - only written in
    # --test-mode so normal runs stay uncluttered.
    if test_mode:
        def _debug_safe(result):
            if not result or "debug" not in result:
                return None
            d = dict(result["debug"])
            # matched_items/rejected_lines can hold non-JSON-serializable
            # numpy floats inside rects - normalize to plain floats.
            return json.loads(json.dumps(d, default=lambda o: float(o) if hasattr(o, "item") else str(o)))

        debug_payload = {
            "ingredients": _debug_safe(ingredient_result),
            "nutrition": _debug_safe(nutrition_result) if domain == "food" else None,
        }
        with open(os.path.join(output_dir, "debug_region_detection.json"), "w", encoding="utf-8") as f:
            json.dump(debug_payload, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t_start

    result = {
        "image": os.path.basename(image_path),
        "domain": domain,
        "packet_detection": {
            "bbox": packet_bbox,
            "confidence": round(float(packet_confidence), 3),
        },
        "ingredients_region": {
            "bbox": ingredient_result["bbox"],
            "confidence": ingredient_result["confidence"],
            "anchor": ingredient_result["anchor"],
            "method": ingredient_result["method"],
        },
        "nutrition_region": (
            {
                "bbox": nutrition_result["bbox"],
                "confidence": nutrition_result["confidence"],
                "anchor": nutrition_result["anchor"],
                "method": nutrition_result["method"],
            }
            if domain == "food"
            else None
        ),
        "ingredients": ingredients_output,
        "nutrition": nutrition_output,
        "processing": {
            "best_ingredient_variant": best_ingredient_variant,
            "best_nutrition_variant": best_nutrition_variant,
            "elapsed_seconds": round(elapsed, 2),
        },
        "image_quality": quality,
    }

    result_json_path = os.path.join(output_dir, "result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result_txt_path = os.path.join(output_dir, "result.txt")
    with open(result_txt_path, "w", encoding="utf-8") as f:
        f.write(f"PicWise Result for: {result['image']}\n")
        f.write(f"Domain: {result['domain']}\n\n")

        f.write("--- Ingredients Region ---\n")
        f.write(f"bbox: {result['ingredients_region']['bbox']}\n")
        f.write(f"confidence: {result['ingredients_region']['confidence']}\n")
        f.write(f"anchor: {result['ingredients_region']['anchor']}\n")
        f.write(f"method: {result['ingredients_region']['method']}\n\n")

        f.write("Ingredients:\n")
        for item in ingredients_output:
            match_str = item["matched_name"] or "(no confident KB match)"
            f.write(f"  - {item['ocr_text']}  ->  {match_str}\n")
        f.write("\n")

        if domain == "food":
            f.write("--- Nutrition Region ---\n")
            f.write(f"bbox: {result['nutrition_region']['bbox']}\n")
            f.write(f"confidence: {result['nutrition_region']['confidence']}\n")
            f.write(f"anchor: {result['nutrition_region']['anchor']}\n")
            f.write(f"method: {result['nutrition_region']['method']}\n\n")

            f.write("Nutrition facts:\n")
            if nutrition_output:
                for k, v in nutrition_output.items():
                    f.write(f"  - {k}: {v['value']} {v['unit']}\n")
            else:
                f.write("  (none extracted)\n")
        else:
            f.write("--- Nutrition Region ---\n  skipped (personal_care domain)\n")

        f.write("\n--- Processing log ---\n")
        for line in log:
            f.write(line + "\n")

    _log(f"Done in {elapsed:.2f}s. Results written to: {output_dir}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="PicWise - Ingredient & Nutrition region detector (OCR + OpenCV, no YOLO in v1)"
    )
    parser.add_argument("--image", required=True, help="Path to input image (jpg/jpeg/png)")
    parser.add_argument(
        "--output", default=config.OUTPUT_DIR, help="Output directory (default: output/)"
    )
    parser.add_argument(
        "--domain",
        default="auto",
        choices=["auto", "food", "personal_care"],
        help="Force domain classification instead of using the heuristic classifier",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Save every intermediate processing stage (crops, deskew, perspective, all preprocessing variants) into output/stages/",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.image):
        print(f"ERROR: image file not found: {args.image}", file=sys.stderr)
        sys.exit(1)

    try:
        run_pipeline(
            image_path=args.image,
            output_dir=args.output,
            domain_arg=args.domain,
            test_mode=args.test_mode,
        )
    except Exception as e:
        print(f"ERROR: pipeline failed: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
