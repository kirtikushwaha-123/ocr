"""
detection/region_reconciliation.py

Final validation step run AFTER Ingredients and Nutrition have been
detected completely independently. Resolves line ownership conflicts if
there is overlap, ensuring each line belongs exclusively to its correct
semantic category (Ingredients, Nutrition, or Other) based on semantic scores.
"""

import config
from detection.geometry import bbox_iou, union_rect, validate_region
from detection.tight_roi import refine_ingredient_roi, refine_nutrition_roi

def reconcile_regions(ingredient_result, nutrition_result, image_shape, ingredient_vocab=None):
    """
    Validates overlap between independently-detected Ingredients and Nutrition regions.
    If overlap is excessive, resolves line ownership using semantic scores.
    """
    ing_bbox = ingredient_result.get("bbox")
    nut_bbox = nutrition_result.get("bbox")

    if not ing_bbox or not nut_bbox:
        return ingredient_result, nutrition_result

    # Compute Intersection over Union
    iou = bbox_iou(ing_bbox, nut_bbox)
    
    # If overlap exceeds threshold, or if there are shared lines, we reconcile
    ing_lines = ingredient_result.get("matched_items", [])
    nut_lines = nutrition_result.get("matched_items", [])
    
    # Find duplicate/contested lines by checking coordinate proximity or text match
    shared_line_indices = []
    for i, il in enumerate(ing_lines):
        for j, nl in enumerate(nut_lines):
            # Two lines are considered the same if their text matches or they have identical boxes
            if il["text"] == nl["text"] or il["rect"] == nl["rect"]:
                shared_line_indices.append((i, j))

    if iou <= config.REGION_OVERLAP_IOU_RECONCILE_THRESHOLD and not shared_line_indices:
        return ingredient_result, nutrition_result

    # Semantic reassignment of contested lines
    resolved_ing_lines = list(ing_lines)
    resolved_nut_lines = list(nut_lines)
    
    to_remove_from_ing = set()
    to_remove_from_nut = set()

    for idx_ing, idx_nut in shared_line_indices:
        ln_ing = ing_lines[idx_ing]
        ln_nut = nut_lines[idx_nut]
        
        ing_score = ln_ing.get("ingredient_score", 0.0)
        nut_score = ln_nut.get("nutrition_score", 0.0)

        # Assign exclusively to the category with the higher semantic score
        if ing_score > nut_score:
            to_remove_from_nut.add(idx_nut)
        elif nut_score > ing_score:
            to_remove_from_ing.add(idx_ing)
        else:
            # Tie breaker: check text length or default to original assignment split
            if len(ln_ing["text"]) > 20:
                to_remove_from_nut.add(idx_nut)
            else:
                to_remove_from_ing.add(idx_ing)

    # Perform secondary semantic cleansing on remaining lines
    for i, ln in enumerate(resolved_ing_lines):
        if i in to_remove_from_ing:
            continue
        ing_score = ln.get("ingredient_score", 0.0)
        nut_score = ln.get("nutrition_score", 0.0)
        other_score = ln.get("other_score", 0.0)
        
        # Disqualify if it is overwhelmingly a nutrition row
        if nut_score > ing_score + 0.3 and nut_score > 0.6:
            to_remove_from_ing.add(i)
        # Disqualify if it is clearly part of customer care, address, etc.
        elif other_score > ing_score + 0.3 and other_score > 0.7:
            to_remove_from_ing.add(i)

    for j, ln in enumerate(resolved_nut_lines):
        if j in to_remove_from_nut:
            continue
        ing_score = ln.get("ingredient_score", 0.0)
        nut_score = ln.get("nutrition_score", 0.0)
        other_score = ln.get("other_score", 0.0)
        
        # Disqualify if it is overwhelmingly an ingredient line
        if ing_score > nut_score + 0.3 and ing_score > 0.6:
            to_remove_from_nut.add(j)
        # Disqualify if it is other info
        elif other_score > nut_score + 0.3 and other_score > 0.7:
            to_remove_from_nut.add(j)

    # Construct clean lists
    final_ing_lines = [ln for i, ln in enumerate(resolved_ing_lines) if i not in to_remove_from_ing]
    final_nut_lines = [ln for j, ln in enumerate(resolved_nut_lines) if j not in to_remove_from_nut]

    new_ingredient_result = dict(ingredient_result)
    new_nutrition_result = dict(nutrition_result)

    # Recompute Ingredients Box
    if final_ing_lines:
        refine_res = refine_ingredient_roi(final_ing_lines, image_shape)
        new_bbox = refine_res["refined_bbox"]
        new_ingredient_result["bbox"] = new_bbox
        new_ingredient_result["refinement"] = refine_res
        new_ingredient_result["matched_items"] = final_ing_lines
        new_ingredient_result["confidence"] = round(max(0.05, ingredient_result["confidence"] - 0.02), 3)
    else:
        new_ingredient_result["bbox"] = None
        new_ingredient_result["matched_items"] = []
        new_ingredient_result["confidence"] = 0.0

    # Recompute Nutrition Box
    if final_nut_lines:
        refine_res = refine_nutrition_roi(final_nut_lines, image_shape)
        new_bbox = refine_res["refined_bbox"]
        new_nutrition_result["bbox"] = new_bbox
        new_nutrition_result["refinement"] = refine_res
        new_nutrition_result["matched_items"] = final_nut_lines
        new_nutrition_result["confidence"] = round(max(0.05, nutrition_result["confidence"] - 0.02), 3)
    else:
        new_nutrition_result["bbox"] = None
        new_nutrition_result["matched_items"] = []
        new_nutrition_result["confidence"] = 0.0

    if "debug" in new_ingredient_result:
        new_ingredient_result["debug"] = dict(new_ingredient_result["debug"])
        new_ingredient_result["debug"]["reconciled_overlap"] = True
        new_ingredient_result["debug"]["iou_before"] = float(iou)
    
    if "debug" in new_nutrition_result:
        new_nutrition_result["debug"] = dict(new_nutrition_result["debug"])
        new_nutrition_result["debug"]["reconciled_overlap"] = True
        new_nutrition_result["debug"]["iou_before"] = float(iou)

    return new_ingredient_result, new_nutrition_result
