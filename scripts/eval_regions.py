import os
import sys
import json
import numpy as np

# Ensure project root is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import main

def compute_iou(boxA, boxB):
    if not boxA or not boxB:
        # If both are None, IoU is 1.0 (perfect match of non-existence)
        # If only one is None, IoU is 0.0
        return 1.0 if (not boxA and not boxB) else 0.0
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea <= 0.0:
        return 0.0
    return interArea / unionArea

def main_eval():
    gt_path = os.path.join(project_root, "ground_truth.json")
    if not os.path.exists(gt_path):
        print(f"Error: ground_truth.json not found at {gt_path}")
        sys.exit(1)
        
    with open(gt_path, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
        
    results = []
    ing_ious = []
    nut_ious = []
    
    output_dir = os.path.join(project_root, "output_eval")
    os.makedirs(output_dir, exist_ok=True)
    
    for img_name, gt in ground_truth.items():
        img_path = os.path.join(project_root, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found at {img_path}, skipping.")
            continue
            
        print(f"Running pipeline on {img_name}...")
        try:
            res = main.run_pipeline(img_path, output_dir, domain_arg="auto", test_mode=False)
            
            pred_ing = res["ingredients_region"]["bbox"]
            pred_nut = res["nutrition_region"]["bbox"] if res["nutrition_region"] else None
            
            gt_ing = gt["ingredients"]
            gt_nut = gt["nutrition"]
            
            ing_iou = compute_iou(pred_ing, gt_ing)
            nut_iou = compute_iou(pred_nut, gt_nut)
            
            avg_iou = (ing_iou + nut_iou) / 2.0
            
            results.append({
                "image": img_name,
                "ing_iou": ing_iou,
                "nut_iou": nut_iou,
                "avg_iou": avg_iou,
                "pred_ing": pred_ing,
                "gt_ing": gt_ing,
                "pred_nut": pred_nut,
                "gt_nut": gt_nut
            })
            
            ing_ious.append(ing_iou)
            nut_ious.append(nut_iou)
            
        except Exception as e:
            print(f"Error running pipeline on {img_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # Sort results: worst offenders first (lowest avg_iou)
    results.sort(key=lambda x: x["avg_iou"])
    
    print("\n" + "="*80)
    print(" EVALUATION RESULTS (Worst Offenders First)")
    print("="*80)
    print(f"{'Image':<16} | {'Ingredients IoU':<15} | {'Nutrition IoU':<15} | {'Average IoU':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['image']:<16} | {r['ing_iou']:15.3f} | {r['nut_iou']:15.3f} | {r['avg_iou']:12.3f}")
    print("-"*80)
    
    mean_ing_iou = np.mean(ing_ious) if ing_ious else 0.0
    mean_nut_iou = np.mean(nut_ious) if nut_ious else 0.0
    overall_miou = (mean_ing_iou + mean_nut_iou) / 2.0
    
    print(f"Mean Ingredients IoU : {mean_ing_iou:.4f}")
    print(f"Mean Nutrition IoU   : {mean_nut_iou:.4f}")
    print(f"Overall Mean IoU     : {overall_miou:.4f}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main_eval()
