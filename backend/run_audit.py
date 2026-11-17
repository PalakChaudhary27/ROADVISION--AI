"""
RoadVision AI - Comprehensive Model Audit Suite
Investigates model detection quality, annotation fidelity, class mapping,
preprocessing, confusion matrix, and generates visual comparisons.
"""

import os
import io
import sys
import json
import time
import base64
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import httpx
import torch

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path("D:/ROADVISION AI")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "roadvision_model_artifacts"
DATA_DIR = Path("D:/archive/data")
OUTPUT_DIR = PROJECT_ROOT / "audit_results"
VIS_DIR = OUTPUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    0: {"name": "background", "color": (0, 0, 0), "label": "Background"},
    1: {"name": "pothole", "color": (255, 59, 48), "label": "Pothole"},
    2: {"name": "crack", "color": (255, 204, 0), "label": "Crack"},
}
DATASET_TO_MODEL = {0: 1, 1: 2, 2: 0}


def step1_audit_checkpoint() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 1: AUDITING MODEL CHECKPOINT & ARCHITECTURE CONTRACT")
    print("=" * 70)

    weights_path = ARTIFACTS_DIR / "best_model.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

    file_size_bytes = weights_path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Compute hashes
    hasher_md5 = hashlib.md5()
    hasher_sha = hashlib.sha256()
    with open(weights_path, "rb") as f:
        while chunk := f.read(65536):
            hasher_md5.update(chunk)
            hasher_sha.update(chunk)

    md5_hash = hasher_md5.hexdigest()
    sha_hash = hasher_sha.hexdigest()

    # Load state dict
    state_dict = torch.load(weights_path, map_location="cpu")
    num_keys = len(state_dict)

    # Verify key architectural layers
    classifier_shape = list(state_dict.get("classifier.4.weight", torch.tensor([])).shape)
    aux_shape = list(state_dict.get("aux_classifier.4.weight", torch.tensor([])).shape)
    backbone_conv1 = list(state_dict.get("backbone.conv1.weight", torch.tensor([])).shape)

    # Check for unexpected or missing shapes
    is_correct_3class = (classifier_shape == [3, 256, 1, 1])
    is_correct_aux = (aux_shape == [3, 256, 1, 1])

    info = {
        "file_path": str(weights_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "md5": md5_hash,
        "sha256": sha_hash,
        "total_state_dict_keys": num_keys,
        "classifier_weight_shape": classifier_shape,
        "aux_classifier_weight_shape": aux_shape,
        "backbone_conv1_shape": backbone_conv1,
        "is_correct_3class": is_correct_3class,
        "is_correct_aux": is_correct_aux,
    }

    print(f"  Checkpoint path: {weights_path}")
    print(f"  File size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")
    print(f"  MD5: {md5_hash}")
    print(f"  Total parameter keys: {num_keys}")
    print(f"  Classifier head weight shape: {classifier_shape} (3 output classes: bg, pothole, crack)")
    print(f"  Auxiliary classifier shape: {aux_shape}")
    print(f"  Architectural Contract Match: {'PASS' if is_correct_3class and is_correct_aux else 'FAIL'}")

    return info


def step2_audit_annotations() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 2: AUDITING DATASET ANNOTATIONS (labels/*.txt)")
    print("=" * 70)

    labels_dir = DATA_DIR / "labels"
    label_files = list(labels_dir.glob("*.txt"))

    total_files = len(label_files)
    total_objects = 0
    pothole_count = 0
    crack_count = 0
    manhole_count = 0
    unknown_count = 0

    points_per_object = {}
    box_aspect_ratios = []
    box_areas = []
    axis_aligned_count = 0

    empty_annotation_files = 0

    for lf in label_files:
        lines = lf.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            empty_annotation_files += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            raw_cls = int(float(parts[0]))
            total_objects += 1

            if raw_cls == 0:
                pothole_count += 1
            elif raw_cls == 1:
                crack_count += 1
            elif raw_cls == 2:
                manhole_count += 1
            else:
                unknown_count += 1

            coords = [float(v) for v in parts[1:]]
            num_pts = len(coords) // 2
            points_per_object[num_pts] = points_per_object.get(num_pts, 0) + 1

            if num_pts == 4:
                xs = coords[0::2]
                ys = coords[1::2]
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
                if w > 0 and h > 0:
                    box_aspect_ratios.append(w / h)
                    box_areas.append(w * h)
                # Check if axis-aligned rectangle
                # In axis-aligned: xs has 2 distinct values, ys has 2 distinct values
                x_unique = len(set([round(x, 4) for x in xs]))
                y_unique = len(set([round(y, 4) for y in ys]))
                if x_unique == 2 and y_unique == 2:
                    axis_aligned_count += 1

    pct_4pt = (points_per_object.get(4, 0) / max(total_objects, 1)) * 100
    pct_axis_aligned = (axis_aligned_count / max(total_objects, 1)) * 100

    info = {
        "total_label_files": total_files,
        "empty_label_files": empty_annotation_files,
        "total_annotated_objects": total_objects,
        "pothole_objects (raw cls 0)": pothole_count,
        "crack_objects (raw cls 1)": crack_count,
        "manhole_objects (raw cls 2)": manhole_count,
        "unknown_objects": unknown_count,
        "points_per_object_distribution": points_per_object,
        "percentage_4_point_annotations": round(pct_4pt, 2),
        "percentage_axis_aligned_boxes": round(pct_axis_aligned, 2),
        "mean_box_area_pct": round(float(np.mean(box_areas)) * 100, 2) if box_areas else 0,
        "median_box_aspect_ratio": round(float(np.median(box_aspect_ratios)), 2) if box_aspect_ratios else 0,
    }

    print(f"  Total annotation files: {total_files}")
    print(f"  Empty annotation files (no damage): {empty_annotation_files}")
    print(f"  Total annotated objects: {total_objects}")
    print(f"    - Potholes (0): {pothole_count} ({pothole_count/total_objects*100:.1f}%)")
    print(f"    - Cracks (1):   {crack_count} ({crack_count/total_objects*100:.1f}%)")
    print(f"    - Manholes (2): {manhole_count} ({manhole_count/total_objects*100:.1f}%) [Excluded in training]")
    print(f"  Points per object distribution: {points_per_object}")
    print(f"  Percentage of annotations that are 4-point bounding boxes: {pct_4pt:.1f}%")
    print(f"  Percentage that are strictly axis-aligned rectangles: {pct_axis_aligned:.1f}%")
    print(f"  Mean bounding box area: {info['mean_box_area_pct']}% of full image")

    return info


def decode_ground_truth_mask(anno_path: Path, width: int, height: int) -> np.ndarray:
    """Renders polygon annotations into ground-truth integer mask."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if anno_path.exists() and anno_path.stat().st_size > 0:
        lines = anno_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            raw_cls = int(float(parts[0]))
            model_cls = DATASET_TO_MODEL.get(raw_cls, 0)
            if model_cls == 0:
                continue

            coords = [float(v) for v in parts[1:]]
            poly_pts = []
            for i in range(0, len(coords) - 1, 2):
                px = int(min(max(coords[i] * width, 0), width - 1))
                py = int(min(max(coords[i + 1] * height, 0), height - 1))
                poly_pts.append((px, py))

            if len(poly_pts) >= 3:
                draw.polygon(poly_pts, fill=model_cls)

    return np.array(mask_img, dtype=np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, info in CLASS_MAP.items():
        rgb[mask == cls_id] = info["color"]
    return rgb


def build_error_map(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Creates pixel-level diagnostic error map:
    - Green (0, 255, 0): True Positive (Damage correctly detected)
    - Red (255, 0, 0): False Negative (GT damage missed by model)
    - Blue/Cyan (0, 180, 255): False Positive (Damage predicted where GT is background)
    - Dark Gray (40, 40, 40): True Negative (Background correctly identified)
    - Yellow (255, 200, 0): Class Confusion (Damage detected, but wrong class)
    """
    h, w = gt_mask.shape
    error_rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # default: TN background

    gt_damage = gt_mask > 0
    pred_damage = pred_mask > 0

    # True Negative (Background -> Background)
    # Already 40, 40, 40

    # True Positive (Exact class match)
    tp_exact = (gt_mask == pred_mask) & gt_damage
    error_rgb[tp_exact] = [0, 230, 80]  # Bright Green

    # Class Confusion (Both say damage, but different class)
    confusion = gt_damage & pred_damage & (gt_mask != pred_mask)
    error_rgb[confusion] = [255, 200, 0]  # Yellow

    # False Negative (Missed damage: GT has damage, pred is background)
    fn = gt_damage & (~pred_damage)
    error_rgb[fn] = [255, 30, 30]  # Red

    # False Positive (Spurious damage: GT is background, pred has damage)
    fp = (~gt_damage) & pred_damage
    error_rgb[fp] = [0, 150, 255]  # Blue/Cyan

    return error_rgb


def create_visual_panel(
    orig_img: Image.Image,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    sample_id: str,
    metrics: Dict[str, Any]
) -> Image.Image:
    w, h = orig_img.size
    panel_w = 400
    panel_h = int(h * (panel_w / w))

    # Resize all to panel dimensions
    orig_resized = orig_img.resize((panel_w, panel_h), Image.Resampling.BILINEAR)

    gt_pil = Image.fromarray(gt_mask).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    gt_rgb = mask_to_rgb(np.array(gt_pil))
    gt_panel = Image.fromarray(gt_rgb)

    pred_pil = Image.fromarray(pred_mask).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    pred_rgb = mask_to_rgb(np.array(pred_pil))
    pred_panel = Image.fromarray(pred_rgb)

    # Overlay
    orig_np = np.array(orig_resized)
    dmg_mask = np.array(pred_pil) > 0
    overlay_np = orig_np.copy()
    overlay_np[dmg_mask] = (0.45 * orig_np[dmg_mask] + 0.55 * pred_rgb[dmg_mask]).astype(np.uint8)
    overlay_panel = Image.fromarray(overlay_np)

    # Error Map
    err_rgb = build_error_map(np.array(gt_pil), np.array(pred_pil))
    err_panel = Image.fromarray(err_rgb)

    # Compose 5 panels side by side with header
    header_h = 60
    total_w = panel_w * 5 + 24
    total_h = panel_h + header_h + 30

    canvas = Image.new("RGB", (total_w, total_h), (24, 28, 36))
    draw = ImageDraw.Draw(canvas)

    titles = [
        "1. Original Image",
        "2. Ground Truth Mask (Approx)",
        "3. Model Prediction",
        "4. Prediction Overlay",
        "5. Diagnostic Error Map"
    ]
    subtitles = [
        f"Sample: {sample_id}",
        f"GT Dmg: {metrics.get('gt_dmg_pct', 0):.2f}%",
        f"Pred Dmg: {metrics.get('pred_dmg_pct', 0):.2f}%",
        f"Severity: {metrics.get('severity', 'N/A')}",
        "Green=TP | Red=FN (Missed) | Blue=FP"
    ]

    panels = [orig_resized, gt_panel, pred_panel, overlay_panel, err_panel]

    for idx, (panel, title, subtitle) in enumerate(zip(panels, titles, subtitles)):
        x_offset = 12 + idx * (panel_w + 2)
        y_offset = header_h

        # Draw panel title
        draw.text((x_offset + 5, 12), title, fill=(255, 255, 255))
        draw.text((x_offset + 5, 34), subtitle, fill=(180, 195, 215))

        # Paste panel
        canvas.paste(panel, (x_offset, y_offset))

        # Panel border
        draw.rectangle(
            [x_offset, y_offset, x_offset + panel_w, y_offset + panel_h],
            outline=(60, 70, 90),
            width=1
        )

    # Footer note
    draw.text(
        (12, total_h - 22),
        f"Metrics: Pothole IoU={metrics.get('iou_pothole', 0):.3f} | Crack IoU={metrics.get('iou_crack', 0):.3f} | FN Pixels={metrics.get('fn_pixels', 0):,}",
        fill=(150, 165, 185)
    )

    return canvas


def step3_evaluate_and_visualize() -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("STEP 3: TESTING PREDICTIONS, COMPUTING CONFUSION MATRIX & VISUALS")
    print("=" * 70)

    images_dir = DATA_DIR / "images"
    labels_dir = DATA_DIR / "labels"

    all_images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    all_labels = {p.stem: p for p in labels_dir.glob("*.txt")}

    paired = []
    for img_path in sorted(all_images):
        if img_path.stem in all_labels:
            paired.append({
                "image_path": img_path,
                "anno_path": all_labels[img_path.stem],
                "stem": img_path.stem
            })

    # Exact deterministic split
    random.seed(42)
    indices = list(range(len(paired)))
    random.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    test_indices = indices[n_train + n_val:]

    test_samples = [paired[i] for i in test_indices]
    print(f"  Total test set size: {len(test_samples)} images")

    # Evaluate a representative curated set of 25 test images (covering both potholes, cracks, and clean roads)
    # plus the 4 prominent UI sample images
    eval_set = []

    # Add 4 UI sample images first
    ui_samples = [
        "20250216_164325",  # road_pothole_and_crack.jpg
        "20250216_164541",  # road_multiple_damage.jpg
        "20250216_164521",  # road_cracks.jpg
        "20250219_164714",  # road_potholes.jpg
    ]
    for stem in ui_samples:
        img_p = images_dir / f"{stem}.jpg"
        lbl_p = labels_dir / f"{stem}.txt"
        if img_p.exists() and lbl_p.exists():
            eval_set.append({
                "image_path": img_p,
                "anno_path": lbl_p,
                "stem": stem,
                "is_ui_sample": True
            })

    # Add 20 test split images
    for s in test_samples[:20]:
        if s["stem"] not in ui_samples:
            eval_set.append({**s, "is_ui_sample": False})

    print(f"  Evaluating {len(eval_set)} real images via live API (4 UI samples + 20 held-out test)...")

    # 3x3 Confusion Matrix: [GT_class, Pred_class]
    # 0=background, 1=pothole, 2=crack
    cm = np.zeros((3, 3), dtype=np.int64)

    client = httpx.Client(timeout=45.0)
    sample_reports = []

    total_images_processed = 0

    for idx, item in enumerate(eval_set):
        stem = item["stem"]
        img_path = item["image_path"]
        anno_path = item["anno_path"]

        orig_pil = Image.open(img_path).convert("RGB")
        w, h = orig_pil.size

        # 1. Ground truth mask
        gt_mask = decode_ground_truth_mask(anno_path, w, h)

        # 2. Get prediction from live server
        with open(img_path, "rb") as f:
            files = {"file": (img_path.name, f.read(), "image/jpeg")}

        t0 = time.time()
        try:
            resp = client.post("http://localhost:8000/predict", files=files)
            if resp.status_code != 200:
                print(f"    [WARN] Failed to predict {stem}: {resp.status_code}")
                continue
            pred_data = resp.json()
        except Exception as e:
            print(f"    [WARN] Error calling predict for {stem}: {e}")
            continue
        latency = time.time() - t0

        # Decode predicted mask PNG from base64
        mask_b64 = pred_data["images"]["mask"].split(",")[-1]
        mask_bytes = base64.b64decode(mask_b64)
        pred_pil_rgba = Image.open(io.BytesIO(mask_bytes))
        
        # In backend: pred_512 is mapped to colors, but the backend also returns pred_mask in images['mask']
        # Let's inspect backend mask encoding:
        # In main.py:
        #   color_mask = np.zeros((orig_h, orig_w, 3))
        #   color_mask[pred_mask == 1] = pothole_rgb
        #   color_mask[pred_mask == 2] = crack_rgb
        # So mask_image is the color mask!
        # We can reconstruct the integer class mask from RGB colors:
        pred_rgb_arr = np.array(pred_pil_rgba.convert("RGB"))
        pred_mask = np.zeros((h, w), dtype=np.uint8)
        # Pothole: [255, 59, 48]
        pothole_diff = np.abs(pred_rgb_arr.astype(int) - [255, 59, 48])
        is_pothole = np.all(pothole_diff <= 10, axis=-1)
        # Crack: [255, 204, 0]
        crack_diff = np.abs(pred_rgb_arr.astype(int) - [255, 204, 0])
        is_crack = np.all(crack_diff <= 10, axis=-1)

        pred_mask[is_pothole] = 1
        pred_mask[is_crack] = 2

        # Accumulate confusion matrix
        gt_flat = gt_mask.flatten()
        pred_flat = pred_mask.flatten()
        for g in range(3):
            for p in range(3):
                cm[g, p] += np.sum((gt_flat == g) & (pred_flat == p))

        # Sample-level stats
        gt_dmg = np.sum(gt_mask > 0)
        pred_dmg = np.sum(pred_mask > 0)
        gt_dmg_pct = (gt_dmg / gt_mask.size) * 100
        pred_dmg_pct = (pred_dmg / pred_mask.size) * 100

        # Per-class IoU for this sample
        sample_metrics = {}
        for c, cname in [(1, "pothole"), (2, "crack")]:
            inter = np.sum((gt_mask == c) & (pred_mask == c))
            union = np.sum((gt_mask == c) | (pred_mask == c))
            iou = inter / union if union > 0 else (1.0 if inter == 0 else 0.0)
            sample_metrics[f"iou_{cname}"] = round(float(iou), 4)

        tp = np.sum((gt_mask > 0) & (pred_mask > 0))
        fn = np.sum((gt_mask > 0) & (pred_mask == 0))
        fp = np.sum((gt_mask == 0) & (pred_mask > 0))

        metrics_for_vis = {
            "gt_dmg_pct": gt_dmg_pct,
            "pred_dmg_pct": pred_dmg_pct,
            "severity": pred_data.get("severity", "N/A"),
            "iou_pothole": sample_metrics["iou_pothole"],
            "iou_crack": sample_metrics["iou_crack"],
            "fn_pixels": int(fn),
            "fp_pixels": int(fp),
            "tp_pixels": int(tp),
        }

        # Generate visual panel
        vis_img = create_visual_panel(orig_pil, gt_mask, pred_mask, stem, metrics_for_vis)
        vis_path = VIS_DIR / f"audit_sample_{idx+1:02d}_{stem}.png"
        vis_img.save(vis_path, quality=92)

        sample_reports.append({
            "sample_index": idx + 1,
            "stem": stem,
            "image_path": str(img_path),
            "vis_image_path": str(vis_path),
            "gt_damage_pct": round(gt_dmg_pct, 2),
            "pred_damage_pct": round(pred_dmg_pct, 2),
            "pothole_iou": sample_metrics["iou_pothole"],
            "crack_iou": sample_metrics["iou_crack"],
            "fn_pixels": int(fn),
            "fp_pixels": int(fp),
            "tp_pixels": int(tp),
            "latency_sec": round(latency, 2),
        })

        total_images_processed += 1
        print(f"    [{total_images_processed}/{len(eval_set)}] Processed {stem}: GT={gt_dmg_pct:.1f}%, Pred={pred_dmg_pct:.1f}%, FN={fn:,}px, IoU(P)={sample_metrics['iou_pothole']:.2f}, IoU(C)={sample_metrics['iou_crack']:.2f} ({latency:.1f}s)")

    # Compute overall confusion matrix metrics
    tp_diag = np.diag(cm)
    total_pixels = cm.sum()
    fp_col = cm.sum(axis=0) - tp_diag
    fn_row = cm.sum(axis=1) - tp_diag

    class_metrics = {}
    ious = []
    dices = []

    for c in range(3):
        cname = CLASS_MAP[c]["name"]
        tp = float(tp_diag[c])
        fp = float(fp_col[c])
        fn = float(fn_row[c])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        class_metrics[cname] = {
            "class_id": c,
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "iou": round(iou, 4),
            "dice": round(dice, 4),
        }
        ious.append(iou)
        dices.append(dice)

    mIoU = float(np.mean(ious))
    mDice = float(np.mean(dices))

    cm_list = cm.tolist()

    # Breakdown of Missed Damage (False Negatives)
    # GT Pothole (row 1) predicted as Background (col 0)
    pothole_missed_as_bg = int(cm[1, 0])
    pothole_confused_as_crack = int(cm[1, 2])
    pothole_correct = int(cm[1, 1])
    total_gt_pothole = int(cm[1, :].sum())

    # GT Crack (row 2) predicted as Background (col 0)
    crack_missed_as_bg = int(cm[2, 0])
    crack_confused_as_pothole = int(cm[2, 1])
    crack_correct = int(cm[2, 2])
    total_gt_crack = int(cm[2, :].sum())

    confusion_analysis = {
        "potholes": {
            "total_gt_pixels": total_gt_pothole,
            "correctly_detected": pothole_correct,
            "missed_as_background (FN)": pothole_missed_as_bg,
            "miss_rate_percentage": round((pothole_missed_as_bg / max(total_gt_pothole, 1)) * 100, 2),
            "confused_as_crack": pothole_confused_as_crack,
            "confusion_rate_with_crack": round((pothole_confused_as_crack / max(total_gt_pothole, 1)) * 100, 2),
        },
        "cracks": {
            "total_gt_pixels": total_gt_crack,
            "correctly_detected": crack_correct,
            "missed_as_background (FN)": crack_missed_as_bg,
            "miss_rate_percentage": round((crack_missed_as_bg / max(total_gt_crack, 1)) * 100, 2),
            "confused_as_pothole": crack_confused_as_pothole,
            "confusion_rate_with_pothole": round((crack_confused_as_pothole / max(total_gt_crack, 1)) * 100, 2),
        }
    }

    print("\n" + "-" * 70)
    print("EVALUATION RESULTS & CONFUSION MATRIX")
    print("-" * 70)
    print(f"  Confusion Matrix [Rows=Ground Truth, Cols=Prediction]:")
    print(f"              Pred_BG       Pred_Pothole   Pred_Crack")
    print(f"  GT_BG      : {cm[0,0]:12d}  {cm[0,1]:12d}  {cm[0,2]:12d}")
    print(f"  GT_Pothole : {cm[1,0]:12d}  {cm[1,1]:12d}  {cm[1,2]:12d}")
    print(f"  GT_Crack   : {cm[2,0]:12d}  {cm[2,1]:12d}  {cm[2,2]:12d}")
    print("-" * 70)
    for cname, m in class_metrics.items():
        print(f"  Class {cname.upper():10s}: Precision={m['precision']:.4f} | Recall={m['recall']:.4f} | IoU={m['iou']:.4f} | Dice={m['dice']:.4f}")
    print(f"  Overall mIoU: {mIoU:.4f} | Overall mDice: {mDice:.4f}")
    print("-" * 70)
    print(f"  POTHOLE MISS RATE: {confusion_analysis['potholes']['miss_rate_percentage']}% of ground-truth pothole pixels were predicted as Background!")
    print(f"  CRACK MISS RATE:   {confusion_analysis['cracks']['miss_rate_percentage']}% of ground-truth crack pixels were predicted as Background!")

    return {
        "images_evaluated": total_images_processed,
        "confusion_matrix": cm_list,
        "class_metrics": class_metrics,
        "mean_iou": round(mIoU, 4),
        "mean_dice": round(mDice, 4),
        "confusion_analysis": confusion_analysis,
        "samples": sample_reports,
    }


def main():
    print("=" * 70)
    print("ROADVISION AI - DEEPLABV3-RESNET50 AUDIT & DIAGNOSTIC INVESTIGATION")
    print("=" * 70)

    # 1. Audit Checkpoint
    ckpt_audit = step1_audit_checkpoint()

    # 2. Audit Annotations
    anno_audit = step2_audit_annotations()

    # 3. Evaluate Predictions and Generate Visuals
    eval_audit = step3_evaluate_and_visualize()

    # Final combined report
    full_report = {
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_audit": ckpt_audit,
        "annotations_audit": anno_audit,
        "evaluation_audit": eval_audit,
    }

    report_path = OUTPUT_DIR / "audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"AUDIT COMPLETE! Full JSON report written to: {report_path}")
    print(f"Visual comparison images written to: {VIS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
