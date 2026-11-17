"""
RoadVision AI - Calibrated Softmax Thresholding Experiment
Evaluates baseline argmax vs. calibrated probability thresholding on DeepLabV3-ResNet50.
Performs grid search on validation data, evaluates on test data, and generates visual comparisons.
"""

import os
import io
import sys
import json
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path("D:/ROADVISION AI")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "roadvision_model_artifacts"
DATA_DIR = Path("D:/archive/data")
EXP_DIR = PROJECT_ROOT / "audit_results" / "threshold_experiment"
VIS_DIR = EXP_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    0: {"name": "background", "color": (0, 0, 0), "label": "Background"},
    1: {"name": "pothole", "color": (255, 59, 48), "label": "Pothole"},
    2: {"name": "crack", "color": (255, 204, 0), "label": "Crack"},
}
DATASET_TO_MODEL = {0: 1, 1: 2, 2: 0}


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
    Diagnostic error map:
    - Green (0, 230, 80): True Positive
    - Red (255, 30, 30): False Negative (GT damage missed as Background)
    - Yellow (255, 200, 0): Class Confusion (Damage detected, but wrong class)
    - Blue/Cyan (0, 150, 255): False Positive (Spurious damage on background)
    - Dark Gray (40, 40, 40): True Negative
    """
    h, w = gt_mask.shape
    err_rgb = np.full((h, w, 3), 40, dtype=np.uint8)

    gt_dmg = gt_mask > 0
    pred_dmg = pred_mask > 0

    tp_exact = (gt_mask == pred_mask) & gt_dmg
    err_rgb[tp_exact] = [0, 230, 80]

    confusion = gt_dmg & pred_dmg & (gt_mask != pred_mask)
    err_rgb[confusion] = [255, 200, 0]

    fn = gt_dmg & (~pred_dmg)
    err_rgb[fn] = [255, 30, 30]

    fp = (~gt_dmg) & pred_dmg
    err_rgb[fp] = [0, 150, 255]

    return err_rgb


def assign_classes(probs: np.ndarray, tau_pothole: float, tau_crack: float) -> np.ndarray:
    """
    Deterministic class-assignment rule:
    - Default is Background (0).
    - Pothole qualifies if P(pothole) >= tau_pothole.
    - Crack qualifies if P(crack) >= tau_crack.
    - If both qualify: resolve by higher probability.
    - Never assign a pixel to multiple classes.
    """
    p_pot = probs[1]
    p_crk = probs[2]

    qual_pot = p_pot >= tau_pothole
    qual_crk = p_crk >= tau_crack

    mask = np.zeros(probs[0].shape, dtype=np.uint8)

    both = qual_pot & qual_crk
    mask[both & (p_pot >= p_crk)] = 1
    mask[both & (p_crk > p_pot)] = 2

    mask[qual_pot & (~qual_crk)] = 1
    mask[qual_crk & (~qual_pot)] = 2

    return mask


def assign_argmax(probs: np.ndarray) -> np.ndarray:
    return np.argmax(probs, axis=0).astype(np.uint8)


def compute_metrics(gt_masks: List[np.ndarray], pred_masks: List[np.ndarray]) -> Dict[str, Any]:
    cm = np.zeros((3, 3), dtype=np.int64)
    for gt, pred in zip(gt_masks, pred_masks):
        gt_f = gt.flatten()
        pr_f = pred.flatten()
        for g in range(3):
            for p in range(3):
                cm[g, p] += np.sum((gt_f == g) & (pr_f == p))

    tp_diag = np.diag(cm)
    fp_col = cm.sum(axis=0) - tp_diag
    fn_row = cm.sum(axis=1) - tp_diag

    class_res = {}
    ious, dices = [], []

    for c in range(3):
        cname = CLASS_MAP[c]["name"]
        tp = float(tp_diag[c])
        fp = float(fp_col[c])
        fn = float(fn_row[c])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        class_res[cname] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "iou": round(iou, 4),
            "dice": round(dice, 4),
        }
        ious.append(iou)
        dices.append(dice)

    # Damage-specific metrics (excluding background)
    dmg_iou = float(np.mean([class_res["pothole"]["iou"], class_res["crack"]["iou"]]))
    dmg_dice = float(np.mean([class_res["pothole"]["dice"], class_res["crack"]["dice"]]))
    dmg_recall = float(np.mean([class_res["pothole"]["recall"], class_res["crack"]["recall"]]))
    dmg_prec = float(np.mean([class_res["pothole"]["precision"], class_res["crack"]["precision"]]))

    # Specific miss rates to background
    pothole_gt_total = cm[1, :].sum()
    pothole_missed_as_bg = cm[1, 0]
    pothole_confused_as_crack = cm[1, 2]

    crack_gt_total = cm[2, :].sum()
    crack_missed_as_bg = cm[2, 0]
    crack_confused_as_pothole = cm[2, 1]

    return {
        "confusion_matrix": cm.tolist(),
        "classes": class_res,
        "mean_iou": round(float(np.mean(ious)), 4),
        "mean_dice": round(float(np.mean(dices)), 4),
        "damage_mean_iou": round(dmg_iou, 4),
        "damage_mean_dice": round(dmg_dice, 4),
        "damage_mean_recall": round(dmg_recall, 4),
        "damage_mean_precision": round(dmg_prec, 4),
        "miss_breakdown": {
            "pothole": {
                "gt_total": int(pothole_gt_total),
                "missed_as_bg": int(pothole_missed_as_bg),
                "miss_rate_bg_pct": round((pothole_missed_as_bg / max(pothole_gt_total, 1)) * 100, 2),
                "confused_as_crack": int(pothole_confused_as_crack),
                "confusion_rate_pct": round((pothole_confused_as_crack / max(pothole_gt_total, 1)) * 100, 2),
            },
            "crack": {
                "gt_total": int(crack_gt_total),
                "missed_as_bg": int(crack_missed_as_bg),
                "miss_rate_bg_pct": round((crack_missed_as_bg / max(crack_gt_total, 1)) * 100, 2),
                "confused_as_pothole": int(crack_confused_as_pothole),
                "confusion_rate_pct": round((crack_confused_as_pothole / max(crack_gt_total, 1)) * 100, 2),
            }
        }
    }


def create_comparison_panel(
    orig_img: Image.Image,
    gt_mask: np.ndarray,
    pred_argmax: np.ndarray,
    pred_thresh: np.ndarray,
    sample_id: str,
    meta_info: Dict[str, Any]
) -> Image.Image:
    w, h = orig_img.size
    panel_w = 340
    panel_h = int(h * (panel_w / w))

    orig_resized = orig_img.resize((panel_w, panel_h), Image.Resampling.BILINEAR)

    # 1. Ground Truth
    gt_pil = Image.fromarray(gt_mask).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    gt_panel = Image.fromarray(mask_to_rgb(np.array(gt_pil)))

    # 2. Argmax prediction
    arm_pil = Image.fromarray(pred_argmax).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    arm_panel = Image.fromarray(mask_to_rgb(np.array(arm_pil)))

    # 3. Threshold prediction
    thr_pil = Image.fromarray(pred_thresh).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    thr_panel = Image.fromarray(mask_to_rgb(np.array(thr_pil)))

    # 4. Argmax error map
    arm_err = Image.fromarray(build_error_map(np.array(gt_pil), np.array(arm_pil)))

    # 5. Threshold error map
    thr_err = Image.fromarray(build_error_map(np.array(gt_pil), np.array(thr_pil)))

    header_h = 55
    total_w = panel_w * 6 + 28
    total_h = panel_h + header_h + 30

    canvas = Image.new("RGB", (total_w, total_h), (24, 28, 36))
    draw = ImageDraw.Draw(canvas)

    # Overlay for threshold prediction
    orig_np = np.array(orig_resized)
    dmg_thr = np.array(thr_pil) > 0
    thr_rgb = np.array(thr_panel)
    overlay_np = orig_np.copy()
    overlay_np[dmg_thr] = (0.45 * orig_np[dmg_thr] + 0.55 * thr_rgb[dmg_thr]).astype(np.uint8)
    overlay_panel = Image.fromarray(overlay_np)

    panels = [orig_resized, gt_panel, arm_panel, thr_panel, arm_err, thr_err]
    titles = [
        "1. Original Road Image",
        "2. Ground Truth Mask",
        "3. Baseline (Argmax)",
        f"4. Calibrated Thresh ({meta_info['tau_p']}/{meta_info['tau_c']})",
        "5. Baseline Error Map",
        "6. Thresh Error Map"
    ]
    subtitles = [
        f"ID: {sample_id}",
        f"GT Dmg: {meta_info.get('gt_dmg_pct', 0):.1f}%",
        f"Argmax Dmg: {meta_info.get('arm_dmg_pct', 0):.1f}%",
        f"Thresh Dmg: {meta_info.get('thr_dmg_pct', 0):.1f}%",
        f"FN Missed: {meta_info.get('arm_fn', 0):,} px",
        f"FN Missed: {meta_info.get('thr_fn', 0):,} px"
    ]

    for idx, (p, title, sub) in enumerate(zip(panels, titles, subtitles)):
        x_off = 10 + idx * (panel_w + 3)
        y_off = header_h

        draw.text((x_off + 4, 10), title, fill=(255, 255, 255))
        draw.text((x_off + 4, 30), sub, fill=(180, 195, 215))

        canvas.paste(p, (x_off, y_off))
        draw.rectangle([x_off, y_off, x_off + panel_w, y_off + panel_h], outline=(60, 70, 90), width=1)

    draw.text(
        (10, total_h - 22),
        f"Error Map Legend: Green = True Positive | Red = False Negative (Missed Damage) | Yellow = Pothole/Crack Confusion | Blue = False Positive",
        fill=(150, 165, 185)
    )

    return canvas


def run_experiment():
    print("=" * 75)
    print("ROADVISION AI: CALIBRATED INFERENCE THRESHOLD EXPERIMENT")
    print("=" * 75)

    # 1. Load model once
    t0 = time.time()
    print("Loading DeepLabV3-ResNet50 model from artifacts...")
    weights_path = ARTIFACTS_DIR / "best_model.pth"
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=3, aux_loss=True)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s.")

    # Normalization parameters
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_probs(image_pil: Image.Image) -> np.ndarray:
        w_orig, h_orig = image_pil.size
        # Resize to 512x512
        img_512 = image_pil.resize((512, 512), Image.Resampling.BILINEAR)
        arr = np.array(img_512, dtype=np.float32) / 255.0
        norm = (arr - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            out = model(tensor)["out"]
            probs_512 = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()  # (3, 512, 512)

        # Upsample probability maps back to original resolution
        probs_orig = np.zeros((3, h_orig, w_orig), dtype=np.float32)
        for c in range(3):
            p_c = Image.fromarray(probs_512[c]).resize((w_orig, h_orig), Image.Resampling.BILINEAR)
            probs_orig[c] = np.array(p_c)

        # Renormalize softmax along class axis
        probs_sum = probs_orig.sum(axis=0, keepdims=True)
        probs_orig = probs_orig / np.maximum(probs_sum, 1e-7)
        return probs_orig

    # 2. Reconstruct exact splits
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

    random.seed(42)
    indices = list(range(len(paired)))
    random.shuffle(indices)

    n_train = int(len(indices) * 0.70)
    n_val = int(len(indices) * 0.15)
    val_samples = [paired[i] for i in indices[n_train : n_train + n_val]]
    test_samples = [paired[i] for i in indices[n_train + n_val :]]

    print(f"Dataset split: Total={len(paired)}, Train={n_train}, Validation={len(val_samples)}, Test={len(test_samples)}")

    # 3. Curate Validation Tuning Subset (22 diverse samples)
    def categorize_samples(s_list):
        pot, crk, both, clean = [], [], [], []
        for s in s_list:
            lines = s["anno_path"].read_text(encoding="utf-8").strip().splitlines()
            clss = set([int(float(l.split()[0])) for l in lines if len(l.split()) >= 5])
            if 0 in clss and 1 in clss:
                both.append(s)
            elif 0 in clss:
                pot.append(s)
            elif 1 in clss:
                crk.append(s)
            else:
                clean.append(s)
        return pot, crk, both, clean

    val_pot, val_crk, val_both, val_clean = categorize_samples(val_samples)
    tuning_val_samples = val_pot[:5] + val_crk[:8] + val_both[:5] + val_clean[:4]
    print(f"Selected {len(tuning_val_samples)} validation samples for threshold tuning (5 Pothole, 8 Crack, 5 Both, 4 Clean).")

    # 4. Extract and cache probabilities for validation tuning samples
    print("\n--- Running inference on Validation Tuning set ---")
    val_cached = []
    for idx, s in enumerate(tuning_val_samples):
        img_pil = Image.open(s["image_path"]).convert("RGB")
        w, h = img_pil.size
        gt_mask = decode_ground_truth_mask(s["anno_path"], w, h)
        probs = predict_probs(img_pil)
        val_cached.append({
            "stem": s["stem"],
            "image_path": s["image_path"],
            "anno_path": s["anno_path"],
            "gt_mask": gt_mask,
            "probs": probs
        })
        print(f"  [{idx+1}/{len(tuning_val_samples)}] Cached val probs for {s['stem']} (GT dmg: {np.mean(gt_mask>0)*100:.1f}%)")

    # 5. Execute Threshold Grid Search on Validation Data
    print("\n--- Performing Threshold Grid Search on Validation Set ---")
    grid_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    grid_results = []

    # Evaluate Baseline Argmax on validation
    val_gt_list = [v["gt_mask"] for v in val_cached]
    val_arm_preds = [assign_argmax(v["probs"]) for v in val_cached]
    val_arm_metrics = compute_metrics(val_gt_list, val_arm_preds)
    print(f"\n[Validation Baseline Argmax]: mIoU={val_arm_metrics['mean_iou']:.4f} | Dmg_mIoU={val_arm_metrics['damage_mean_iou']:.4f} | Pothole_IoU={val_arm_metrics['classes']['pothole']['iou']:.4f} (Rec={val_arm_metrics['classes']['pothole']['recall']:.4f}) | Crack_IoU={val_arm_metrics['classes']['crack']['iou']:.4f} (Rec={val_arm_metrics['classes']['crack']['recall']:.4f})")

    best_val_cfg = None
    best_val_score = -1.0  # target: damage_mean_iou or mean_iou

    for tau_p in grid_thresholds:
        for tau_c in grid_thresholds:
            val_thr_preds = [assign_classes(v["probs"], tau_p, tau_c) for v in val_cached]
            metrics = compute_metrics(val_gt_list, val_thr_preds)

            entry = {
                "tau_pothole": tau_p,
                "tau_crack": tau_c,
                "mean_iou": metrics["mean_iou"],
                "damage_mean_iou": metrics["damage_mean_iou"],
                "mean_dice": metrics["mean_dice"],
                "damage_mean_dice": metrics["damage_mean_dice"],
                "pothole": metrics["classes"]["pothole"],
                "crack": metrics["classes"]["crack"],
                "background": metrics["classes"]["background"],
                "miss_breakdown": metrics["miss_breakdown"],
            }
            grid_results.append(entry)

            # Score based on Damage Mean IoU
            score = metrics["damage_mean_iou"]
            if score > best_val_score:
                best_val_score = score
                best_val_cfg = entry

    # Proposed configuration check
    proposed_entry = next((g for g in grid_results if abs(g["tau_pothole"] - 0.30) < 1e-4 and abs(g["tau_crack"] - 0.35) < 1e-4), None)

    print(f"\nTop Validation Grid Results:")
    print(f"  Proposed (P=0.30, C=0.35): Dmg_mIoU={proposed_entry['damage_mean_iou']:.4f} | P_IoU={proposed_entry['pothole']['iou']:.4f} (Rec={proposed_entry['pothole']['recall']:.4f}) | C_IoU={proposed_entry['crack']['iou']:.4f} (Rec={proposed_entry['crack']['recall']:.4f})")
    print(f"  Optimal  (P={best_val_cfg['tau_pothole']}, C={best_val_cfg['tau_crack']}): Dmg_mIoU={best_val_cfg['damage_mean_iou']:.4f} | P_IoU={best_val_cfg['pothole']['iou']:.4f} (Rec={best_val_cfg['pothole']['recall']:.4f}) | C_IoU={best_val_cfg['crack']['iou']:.4f} (Rec={best_val_cfg['crack']['recall']:.4f})")

    # 6. Final Evaluation on Held-Out Test Set + UI Reference Images
    print("\n--- Running Final Evaluation on Held-Out Test Set ---")
    test_pot, test_crk, test_both, test_clean = categorize_samples(test_samples)
    final_test_samples = test_pot[:4] + test_crk[:6] + test_both[:4] + test_clean[:3]

    # Also include the 4 prominent UI reference samples for side-by-side analysis
    ui_stems = [
        "20250216_164325",  # road_pothole_and_crack.jpg
        "20250216_164541",  # road_multiple_damage.jpg
        "20250216_164521",  # road_cracks.jpg
        "20250219_164714",  # road_potholes.jpg
    ]
    ui_samples = []
    for sname in ui_stems:
        ip = images_dir / f"{sname}.jpg"
        ap = labels_dir / f"{sname}.txt"
        if ip.exists() and ap.exists():
            ui_samples.append({"image_path": ip, "anno_path": ap, "stem": sname, "is_ui": True})

    eval_test_pool = ui_samples + [{**s, "is_ui": False} for s in final_test_samples if s["stem"] not in ui_stems]
    print(f"Evaluating {len(eval_test_pool)} images on Test pool (4 UI samples + {len(final_test_samples)} held-out test)...")

    test_cached = []
    for idx, s in enumerate(eval_test_pool):
        img_pil = Image.open(s["image_path"]).convert("RGB")
        w, h = img_pil.size
        gt_mask = decode_ground_truth_mask(s["anno_path"], w, h)
        probs = predict_probs(img_pil)
        test_cached.append({
            "stem": s["stem"],
            "image_path": s["image_path"],
            "anno_path": s["anno_path"],
            "is_ui": s.get("is_ui", False),
            "gt_mask": gt_mask,
            "probs": probs
        })
        print(f"  [{idx+1}/{len(eval_test_pool)}] Cached test probs for {s['stem']} (GT dmg: {np.mean(gt_mask>0)*100:.1f}%)")

    test_gt_list = [t["gt_mask"] for t in test_cached]

    # Configurations to evaluate on Test Set:
    configs_to_eval = [
        ("Baseline Argmax", None, None),
        ("Proposed (P=0.30, C=0.35)", 0.30, 0.35),
        (f"Val-Optimal (P={best_val_cfg['tau_pothole']}, C={best_val_cfg['tau_crack']})", best_val_cfg["tau_pothole"], best_val_cfg["tau_crack"]),
        ("High-Recall (P=0.25, C=0.25)", 0.25, 0.25),
        ("Balanced-Mid (P=0.35, C=0.35)", 0.35, 0.35),
        ("High-Precision (P=0.45, C=0.45)", 0.45, 0.45),
    ]

    test_comparison_results = []
    print("\n" + "=" * 75)
    print("FINAL TEST EVALUATION: BASELINE VS CALIBRATED THRESHOLDS")
    print("=" * 75)

    test_preds_by_cfg = {}
    for name, tau_p, tau_c in configs_to_eval:
        if tau_p is None:
            preds = [assign_argmax(t["probs"]) for t in test_cached]
        else:
            preds = [assign_classes(t["probs"], tau_p, tau_c) for t in test_cached]

        test_preds_by_cfg[name] = preds
        metrics = compute_metrics(test_gt_list, preds)

        print(f"\nConfiguration: {name}")
        print(f"  Overall mIoU: {metrics['mean_iou']:.4f} | Overall mDice: {metrics['mean_dice']:.4f}")
        print(f"  Damage Mean IoU: {metrics['damage_mean_iou']:.4f} | Damage Mean Recall: {metrics['damage_mean_recall']:.4f} | Damage Mean Precision: {metrics['damage_mean_precision']:.4f}")
        print(f"  - POTHOLE: IoU={metrics['classes']['pothole']['iou']:.4f} | Recall={metrics['classes']['pothole']['recall']:.4f} | Precision={metrics['classes']['pothole']['precision']:.4f} | Missed as BG: {metrics['miss_breakdown']['pothole']['miss_rate_bg_pct']}%")
        print(f"  - CRACK  : IoU={metrics['classes']['crack']['iou']:.4f} | Recall={metrics['classes']['crack']['recall']:.4f} | Precision={metrics['classes']['crack']['precision']:.4f} | Missed as BG: {metrics['miss_breakdown']['crack']['miss_rate_bg_pct']}%")

        test_comparison_results.append({
            "name": name,
            "tau_pothole": tau_p,
            "tau_crack": tau_c,
            "metrics": metrics
        })

    # 7. Generate Side-by-Side Visual Comparisons for key test samples
    print("\n--- Generating Side-by-Side Visual Comparisons ---")
    chosen_thr_name = f"Val-Optimal (P={best_val_cfg['tau_pothole']}, C={best_val_cfg['tau_crack']})"
    if chosen_thr_name not in test_preds_by_cfg:
        chosen_thr_name = "Proposed (P=0.30, C=0.35)"

    chosen_thr_preds = test_preds_by_cfg[chosen_thr_name]
    argmax_preds = test_preds_by_cfg["Baseline Argmax"]

    selected_vis_samples = test_cached[:8]  # first 8 including the 4 UI reference images
    vis_records = []

    for idx, item in enumerate(selected_vis_samples):
        stem = item["stem"]
        img_pil = Image.open(item["image_path"]).convert("RGB")
        gt = item["gt_mask"]
        p_arm = argmax_preds[idx]
        p_thr = chosen_thr_preds[idx]

        gt_dmg = np.mean(gt > 0) * 100
        arm_dmg = np.mean(p_arm > 0) * 100
        thr_dmg = np.mean(p_thr > 0) * 100

        arm_fn = int(np.sum((gt > 0) & (p_arm == 0)))
        thr_fn = int(np.sum((gt > 0) & (p_thr == 0)))

        meta_info = {
            "tau_p": best_val_cfg["tau_pothole"],
            "tau_c": best_val_cfg["tau_crack"],
            "gt_dmg_pct": gt_dmg,
            "arm_dmg_pct": arm_dmg,
            "thr_dmg_pct": thr_dmg,
            "arm_fn": arm_fn,
            "thr_fn": thr_fn,
        }

        panel_img = create_comparison_panel(img_pil, gt, p_arm, p_thr, stem, meta_info)
        vis_path = VIS_DIR / f"thresh_comp_{idx+1:02d}_{stem}.png"
        panel_img.save(vis_path, quality=92)
        vis_records.append({
            "sample_index": idx + 1,
            "stem": stem,
            "vis_path": str(vis_path),
            "gt_damage_pct": round(gt_dmg, 2),
            "argmax_damage_pct": round(arm_dmg, 2),
            "thresh_damage_pct": round(thr_dmg, 2),
            "argmax_fn_pixels": arm_fn,
            "thresh_fn_pixels": thr_fn,
            "fn_reduction_pct": round(((arm_fn - thr_fn) / max(arm_fn, 1)) * 100, 2) if arm_fn > 0 else 0
        })
        print(f"  Generated comparison {idx+1}/{len(selected_vis_samples)}: {stem} (FN reduced: {arm_fn:,} -> {thr_fn:,} px, {vis_records[-1]['fn_reduction_pct']}%)")

    # 8. Save comprehensive experiment report
    experiment_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "validation_tuning": {
            "samples_evaluated": len(tuning_val_samples),
            "baseline_argmax": val_arm_metrics,
            "optimal_thresholds": {
                "tau_pothole": best_val_cfg["tau_pothole"],
                "tau_crack": best_val_cfg["tau_crack"],
                "score_damage_mean_iou": best_val_cfg["damage_mean_iou"]
            },
            "proposed_thresholds": {
                "tau_pothole": 0.30,
                "tau_crack": 0.35,
                "score_damage_mean_iou": proposed_entry["damage_mean_iou"]
            },
            "grid_results": grid_results
        },
        "test_evaluation": {
            "samples_evaluated": len(test_cached),
            "configurations": test_comparison_results,
        },
        "visual_comparisons": vis_records
    }

    report_path = EXP_DIR / "calibrated_threshold_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(experiment_report, f, indent=2)

    print("\n" + "=" * 75)
    print(f"EXPERIMENT COMPLETE! Report written to: {report_path}")
    print(f"Visualizations saved to: {VIS_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    run_experiment()
