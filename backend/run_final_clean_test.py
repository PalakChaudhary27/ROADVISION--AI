"""
RoadVision AI - Final Clean Test Evaluation Suite
Evaluates ONLY the 17 independent held-out test images identified in the verification report.
Excludes the 4 UI reference images completely.
Compares Baseline Argmax, Proposed (0.30, 0.35), Balanced-Mid (0.35, 0.35), and High-Recall (0.25, 0.25).
"""

import os
import io
import sys
import json
import time
import shutil
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
OUT_DIR = PROJECT_ROOT / "audit_results" / "threshold_experiment" / "final_clean_test"
VIS_DIR = OUT_DIR / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)

CLASS_MAP = {
    0: {"name": "background", "color": (0, 0, 0), "label": "Background"},
    1: {"name": "pothole", "color": (255, 59, 48), "label": "Pothole"},
    2: {"name": "crack", "color": (255, 204, 0), "label": "Crack"},
}
DATASET_TO_MODEL = {0: 1, 1: 2, 2: 0}

CLEAN_17_STEMS = [
    "20250223_104743",
    "vlcsnap-2025-02-19-14h51m35s903",
    "vlcsnap-2025-02-19-15h06m52s866",
    "vlcsnap-2025-02-19-14h25m44s752",
    "vlcsnap-2025-02-18-18h37m07s201",
    "vlcsnap-2025-02-18-18h32m27s822",
    "vlcsnap-2025-02-18-18h48m05s740",
    "vlcsnap-2025-02-18-18h31m35s101",
    "vlcsnap-00019",
    "20250219_164848",
    "vlcsnap-2025-02-19-14h14m20s512",
    "vlcsnap-2025-02-26-20h34m21s541",
    "vlcsnap-00037",
    "vlcsnap-2025-02-19-14h30m17s564",
    "vlcsnap-2025-02-19-15h05m39s456",
    "vlcsnap-2025-02-18-23h15m45s809",
    "vlcsnap-2025-02-18-17h10m30s786"
]

AUDIT_24_EXPOSURE_STEMS = {
    "20250223_104743",
    "vlcsnap-2025-02-18-18h37m07s201",
    "vlcsnap-2025-02-18-18h32m27s822",
    "vlcsnap-2025-02-18-18h48m05s740",
    "vlcsnap-2025-02-18-18h31m35s101",
    "vlcsnap-00019",
    "20250219_164848",
    "vlcsnap-2025-02-19-14h14m20s512",
    "vlcsnap-2025-02-26-20h34m21s541",
    "vlcsnap-00037",
    "vlcsnap-2025-02-19-14h30m17s564",
    "vlcsnap-2025-02-19-15h05m39s456"
}


def decode_ground_truth_mask(anno_path: Path, width: int, height: int) -> np.ndarray:
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
    h, w = gt_mask.shape
    err_rgb = np.full((h, w, 3), 40, dtype=np.uint8)  # TN dark gray

    gt_dmg = gt_mask > 0
    pred_dmg = pred_mask > 0

    tp_exact = (gt_mask == pred_mask) & gt_dmg
    err_rgb[tp_exact] = [0, 230, 80]  # Green: TP

    confusion = gt_dmg & pred_dmg & (gt_mask != pred_mask)
    err_rgb[confusion] = [255, 200, 0]  # Yellow: Class confusion

    fn = gt_dmg & (~pred_dmg)
    err_rgb[fn] = [255, 30, 30]  # Red: Missed damage

    fp = (~gt_dmg) & pred_dmg
    err_rgb[fp] = [0, 150, 255]  # Blue/Cyan: False positive

    return err_rgb


def assign_classes(probs: np.ndarray, tau_pothole: float, tau_crack: float) -> np.ndarray:
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

        prec = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 1.0

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

    dmg_iou = float(np.mean([class_res["pothole"]["iou"], class_res["crack"]["iou"]]))
    dmg_dice = float(np.mean([class_res["pothole"]["dice"], class_res["crack"]["dice"]]))
    dmg_recall = float(np.mean([class_res["pothole"]["recall"], class_res["crack"]["recall"]]))
    dmg_prec = float(np.mean([class_res["pothole"]["precision"], class_res["crack"]["precision"]]))

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
    pred_mid: np.ndarray,
    sample_id: str,
    meta_info: Dict[str, Any]
) -> Image.Image:
    w, h = orig_img.size
    panel_w = 380
    panel_h = int(h * (panel_w / w))

    orig_res = orig_img.resize((panel_w, panel_h), Image.Resampling.BILINEAR)

    # 1. Ground Truth
    gt_pil = Image.fromarray(gt_mask).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    gt_rgb = mask_to_rgb(np.array(gt_pil))

    # 2. Baseline Argmax
    arm_pil = Image.fromarray(pred_argmax).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    arm_rgb = mask_to_rgb(np.array(arm_pil))

    # 3. Balanced-Mid Prediction
    mid_pil = Image.fromarray(pred_mid).resize((panel_w, panel_h), Image.Resampling.NEAREST)
    mid_rgb = mask_to_rgb(np.array(mid_pil))

    # 4. Argmax Error Map
    arm_err = build_error_map(np.array(gt_pil), np.array(arm_pil))

    # 5. Balanced-Mid Error Map
    mid_err = build_error_map(np.array(gt_pil), np.array(mid_pil))

    header_h = 55
    total_w = panel_w * 5 + 24
    total_h = panel_h + header_h + 30

    canvas = Image.new("RGB", (total_w, total_h), (24, 28, 36))
    draw = ImageDraw.Draw(canvas)

    panels = [orig_res, Image.fromarray(gt_rgb), Image.fromarray(arm_rgb), Image.fromarray(mid_rgb), Image.fromarray(mid_err)]
    titles = [
        "1. Original Road Image",
        "2. Ground Truth Mask",
        "3. Baseline Argmax",
        "4. Balanced-Mid (0.35/0.35)",
        "5. Diagnostic Error Map (Balanced-Mid)"
    ]
    subtitles = [
        f"Sample: {sample_id}",
        f"GT Dmg: {meta_info.get('gt_dmg_pct', 0):.2f}%",
        f"Argmax Dmg: {meta_info.get('arm_dmg_pct', 0):.2f}%",
        f"Mid Dmg: {meta_info.get('mid_dmg_pct', 0):.2f}%",
        f"FN Missed: {meta_info.get('mid_fn', 0):,} px (Argmax: {meta_info.get('arm_fn', 0):,})"
    ]

    for idx, (p, title, sub) in enumerate(zip(panels, titles, subtitles)):
        x_off = 12 + idx * (panel_w + 2)
        y_off = header_h

        draw.text((x_off + 4, 10), title, fill=(255, 255, 255))
        draw.text((x_off + 4, 30), sub, fill=(180, 195, 215))

        canvas.paste(p, (x_off, y_off))
        draw.rectangle([x_off, y_off, x_off + panel_w, y_off + panel_h], outline=(60, 70, 90), width=1)

    draw.text(
        (12, total_h - 22),
        f"Legend: Green = True Positive | Red = Missed Damage (FN) | Yellow = Class Confusion | Blue = False Positive",
        fill=(150, 165, 185)
    )

    return canvas


def run_evaluation():
    print("=" * 75)
    print("ROADVISION AI: FINAL CLEAN TEST EVALUATION (17 INDEPENDENT TEST SAMPLES)")
    print("=" * 75)

    # 1. Load model once
    t0 = time.time()
    weights_path = ARTIFACTS_DIR / "best_model.pth"
    model = deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=3, aux_loss=True)
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s.")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def predict_probs(image_pil: Image.Image) -> np.ndarray:
        w_orig, h_orig = image_pil.size
        img_512 = image_pil.resize((512, 512), Image.Resampling.BILINEAR)
        arr = np.array(img_512, dtype=np.float32) / 255.0
        norm = (arr - mean) / std
        tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()

        with torch.no_grad():
            out = model(tensor)["out"]
            probs_512 = torch.softmax(out, dim=1).squeeze(0).cpu().numpy()

        probs_orig = np.zeros((3, h_orig, w_orig), dtype=np.float32)
        for c in range(3):
            p_c = Image.fromarray(probs_512[c]).resize((w_orig, h_orig), Image.Resampling.BILINEAR)
            probs_orig[c] = np.array(p_c)

        probs_sum = probs_orig.sum(axis=0, keepdims=True)
        probs_orig = probs_orig / np.maximum(probs_sum, 1e-7)
        return probs_orig

    # 2. Verify and Load the 17 clean test images
    images_dir = DATA_DIR / "images"
    labels_dir = DATA_DIR / "labels"

    samples_pool = []
    print("\nVerifying 17 independent test samples:")
    for sname in CLEAN_17_STEMS:
        img_p = images_dir / f"{sname}.jpg"
        lbl_p = labels_dir / f"{sname}.txt"
        assert img_p.exists(), f"Image not found: {img_p}"
        assert lbl_p.exists(), f"Label not found: {lbl_p}"

        was_in_audit_24 = sname in AUDIT_24_EXPOSURE_STEMS
        samples_pool.append({
            "stem": sname,
            "filename": f"{sname}.jpg",
            "image_path": img_p,
            "anno_path": lbl_p,
            "was_in_audit_24": was_in_audit_24
        })
        print(f"  {sname}.jpg | Exists: True | In Earlier Diagnostic Audit: {was_in_audit_24}")

    # 3. Cache probabilities and ground-truth masks for each
    print("\n--- Running inference and caching predictions ---")
    cached_data = []
    for idx, s in enumerate(samples_pool):
        img_pil = Image.open(s["image_path"]).convert("RGB")
        w, h = img_pil.size
        gt_mask = decode_ground_truth_mask(s["anno_path"], w, h)
        probs = predict_probs(img_pil)

        cached_data.append({
            **s,
            "gt_mask": gt_mask,
            "probs": probs,
            "width": w,
            "height": h
        })
        print(f"  [{idx+1}/17] Processed {s['filename']} (GT Damage: {np.mean(gt_mask>0)*100:.2f}%)")

    # 4. Evaluate the 4 target configurations
    configs = [
        ("Baseline Argmax", None, None),
        ("Proposed (0.30, 0.35)", 0.30, 0.35),
        ("Balanced-Mid (0.35, 0.35)", 0.35, 0.35),
        ("High-Recall (0.25, 0.25)", 0.25, 0.25),
    ]

    gt_all_17 = [c["gt_mask"] for c in cached_data]
    preds_by_cfg_all = {}

    for name, tau_p, tau_c in configs:
        if tau_p is None:
            preds = [assign_argmax(c["probs"]) for c in cached_data]
        else:
            preds = [assign_classes(c["probs"], tau_p, tau_c) for c in cached_data]
        preds_by_cfg_all[name] = preds

    # Also extract subset of 5 "strictly unaudited" images
    idx_5_unaudited = [i for i, c in enumerate(cached_data) if not c["was_in_audit_24"]]
    gt_subset_5 = [gt_all_17[i] for i in idx_5_unaudited]

    # Compute aggregate metrics for all 17
    aggregate_results_17 = {}
    for name in preds_by_cfg_all:
        m = compute_metrics(gt_all_17, preds_by_cfg_all[name])
        aggregate_results_17[name] = m

    # Compute aggregate metrics for the 5 unaudited subset
    aggregate_results_5 = {}
    for name in preds_by_cfg_all:
        p_sub = [preds_by_cfg_all[name][i] for i in idx_5_unaudited]
        m = compute_metrics(gt_subset_5, p_sub)
        aggregate_results_5[name] = m

    # Per-image metric calculation
    per_image_results = []
    for idx, c in enumerate(cached_data):
        stem = c["stem"]
        gt_single = [c["gt_mask"]]
        rec = {
            "index": idx + 1,
            "filename": c["filename"],
            "stem": stem,
            "was_in_earlier_audit_24": c["was_in_audit_24"],
            "gt_damage_pct": round(float(np.mean(c["gt_mask"] > 0) * 100), 2),
            "configs": {}
        }
        for name in preds_by_cfg_all:
            pr_single = [preds_by_cfg_all[name][idx]]
            m_single = compute_metrics(gt_single, pr_single)
            rec["configs"][name] = {
                "pred_damage_pct": round(float(np.mean(pr_single[0] > 0) * 100), 2),
                "damage_iou": m_single["damage_mean_iou"],
                "pothole_iou": m_single["classes"]["pothole"]["iou"],
                "pothole_rec": m_single["classes"]["pothole"]["recall"],
                "crack_iou": m_single["classes"]["crack"]["iou"],
                "crack_rec": m_single["classes"]["crack"]["recall"],
                "fn_pixels": int(np.sum((c["gt_mask"] > 0) & (pr_single[0] == 0))),
                "fp_pixels": int(np.sum((c["gt_mask"] == 0) & (pr_single[0] > 0))),
            }
        per_image_results.append(rec)

    # 5. Generate Visual Comparisons
    print("\n--- Generating Visual Comparisons for Representative Images ---")
    arm_preds = preds_by_cfg_all["Baseline Argmax"]
    mid_preds = preds_by_cfg_all["Balanced-Mid (0.35, 0.35)"]

    vis_records = []
    # Pick 6 diverse representative test scenes:
    rep_indices = [0, 1, 2, 4, 9, 13]  # covers severe damage, subtle crack, potholes, clean pavement
    for local_idx, idx in enumerate(rep_indices):
        item = cached_data[idx]
        stem = item["stem"]
        img_pil = Image.open(item["image_path"]).convert("RGB")
        gt = item["gt_mask"]
        p_arm = arm_preds[idx]
        p_mid = mid_preds[idx]

        gt_dmg = np.mean(gt > 0) * 100
        arm_dmg = np.mean(p_arm > 0) * 100
        mid_dmg = np.mean(p_mid > 0) * 100

        arm_fn = int(np.sum((gt > 0) & (p_arm == 0)))
        mid_fn = int(np.sum((gt > 0) & (p_mid == 0)))

        meta = {
            "gt_dmg_pct": gt_dmg,
            "arm_dmg_pct": arm_dmg,
            "mid_dmg_pct": mid_dmg,
            "arm_fn": arm_fn,
            "mid_fn": mid_fn,
        }

        panel = create_comparison_panel(img_pil, gt, p_arm, p_mid, stem, meta)
        vis_filename = f"clean_test_{local_idx+1:02d}_{stem}.png"
        vis_path = VIS_DIR / vis_filename
        panel.save(vis_path, quality=92)

        vis_records.append({
            "sample_index": local_idx + 1,
            "filename": item["filename"],
            "stem": stem,
            "vis_filename": vis_filename,
            "gt_damage_pct": round(gt_dmg, 2),
            "argmax_fn": arm_fn,
            "mid_fn": mid_fn,
            "fn_reduction_pct": round(((arm_fn - mid_fn) / max(arm_fn, 1)) * 100, 2) if arm_fn > 0 else 0
        })
        print(f"  Visual {local_idx+1}/6 generated: {item['filename']} (FN: {arm_fn:,} -> {mid_fn:,} px, {vis_records[-1]['fn_reduction_pct']}%)")

    # 6. Save complete JSON report
    final_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_images_count": len(CLEAN_17_STEMS),
        "test_images": [s["filename"] for s in samples_pool],
        "strictly_unaudited_subset_count": len(idx_5_unaudited),
        "strictly_unaudited_subset_files": [cached_data[i]["filename"] for i in idx_5_unaudited],
        "aggregate_metrics_all_17": aggregate_results_17,
        "aggregate_metrics_subset_5": aggregate_results_5,
        "per_image_results": per_image_results,
        "visual_comparisons": vis_records
    }

    json_path = OUT_DIR / "final_clean_test_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nResults successfully written to: {json_path}")


if __name__ == "__main__":
    run_evaluation()
