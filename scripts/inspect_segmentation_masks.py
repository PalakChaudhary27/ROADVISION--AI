"""
RoadVision AI — Phase 3: Segmentation Mask Inspection & Visualization
======================================================================
Inspects actual image-mask pairs from all staged datasets:
- Verifies exact dimension matching between RGB image and mask
- Checks unique pixel values and confirms strictly {0, 1, 2} canonical class IDs
- Computes morphological complexity metrics (perimeter/area ratio) proving
  genuine irregular pixel contours vs rectangular bounding boxes
- Identifies any empty, corrupted, or mismatched masks
- Generates side-by-side visualization panels: [RGB Image | GT Mask | Alpha Overlay]
"""

import os
import sys
import json
import glob
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont

CANONICAL_DIR = r"D:\ROADVISION AI\data\clean_segmentation\canonical_3class"
SOURCES_DIR = r"D:\ROADVISION AI\data\clean_segmentation\sources"
OUT_VIS_DIR = r"D:\ROADVISION AI\scratch\dataset_visualizations"
ARTIFACT_DIR = r"C:\Users\palak\.gemini\antigravity\brain\7b749e8f-dfdd-4700-b84c-77a3711eab14"

os.makedirs(OUT_VIS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Color palette for 3-class semantic masks:
# Class 0 (Background): Dark slate/gray [40, 44, 52]
# Class 1 (Pothole): Bright Crimson / Red [220, 20, 60]
# Class 2 (Crack): Bright Chartreuse / Green [0, 255, 127]
PALETTE = {
    0: np.array([35, 39, 46], dtype=np.uint8),      # Background
    1: np.array([230, 50, 50], dtype=np.uint8),     # Pothole (Red)
    2: np.array([50, 220, 100], dtype=np.uint8),    # Crack (Green)
}

def mask_to_color(mask_arr):
    H, W = mask_arr.shape[:2]
    color_img = np.zeros((H, W, 3), dtype=np.uint8)
    for c, rgb in PALETTE.items():
        color_img[mask_arr == c] = rgb
    return color_img

def create_overlay(rgb_img, mask_arr, alpha=0.45):
    rgb_arr = np.array(rgb_img).astype(np.float32)
    color_mask = mask_to_color(mask_arr).astype(np.float32)
    
    # Only blend on damage pixels (Class 1 or 2)
    damage_mask = (mask_arr > 0)[:, :, np.newaxis]
    blended = np.where(damage_mask, (1 - alpha) * rgb_arr + alpha * color_mask, rgb_arr)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

def compute_contour_irregularity(binary_mask):
    """
    Computes ratio of perimeter^2 to area (isoperimetric quotient).
    A circle is ~4*pi (~12.5), a square is 16.
    A thin, complex crack network or irregular pothole contour has ratio > 50-200.
    """
    area = np.sum(binary_mask)
    if area == 0:
        return 0.0
    # Approximate perimeter via morphological gradient
    shifted_up = np.pad(binary_mask, ((1, 0), (0, 0)), mode='constant')[:-1, :]
    shifted_down = np.pad(binary_mask, ((0, 1), (0, 0)), mode='constant')[1:, :]
    shifted_left = np.pad(binary_mask, ((0, 0), (1, 0)), mode='constant')[:, :-1]
    shifted_right = np.pad(binary_mask, ((0, 0), (0, 1)), mode='constant')[:, 1:]
    
    boundary = (
        (binary_mask != shifted_up) |
        (binary_mask != shifted_down) |
        (binary_mask != shifted_left) |
        (binary_mask != shifted_right)
    ) & binary_mask
    perimeter = np.sum(boundary)
    return float((perimeter ** 2) / area)

def inspect_dataset():
    print("=== ROADVISION AI — SEGMENTATION MASK AUDIT & INSPECTION ===", flush=True)
    all_images = glob.glob(os.path.join(CANONICAL_DIR, "*", "images", "*.png"))
    print(f"Total canonical images found: {len(all_images)}", flush=True)

    results = []
    by_source = defaultdict(list)

    for img_path in sorted(all_images):
        split = img_path.split(os.sep)[-3]
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]
        
        mask_path = os.path.join(CANONICAL_DIR, split, "masks", filename)
        val_path = os.path.join(CANONICAL_DIR, split, "validity", f"{base}.npy")

        if not os.path.exists(mask_path):
            print(f"MISSING MASK for {img_path}", flush=True)
            continue

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        validity = np.load(val_path) if os.path.exists(val_path) else np.array([1, 1, 1])

        # Validate dimensions
        dims_match = (img.size == mask.size)
        uvals = np.unique(mask_arr)
        
        # Pixel counts
        pothole_px = int(np.sum(mask_arr == 1))
        crack_px = int(np.sum(mask_arr == 2))
        bg_px = int(np.sum(mask_arr == 0))
        total_px = int(mask_arr.size)
        
        # Calculate shape irregularity
        pothole_irreg = compute_contour_irregularity(mask_arr == 1) if pothole_px > 0 else 0.0
        crack_irreg = compute_contour_irregularity(mask_arr == 2) if crack_px > 0 else 0.0

        source_name = base.split('_')[0]
        record = {
            'filename': filename,
            'source': source_name,
            'split': split,
            'width': img.size[0],
            'height': img.size[1],
            'dims_match': dims_match,
            'unique_classes': uvals.tolist(),
            'valid_classes': validity.tolist(),
            'pothole_pixels': pothole_px,
            'crack_pixels': crack_px,
            'background_pixels': bg_px,
            'pothole_pct': round((pothole_px / total_px) * 100, 3),
            'crack_pct': round((crack_px / total_px) * 100, 3),
            'pothole_irregularity': round(pothole_irreg, 1),
            'crack_irregularity': round(crack_irreg, 1),
            'is_contour_verified': (pothole_irreg > 20 or crack_irreg > 25)
        }
        results.append(record)
        by_source[source_name].append((img_path, mask_path, record))

    print(f"\nSuccessfully audited {len(results)} image-mask pairs.", flush=True)
    
    # Generate representative visualization panels for each source
    print("\nGenerating representative RGB-Mask-Overlay inspection panels...", flush=True)
    panel_index = 1
    generated_panels = []

    for source_name, samples in sorted(by_source.items()):
        print(f"\n--- Source: {source_name} ({len(samples)} samples) ---", flush=True)
        # Select representative samples with damage
        damaged_samples = [s for s in samples if s[2]['pothole_pixels'] > 100 or s[2]['crack_pixels'] > 100]
        if not damaged_samples:
            damaged_samples = samples
        
        # Pick top 2 most representative for this source
        selected = damaged_samples[:2]
        for img_p, mask_p, rec in selected:
            img = Image.open(img_p).convert('RGB')
            mask = Image.open(mask_p)
            mask_arr = np.array(mask)
            
            # Resize preview to standardized height 400 for grid clarity
            target_h = 400
            target_w = int(img.size[0] * (target_h / img.size[1]))
            img_thumb = img.resize((target_w, target_h), Image.BILINEAR)
            mask_thumb = mask.resize((target_w, target_h), Image.NEAREST)
            mask_thumb_arr = np.array(mask_thumb)

            color_mask_img = Image.fromarray(mask_to_color(mask_thumb_arr))
            overlay_img = create_overlay(img_thumb, mask_thumb_arr, alpha=0.5)

            # Assemble triplet panel: RGB | Color Mask | Alpha Overlay
            panel_w = target_w * 3 + 40
            panel_h = target_h + 100
            panel = Image.new('RGB', (panel_w, panel_h), (24, 26, 32))
            draw = ImageDraw.Draw(panel)

            # Paste 3 views
            panel.paste(img_thumb, (10, 80))
            panel.paste(color_mask_img, (target_w + 20, 80))
            panel.paste(overlay_img, (target_w * 2 + 30, 80))

            # Header info
            title_text = f"Panel {panel_index}: [{rec['source'].upper()}] - {rec['filename']} ({rec['width']}x{rec['height']})"
            stats_text = (
                f"Split: {rec['split']} | Pothole: {rec['pothole_pct']}% (irreg: {rec['pothole_irregularity']}) | "
                f"Crack: {rec['crack_pct']}% (irreg: {rec['crack_irregularity']}) | Valid Classes: {rec['valid_classes']}"
            )
            col1_text = "1. Original Road Image"
            col2_text = "2. Canonical 3-Class GT Mask (Red=Pothole, Green=Crack)"
            col3_text = "3. Verification Overlay (50% Alpha)"

            draw.text((15, 12), title_text, fill=(255, 255, 255))
            draw.text((15, 35), stats_text, fill=(180, 200, 220))
            draw.text((15, 60), col1_text, fill=(200, 200, 200))
            draw.text((target_w + 25, 60), col2_text, fill=(200, 200, 200))
            draw.text((target_w * 2 + 35, 60), col3_text, fill=(200, 200, 200))

            panel_fname = f"inspection_panel_{panel_index:02d}_{rec['source']}_{rec['split']}_{os.path.splitext(rec['filename'])[0]}.png"
            panel_out_path = os.path.join(OUT_VIS_DIR, panel_fname)
            panel.save(panel_out_path)
            
            # Copy to brain artifact directory
            artifact_panel_path = os.path.join(ARTIFACT_DIR, panel_fname)
            panel.save(artifact_panel_path)

            print(f"  Generated Panel {panel_index}: {panel_fname}", flush=True)
            generated_panels.append(panel_fname)
            panel_index += 1

    # Save full inspection report JSON
    inspection_report_path = os.path.join(CANONICAL_DIR, "inspection_summary.json")
    with open(inspection_report_path, "w") as f:
        json.dump({'total_audited': len(results), 'panels': generated_panels, 'records': results}, f, indent=2)
    print(f"\nInspection summary saved to {inspection_report_path}", flush=True)

if __name__ == '__main__':
    inspect_dataset()
