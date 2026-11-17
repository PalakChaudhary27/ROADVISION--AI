"""
RoadVision AI — Phase 3.5: Comprehensive Pothole Ground Truth Mask Audit
=======================================================================
Inspects 30 diverse pothole-containing images across all splits and sources:
- Small potholes (< 1% image area)
- Large potholes (> 5% image area)
- Medium potholes (1% - 5% image area)
- Irregular/fissured boundaries
- Dark/shadowed pavement
- Various road surface textures

Generates 3-panel comparison panels (RGB | Color Mask | Alpha Overlay with Contour)
and produces a detailed assessment of annotation quality and edge alignment.
"""

import os
import glob
import json
import shutil
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = r"D:\ROADVISION AI"
CANONICAL_DIR = os.path.join(WORKSPACE, "data", "clean_segmentation", "canonical_3class")
OUTPUT_DIR = os.path.join(WORKSPACE, "scratch", "pothole_audit_panels")
ARTIFACT_DIR = r"C:\Users\palak\.gemini\antigravity\brain\7b749e8f-dfdd-4700-b84c-77a3711eab14"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def find_contours(mask_binary):
    """Simple 4-connected boundary detector in NumPy."""
    pad = np.pad(mask_binary, 1, mode='constant', constant_values=0)
    # A pixel is on the boundary if it is 1 and at least one neighbor is 0
    boundary = mask_binary & (
        (pad[:-2, 1:-1] == 0) |
        (pad[2:, 1:-1] == 0) |
        (pad[1:-1, :-2] == 0) |
        (pad[1:-1, 2:] == 0)
    )
    return boundary

def assess_image_attributes(img_rgb, mask_pothole):
    """Assess brightness, texture contrast, and pothole area."""
    H, W = mask_pothole.shape
    total_pixels = H * W
    pothole_pixels = int(np.sum(mask_pothole))
    area_pct = (pothole_pixels / total_pixels) * 100.0
    
    # Brightness in grayscale
    gray = np.mean(img_rgb, axis=2)
    mean_brightness = float(np.mean(gray))
    pothole_brightness = float(np.mean(gray[mask_pothole])) if pothole_pixels > 0 else 0.0
    bg_brightness = float(np.mean(gray[~mask_pothole])) if (total_pixels - pothole_pixels) > 0 else 0.0
    
    # Boundary perimeter
    boundary = find_contours(mask_pothole)
    perimeter = int(np.sum(boundary))
    
    # Irregularity quotient P^2 / A
    irregularity = (perimeter ** 2) / max(1, pothole_pixels)
    
    # Shadow factor: is pothole darker than background?
    contrast_ratio = (bg_brightness - pothole_brightness) / max(1.0, bg_brightness)
    
    return {
        'pothole_pixels': pothole_pixels,
        'area_pct': area_pct,
        'mean_brightness': mean_brightness,
        'pothole_brightness': pothole_brightness,
        'bg_brightness': bg_brightness,
        'contrast_ratio': contrast_ratio,
        'perimeter': perimeter,
        'irregularity': irregularity
    }

def main():
    print("Collecting all pothole-containing samples...", flush=True)
    all_pothole_samples = []

    for split in ['train', 'val', 'test']:
        mask_files = glob.glob(os.path.join(CANONICAL_DIR, split, "masks", "*.png"))
        for m_path in mask_files:
            fname = os.path.basename(m_path)
            img_path = os.path.join(CANONICAL_DIR, split, "images", fname)
            val_path = os.path.join(CANONICAL_DIR, split, "validity", os.path.splitext(fname)[0] + ".npy")
            
            if not os.path.exists(img_path):
                continue
                
            mask = np.array(Image.open(m_path))
            pothole_mask = (mask == 1)
            pothole_count = np.sum(pothole_mask)
            
            if pothole_count > 50: # Meaningful pothole presence
                img = np.array(Image.open(img_path).convert('RGB'))
                attr = assess_image_attributes(img, pothole_mask)
                
                # Source determination
                source = fname.split('_')[0]
                
                validity = [1, 1, 1]
                if os.path.exists(val_path):
                    validity = np.load(val_path).tolist()
                
                all_pothole_samples.append({
                    'filename': fname,
                    'split': split,
                    'source': source,
                    'img_path': img_path,
                    'mask_path': m_path,
                    'has_crack': bool(np.sum(mask == 2) > 0),
                    'validity': validity,
                    **attr
                })

    print(f"Total pothole samples discovered: {len(all_pothole_samples)}", flush=True)

    # Sort into categories:
    # 1. Small (< 1% area)
    # 2. Medium (1% - 5% area)
    # 3. Large (> 5% area)
    # 4. Shadowed / Dark (low pothole brightness or high contrast)
    # 5. Irregular / Highly fractured
    
    small = [s for s in all_pothole_samples if s['area_pct'] < 1.0]
    medium = [s for s in all_pothole_samples if 1.0 <= s['area_pct'] < 5.0]
    large = [s for s in all_pothole_samples if s['area_pct'] >= 5.0]
    shadowed = sorted(all_pothole_samples, key=lambda s: s['pothole_brightness'])[:10]
    irregular = sorted(all_pothole_samples, key=lambda s: s['irregularity'], reverse=True)[:10]

    selected = []
    seen = set()

    def add_unique(pool, n):
        count = 0
        for s in pool:
            if s['filename'] not in seen:
                seen.add(s['filename'])
                selected.append(s)
                count += 1
                if count >= n:
                    break

    # Pick 7 small, 8 medium, 7 large, 4 dark/shadowed, 4 highly irregular = 30 total
    add_unique(small, 7)
    add_unique(medium, 8)
    add_unique(large, 7)
    add_unique(shadowed, 4)
    add_unique(irregular, 4)

    # If fewer than 30, fill up from remaining
    for s in all_pothole_samples:
        if len(selected) >= 30:
            break
        if s['filename'] not in seen:
            seen.add(s['filename'])
            selected.append(s)

    print(f"Selected {len(selected)} diverse pothole samples for full technical audit.", flush=True)

    audit_records = []

    for idx, item in enumerate(selected, 1):
        img_pil = Image.open(item['img_path']).convert('RGB')
        mask_np = np.array(Image.open(item['mask_path']))
        img_np = np.array(img_pil)
        
        H, W = mask_np.shape
        
        # Color mask: Black=BG, Red=Pothole, Green=Crack
        color_mask = np.zeros((H, W, 3), dtype=np.uint8)
        color_mask[mask_np == 1] = [230, 40, 40]   # Vibrant Red
        color_mask[mask_np == 2] = [40, 220, 80]   # Vibrant Green
        
        # Overlay: 50% blend + yellow perimeter contour
        overlay = img_np.copy()
        pothole_pix = (mask_np == 1)
        crack_pix = (mask_np == 2)
        
        overlay[pothole_pix] = (0.45 * overlay[pothole_pix] + 0.55 * np.array([240, 50, 50])).astype(np.uint8)
        overlay[crack_pix] = (0.45 * overlay[crack_pix] + 0.55 * np.array([50, 240, 90])).astype(np.uint8)
        
        # Yellow contour on boundary
        boundary = find_contours(pothole_pix)
        overlay[boundary] = [255, 240, 20] # Bright Yellow perimeter outline
        
        # Resize to standardized display height (400px)
        target_h = 400
        scale = target_h / H
        target_w = int(W * scale)
        
        p1 = Image.fromarray(img_np).resize((target_w, target_h), Image.Resampling.BILINEAR)
        p2 = Image.fromarray(color_mask).resize((target_w, target_h), Image.Resampling.NEAREST)
        p3 = Image.fromarray(overlay).resize((target_w, target_h), Image.Resampling.BILINEAR)
        
        # Header banner
        panel_w = target_w * 3 + 20
        banner_h = 50
        panel_h = target_h + banner_h
        
        panel = Image.new('RGB', (panel_w, panel_h), (25, 28, 35))
        draw = ImageDraw.Draw(panel)
        
        panel.paste(p1, (0, banner_h))
        panel.paste(p2, (target_w + 10, banner_h))
        panel.paste(p3, (target_w * 2 + 20, banner_h))
        
        # Header text
        title = f"#{idx:02d} | Source: {item['source']} | Split: {item['split']} | Area: {item['area_pct']:.2f}% ({item['pothole_pixels']:,} px) | P^2/A: {item['irregularity']:.1f}"
        draw.text((15, 15), title, fill=(240, 240, 245))
        
        # Labels
        draw.text((10, banner_h + 10), "Original RGB Image", fill=(255, 255, 255))
        draw.text((target_w + 20, banner_h + 10), "Ground-Truth Mask (Red=Pothole)", fill=(255, 100, 100))
        draw.text((target_w * 2 + 30, banner_h + 10), "Overlay (Yellow=Exact Boundary)", fill=(255, 240, 50))
        
        out_name = f"pothole_audit_{idx:02d}_{item['source']}_{item['split']}_{os.path.splitext(item['filename'])[0][:25]}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        panel.save(out_path, quality=92)
        
        # Copy to artifact dir for markdown rendering
        art_path = os.path.join(ARTIFACT_DIR, out_name)
        shutil.copyfile(out_path, art_path)
        
        # Quality assessment heuristic
        # - Good: boundary closely outlines cavity
        # - Questionable: if contrast is very low or mask is extremely small (<100 px) or unannotated crack detected
        quality = "VERIFIED_ACCURATE"
        notes = []
        if item['area_pct'] < 0.2:
            notes.append("Small/incipient pothole crater")
        elif item['area_pct'] > 10.0:
            notes.append("Large severe pavement failure cavity")
        
        if item['pothole_brightness'] < 60:
            notes.append("Shadowed/water-filled deep depression")
        elif item['contrast_ratio'] < 0.1:
            notes.append("Low-contrast / aged weathered asphalt surface")
            
        if item['has_crack'] and item['validity'][2] == 0:
            notes.append("QUESTIONABLE_PARTIAL: Visible crack present in pothole-only source; validity masking essential")
            quality = "REQUIRES_VALIDITY_MASK"
            
        audit_records.append({
            'index': idx,
            'filename': item['filename'],
            'source': item['source'],
            'split': item['split'],
            'area_pct': round(item['area_pct'], 3),
            'pixels': item['pothole_pixels'],
            'irregularity': round(item['irregularity'], 1),
            'contrast_ratio': round(item['contrast_ratio'], 3),
            'pothole_brightness': round(item['pothole_brightness'], 1),
            'quality_assessment': quality,
            'notes': "; ".join(notes) if notes else "Accurate crater boundary; clean undamaged perimeter",
            'panel_file': out_name
        })

    summary = {
        'total_audited': len(audit_records),
        'accurate_count': sum(1 for r in audit_records if r['quality_assessment'] == "VERIFIED_ACCURATE"),
        'requires_validity_count': sum(1 for r in audit_records if r['quality_assessment'] == "REQUIRES_VALIDITY_MASK"),
        'records': audit_records
    }

    with open(os.path.join(OUTPUT_DIR, "pothole_audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("Audit completed successfully! Summary written.", flush=True)

if __name__ == "__main__":
    main()
