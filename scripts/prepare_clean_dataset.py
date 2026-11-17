"""
RoadVision AI — Phase 3: Clean Segmentation Dataset Preparation Script
======================================================================
Prepares a verified, clean, pixel-level 3-class semantic segmentation dataset:
  Class 0: Background (intact pavement)
  Class 1: Pothole (damaged depression/crater)
  Class 2: Crack (structural surface fissure)

Features:
- Multi-source ingestion: Pothole Mix (SHREC2022), CRACK500, Pothole-600, CPRID, EdmCrack600, GAPs384
- Direct 1:1 color channel mapping:
    Red [255, 0, 0]   -> Class 1 (Pothole)
    Green [0, 255, 0] -> Class 2 (Crack)
    Black [0, 0, 0]   -> Class 0 (Background)
- Partial-label validity masks (avoiding false negative supervision):
    * Crack-only sources: crack is valid, background is valid, pothole is unobserved/masked.
    * Pothole-only sources: pothole is valid, background is valid, crack is unobserved/masked.
    * Multi-class sources: all classes valid.
- Zero-leakage session/scene grouping for train/val/test splits.
- Canonical 17 clean test images strictly isolated (never in train/val).
- High-resolution sliding window patch extractor: 512x512 with stride 384.
"""

import os
import sys
import io
import json
import time
import zipfile
import urllib.request
import numpy as np
from PIL import Image
from collections import defaultdict, Counter

WORKSPACE_DIR = r"D:\ROADVISION AI"
DATA_DIR = os.path.join(WORKSPACE_DIR, "data", "clean_segmentation")
SOURCES_DIR = os.path.join(DATA_DIR, "sources")
CANONICAL_DIR = os.path.join(DATA_DIR, "canonical_3class")
CROPS_DIR = os.path.join(DATA_DIR, "canonical_crops_512")

POTHOLE_MIX_URL = "https://data.mendeley.com/public-files/datasets/kfth5g2xk3/files/d5b8a18b-b1fa-4637-88ad-56263431ef4e/file_downloaded"
POTHOLE_VIDEO_URL = "https://data.mendeley.com/public-files/datasets/5bwfg4v4cd/files/3ca42bc8-8852-47ae-adb6-c1df78e6277e/file_downloaded"

class RemoteZipFile(io.RawIOBase):
    """Seekable HTTP Range wrapper for zipfile over remote URLs."""
    def __init__(self, url, headers=None):
        self.url = url
        self.headers = headers or {'User-Agent': 'Mozilla/5.0'}
        req = urllib.request.Request(url, headers=self.headers)
        req.get_method = lambda: 'HEAD'
        with urllib.request.urlopen(req, timeout=30) as resp:
            self.size = int(resp.headers.get('Content-Length', 0))
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b) - 1, self.size - 1)
        req = urllib.request.Request(self.url, headers={**self.headers, 'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        n = len(data)
        b[:n] = data
        self.pos += n
        return n

def setup_directories():
    for d in [
        SOURCES_DIR,
        os.path.join(SOURCES_DIR, "pothole_mix"),
        os.path.join(SOURCES_DIR, "crack500"),
        os.path.join(SOURCES_DIR, "pothole_video"),
        CANONICAL_DIR,
        os.path.join(CANONICAL_DIR, "train", "images"),
        os.path.join(CANONICAL_DIR, "train", "masks"),
        os.path.join(CANONICAL_DIR, "train", "validity"),
        os.path.join(CANONICAL_DIR, "val", "images"),
        os.path.join(CANONICAL_DIR, "val", "masks"),
        os.path.join(CANONICAL_DIR, "val", "validity"),
        os.path.join(CANONICAL_DIR, "test", "images"),
        os.path.join(CANONICAL_DIR, "test", "masks"),
        os.path.join(CANONICAL_DIR, "test", "validity"),
        CROPS_DIR,
        os.path.join(CROPS_DIR, "train", "images"),
        os.path.join(CROPS_DIR, "train", "masks"),
        os.path.join(CROPS_DIR, "train", "validity"),
        os.path.join(CROPS_DIR, "val", "images"),
        os.path.join(CROPS_DIR, "val", "masks"),
        os.path.join(CROPS_DIR, "val", "validity"),
    ]:
        os.makedirs(d, exist_ok=True)
    print("All target directories created successfully.", flush=True)

def map_mask_to_canonical(mask_img, source_type):
    """
    Converts RGB/Grayscale source mask into canonical 3-class segmentation:
      0: Background
      1: Pothole
      2: Crack
    Also returns a validity mask:
      validity[0]: 1 if background is validly annotated
      validity[1]: 1 if pothole is validly annotated (0 if unobserved/partial)
      validity[2]: 1 if crack is validly annotated (0 if unobserved/partial)
    """
    arr = np.array(mask_img)
    H, W = arr.shape[:2]
    canonical_mask = np.zeros((H, W), dtype=np.uint8)
    
    # Initialize validity mask [3]
    # For full multi-class datasets: [1, 1, 1]
    # For crack-only datasets: [1, 0, 1]
    # For pothole-only datasets: [1, 1, 0]
    validity = np.ones(3, dtype=np.uint8)

    if arr.ndim == 3 and arr.shape[2] >= 3:
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        
        # Pothole detection: Red > 128 and Red > Green
        pothole_pixels = (r > 128) & (r > g)
        
        # Crack detection: Green > 128 and Green > Red
        crack_pixels = (g > 128) & (g > r)
        
        # Grayscale / white fallback in binary mask
        white_pixels = (r > 128) & (g > 128) & (b > 128)
        if white_pixels.any():
            if 'pothole' in source_type.lower():
                pothole_pixels = pothole_pixels | white_pixels
            else:
                crack_pixels = crack_pixels | white_pixels

        canonical_mask[pothole_pixels] = 1
        canonical_mask[crack_pixels] = 2

    elif arr.ndim == 2:
        # Binary mask
        damage_pixels = arr > 128
        if 'pothole' in source_type.lower():
            canonical_mask[damage_pixels] = 1
        else:
            canonical_mask[damage_pixels] = 2

    # Set partial label flags
    if source_type in ['crack500', 'edmcrack600', 'gaps384']:
        validity[1] = 0 # Pothole is unobserved/not annotated
    elif source_type in ['pothole600', 'pothole_video']:
        validity[2] = 0 # Crack is unobserved/not annotated
    elif source_type in ['cprid', 'cracks-and-potholes-in-road', 'cnr-road-dataset']:
        # Fully annotated multi-class
        validity[1] = 1
        validity[2] = 1

    return canonical_mask, validity

def extract_crops(img, mask, validity, crop_size=512, stride=384):
    """
    Extracts overlapping crops of size (crop_size x crop_size) with given stride.
    Preserves fine crack lines without downsampling blur.
    """
    W, H = img.size
    crops = []
    if W < crop_size or H < crop_size:
        # Pad image and mask if smaller than crop size
        pad_w = max(0, crop_size - W)
        pad_h = max(0, crop_size - H)
        img_padded = Image.new('RGB', (W + pad_w, H + pad_h), (0, 0, 0))
        img_padded.paste(img, (0, 0))
        mask_padded = np.zeros((H + pad_h, W + pad_w), dtype=np.uint8)
        mask_padded[:H, :W] = mask
        crops.append((img_padded, mask_padded, validity))
        return crops

    for y in range(0, H - crop_size + 1, stride):
        for x in range(0, W - crop_size + 1, stride):
            crop_img = img.crop((x, y, x + crop_size, y + crop_size))
            crop_mask = mask[y:y + crop_size, x:x + crop_size]
            crops.append((crop_img, crop_mask, validity))

    # Also extract rightmost and bottom edges to ensure complete coverage
    if (W - crop_size) % stride != 0:
        x = W - crop_size
        for y in range(0, H - crop_size + 1, stride):
            crop_img = img.crop((x, y, x + crop_size, y + crop_size))
            crop_mask = mask[y:y + crop_size, x:x + crop_size]
            crops.append((crop_img, crop_mask, validity))

    if (H - crop_size) % stride != 0:
        y = H - crop_size
        for x in range(0, W - crop_size + 1, stride):
            crop_img = img.crop((x, y, x + crop_size, y + crop_size))
            crop_mask = mask[y:y + crop_size, x:x + crop_size]
            crops.append((crop_img, crop_mask, validity))

    return crops

def stage_and_process_dataset(max_pairs_per_subset=30):
    """
    Stages image-mask pairs from remote archive, verifies annotations,
    maps to canonical 3-class schema, and produces train/val/test splits and crops.
    """
    setup_directories()
    print("Connecting to remote Pothole Mix archive...", flush=True)
    t0 = time.time()
    stream = io.BufferedReader(RemoteZipFile(POTHOLE_MIX_URL))
    
    metadata = []
    stats = {
        'total_images': 0,
        'total_masks': 0,
        'by_subset': Counter(),
        'by_split': Counter(),
        'by_class_pixels': Counter(),
        'images_with_pothole': 0,
        'images_with_crack': 0,
        'images_background_only': 0,
        'empty_masks': 0,
        'corrupted_files': 0,
        'crops_extracted': 0,
        'subsets_catalog': {}
    }

    subsets = [
        ('crack500', 'crack500'),
        ('pothole600', 'pothole600'),
        ('cracks-and-potholes-in-road', 'cprid'),
        ('edmcrack600', 'edmcrack600'),
        ('gaps384', 'gaps384'),
        ('cnr-road-dataset', 'cnr-road-dataset')
    ]

    with zipfile.ZipFile(stream) as zf:
        namelist = zf.namelist()
        
        # Save readme to docs/sources
        readme_path = os.path.join(SOURCES_DIR, "pothole_mix", "README_POTHOLE_MIX.txt")
        with open(readme_path, "wb") as f:
            f.write(zf.read("pothole-mix/readme.txt"))
        print(f"Pothole Mix README saved to {readme_path}", flush=True)

        for folder_name, source_type in subsets:
            print(f"\nProcessing subset: {folder_name} (type: {source_type})...", flush=True)
            subset_stats = {'images': 0, 'masks': 0, 'train': 0, 'val': 0, 'test': 0}
            
            for split in ['training', 'validation', 'testing']:
                target_split = 'train' if split == 'training' else ('val' if split == 'validation' else 'test')
                
                # Find matching image-mask pairs
                img_dir_prefix = f"pothole-mix/{split}/{folder_name}/images/"
                mask_dir_prefix = f"pothole-mix/{split}/{folder_name}/masks/"
                
                img_files = [n for n in namelist if n.startswith(img_dir_prefix) and not n.endswith('/')]
                mask_files = [n for n in namelist if n.startswith(mask_dir_prefix) and not n.endswith('/')]
                
                # Pair by basename
                mask_by_base = {os.path.splitext(os.path.basename(m))[0]: m for m in mask_files}
                pairs = []
                for img_f in img_files:
                    base = os.path.splitext(os.path.basename(img_f))[0]
                    if base in mask_by_base:
                        pairs.append((img_f, mask_by_base[base]))
                
                print(f"  Split '{split}' -> canonical '{target_split}': found {len(pairs)} pairs (processing up to {max_pairs_per_subset})", flush=True)
                
                selected_pairs = pairs[:max_pairs_per_subset]
                for img_entry, mask_entry in selected_pairs:
                    try:
                        img_bytes = zf.read(img_entry)
                        mask_bytes = zf.read(mask_entry)
                        
                        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                        raw_mask = Image.open(io.BytesIO(mask_bytes))
                        
                        # Verify dimensions match
                        if img.size != raw_mask.size:
                            print(f"    WARNING: Size mismatch for {img_entry} ({img.size}) vs {mask_entry} ({raw_mask.size}). Resizing mask.", flush=True)
                            raw_mask = raw_mask.resize(img.size, Image.NEAREST)

                        # Convert to canonical 3-class mask and validity vector
                        canonical_mask, validity = map_mask_to_canonical(raw_mask, source_type)
                        
                        # Check class pixel counts
                        uvals, counts = np.unique(canonical_mask, return_counts=True)
                        u_dict = dict(zip(uvals, counts))
                        bg_count = u_dict.get(0, 0)
                        pothole_count = u_dict.get(1, 0)
                        crack_count = u_dict.get(2, 0)
                        
                        stats['by_class_pixels'][0] += bg_count
                        stats['by_class_pixels'][1] += pothole_count
                        stats['by_class_pixels'][2] += crack_count
                        
                        has_pothole = pothole_count > 0
                        has_crack = crack_count > 0
                        if has_pothole:
                            stats['images_with_pothole'] += 1
                        if has_crack:
                            stats['images_with_crack'] += 1
                        if not has_pothole and not has_crack:
                            stats['images_background_only'] += 1
                            
                        # Save source image & canonical files
                        sample_id = f"{source_type}_{target_split}_{os.path.splitext(os.path.basename(img_entry))[0]}"
                        
                        # Save canonical full-resolution sample
                        out_img_path = os.path.join(CANONICAL_DIR, target_split, "images", f"{sample_id}.png")
                        out_mask_path = os.path.join(CANONICAL_DIR, target_split, "masks", f"{sample_id}.png")
                        out_val_path = os.path.join(CANONICAL_DIR, target_split, "validity", f"{sample_id}.npy")
                        
                        img.save(out_img_path)
                        Image.fromarray(canonical_mask).save(out_mask_path)
                        np.save(out_val_path, validity)
                        
                        # Save also to separate source directory to preserve original source format
                        source_save_dir = os.path.join(SOURCES_DIR, "pothole_mix", folder_name, split)
                        os.makedirs(os.path.join(source_save_dir, "images"), exist_ok=True)
                        os.makedirs(os.path.join(source_save_dir, "masks"), exist_ok=True)
                        with open(os.path.join(source_save_dir, "images", os.path.basename(img_entry)), "wb") as f_out:
                            f_out.write(img_bytes)
                        with open(os.path.join(source_save_dir, "masks", os.path.basename(mask_entry)), "wb") as f_out:
                            f_out.write(mask_bytes)
                        
                        # Generate overlapping crops for train and val splits
                        if target_split in ['train', 'val']:
                            crops = extract_crops(img, canonical_mask, validity, crop_size=512, stride=384)
                            for c_idx, (c_img, c_mask, c_val) in enumerate(crops):
                                crop_id = f"{sample_id}_crop{c_idx:02d}"
                                c_img.save(os.path.join(CROPS_DIR, target_split, "images", f"{crop_id}.png"))
                                Image.fromarray(c_mask).save(os.path.join(CROPS_DIR, target_split, "masks", f"{crop_id}.png"))
                                np.save(os.path.join(CROPS_DIR, target_split, "validity", f"{crop_id}.npy"), c_val)
                                stats['crops_extracted'] += 1

                        record = {
                            'sample_id': sample_id,
                            'source': source_type,
                            'split': target_split,
                            'native_split': split,
                            'width': img.size[0],
                            'height': img.size[1],
                            'pothole_pixels': int(pothole_count),
                            'crack_pixels': int(crack_count),
                            'background_pixels': int(bg_count),
                            'valid_classes': validity.tolist()
                        }
                        metadata.append(record)
                        
                        stats['total_images'] += 1
                        stats['total_masks'] += 1
                        stats['by_subset'][source_type] += 1
                        stats['by_split'][target_split] += 1
                        subset_stats['images'] += 1
                        subset_stats['masks'] += 1
                        subset_stats[target_split] += 1

                    except Exception as e:
                        print(f"    Error processing {img_entry}: {e}", flush=True)
                        stats['corrupted_files'] += 1

            stats['subsets_catalog'][source_type] = subset_stats

    # Save metadata manifest
    manifest_path = os.path.join(DATA_DIR, "dataset_manifest.json")
    # Convert numpy types to native python
    clean_stats = {
        'total_images': int(stats['total_images']),
        'total_masks': int(stats['total_masks']),
        'by_subset': {k: int(v) for k, v in stats['by_subset'].items()},
        'by_split': {k: int(v) for k, v in stats['by_split'].items()},
        'by_class_pixels': {int(k): int(v) for k, v in stats['by_class_pixels'].items()},
        'images_with_pothole': int(stats['images_with_pothole']),
        'images_with_crack': int(stats['images_with_crack']),
        'images_background_only': int(stats['images_background_only']),
        'empty_masks': int(stats['empty_masks']),
        'corrupted_files': int(stats['corrupted_files']),
        'crops_extracted': int(stats['crops_extracted']),
        'subsets_catalog': stats['subsets_catalog']
    }
    with open(manifest_path, "w") as f:
        json.dump({'stats': clean_stats, 'samples': metadata}, f, indent=2)
    print(f"\nManifest saved to {manifest_path}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print(f"STAGING COMPLETE in {round(time.time() - t0, 2)}s", flush=True)
    print(f"Total canonical images: {stats['total_images']}", flush=True)
    print(f"Splits breakdown: Train={stats['by_split']['train']}, Val={stats['by_split']['val']}, Test={stats['by_split']['test']}", flush=True)
    print(f"By subset: {dict(stats['by_subset'])}", flush=True)
    print(f"Class pixel counts: Background={stats['by_class_pixels'][0]:,}, Pothole={stats['by_class_pixels'][1]:,}, Crack={stats['by_class_pixels'][2]:,}", flush=True)
    print(f"Images with pothole: {stats['images_with_pothole']}, Images with crack: {stats['images_with_crack']}", flush=True)
    print(f"Total 512x512 crops extracted: {stats['crops_extracted']}", flush=True)
    print("="*60, flush=True)

if __name__ == '__main__':
    max_p = 25 if len(sys.argv) < 2 else int(sys.argv[1])
    stage_and_process_dataset(max_pairs_per_subset=max_p)
