"""
Evaluates model_A, model_B, model_C on the validation set to obtain actual measured metrics.
"""
import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50

WORKSPACE = Path(r"D:\ROADVISION AI")
CANONICAL_VAL_DIR = WORKSPACE / "data" / "clean_segmentation" / "canonical_3class" / "val"
ARTIFACTS_DIR = WORKSPACE / "artifacts" / "phase4_experiments"

class ValDataset(Dataset):
    def __init__(self, val_dir, max_samples=30):
        self.img_paths = sorted(list((Path(val_dir) / "images").glob("*.png")))[:max_samples]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        m_p = p.parent.parent / "masks" / p.name
        
        img = Image.open(p).convert('RGB').resize((512, 512), Image.Resampling.BILINEAR)
        mask = Image.open(m_p).resize((512, 512), Image.Resampling.NEAREST)
        
        img_arr = (np.array(img, dtype=np.float32) / 255.0 - self.mean) / self.std
        img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float()
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64)).long()
        
        return img_tensor, mask_tensor

def evaluate_model(model_path, tau=0.30):
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]
    
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
    model.load_state_dict(state_dict)
    model.eval()
    
    ds = ValDataset(CANONICAL_VAL_DIR, max_samples=25)
    loader = DataLoader(ds, batch_size=5, shuffle=False)
    
    conf_matrix = np.zeros((3, 3), dtype=np.int64)
    
    with torch.no_grad():
        for imgs, masks in loader:
            out = model(imgs)['out']
            probs = torch.sigmoid(out)
            
            p1 = probs[:, 1]
            p2 = probs[:, 2]
            
            preds = torch.zeros_like(masks)
            preds[(p2 > tau) & (p2 > p1)] = 2
            preds[(p1 > tau) & (p1 >= p2)] = 1
            
            t_flat = masks.numpy().flatten()
            p_flat = preds.numpy().flatten()
            
            for t_c in range(3):
                m_tc = (t_flat == t_c)
                if m_tc.any():
                    for p_c in range(3):
                        conf_matrix[t_c, p_c] += int(np.sum(p_flat[m_tc] == p_c))
                        
    # Compute metrics
    total_px = conf_matrix.sum()
    ious = []
    names = ["background", "pothole", "crack"]
    metrics = {}
    
    for c in range(3):
        tp = conf_matrix[c, c]
        fp = conf_matrix[:, c].sum() - tp
        fn = conf_matrix[c, :].sum() - tp
        prec = float(tp / max(1, tp + fp))
        rec = float(tp / max(1, tp + fn))
        iou = float(tp / max(1, tp + fp + fn))
        dice = float(2.0 * tp / max(1, 2 * tp + fp + fn))
        ious.append(iou)
        metrics[names[c]] = {
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'iou': round(iou * 100, 2),
            'dice': round(dice * 100, 2),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
        }
        
    damage_miou = round(float(np.mean([ious[1], ious[2]])) * 100, 2)
    overall_miou = round(float(np.mean(ious)) * 100, 2)
    metrics['summary'] = {
        'damage_miou': damage_miou,
        'overall_miou': overall_miou,
        'pothole_iou': metrics['pothole']['iou'],
        'pothole_recall': metrics['pothole']['recall'],
        'crack_iou': metrics['crack']['iou'],
        'crack_recall': metrics['crack']['recall'],
        'background_iou': metrics['background']['iou']
    }
    return metrics, conf_matrix.tolist()

if __name__ == "__main__":
    print("=" * 80, flush=True)
    print("REPRODUCING ACTUAL VALIDATION METRICS FROM CHECKPOINT WEIGHTS", flush=True)
    print("=" * 80, flush=True)
    
    results = {}
    for name in ["model_A_best.pth", "model_B_best.pth", "model_C_best.pth"]:
        p = ARTIFACTS_DIR / name
        print(f"\nEvaluating {name}...", flush=True)
        m, cm = evaluate_model(p)
        print(f"  Background IoU: {m['background']['iou']}%", flush=True)
        print(f"  Pothole IoU:    {m['pothole']['iou']}% (Recall: {m['pothole']['recall']}%)", flush=True)
        print(f"  Crack IoU:      {m['crack']['iou']}% (Recall: {m['crack']['recall']}%)", flush=True)
        print(f"  Damage mIoU:    {m['summary']['damage_miou']}%", flush=True)
        results[name] = {'actual_measured': m, 'confusion_matrix': cm}
        
    with open(WORKSPACE / "scratch" / "actual_measured_val_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved actual measured metrics to scratch/actual_measured_val_metrics.json", flush=True)
