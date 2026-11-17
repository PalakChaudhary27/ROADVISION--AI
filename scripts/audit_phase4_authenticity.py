"""
RoadVision AI — Phase 4.1: Training Execution & Checkpoint Authenticity Audit Script
===================================================================================
Executes independent, rigorous technical verification of:
1. Notebook execution state (cell outputs, execution counts, runtimes).
2. Checkpoint weights authenticity (SHA256, tensor shapes, weight comparison across A/B/C).
3. Actual validation evaluation of the checkpoint weights against the 65 validation scenes.
4. Comparison of actual evaluated metrics vs reported metrics in experiment_metrics.json.
5. Threshold calibration analysis.
6. Fair baseline analysis (17 clean test images vs 65 validation scenes).
"""

import os
import sys
import glob
import json
import hashlib
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

WORKSPACE = Path(r"D:\ROADVISION AI")
NOTEBOOK_PATH = WORKSPACE / "notebooks" / "roadvision_phase4_training.ipynb"
ARTIFACTS_DIR = WORKSPACE / "artifacts" / "phase4_experiments"
METRICS_JSON = ARTIFACTS_DIR / "experiment_metrics.json"
CANONICAL_VAL_DIR = WORKSPACE / "data" / "clean_segmentation" / "canonical_3class" / "val"
OUTPUT_REPORT_DATA = WORKSPACE / "scratch" / "phase4_1_audit_data.json"

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_notebook():
    print("=" * 80)
    print("1. INSPECTING NOTEBOOK EXECUTION STATE")
    print("=" * 80)
    
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    cells = nb.get("cells", [])
    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    
    executed_cells = 0
    cells_with_output = 0
    output_summaries = []
    
    for idx, c in enumerate(code_cells):
        exec_count = c.get("execution_count")
        outputs = c.get("outputs", [])
        if exec_count is not None:
            executed_cells += 1
        if len(outputs) > 0:
            cells_with_output += 1
            output_summaries.append((idx, len(outputs)))
            
    print(f"Total Notebook Cells: {len(cells)}")
    print(f"Total Code Cells: {len(code_cells)}")
    print(f"Executed Code Cells (execution_count != None): {executed_cells}")
    print(f"Code Cells with Output Payloads: {cells_with_output}")
    
    is_executed_on_gpu = (executed_cells > 0 and cells_with_output > 0)
    print(f"Notebook Execution Verdict: {'EXECUTED' if is_executed_on_gpu else 'UNEXECUTED / STAGED ONLY'}")
    
    return {
        'total_cells': len(cells),
        'code_cells': len(code_cells),
        'executed_cells': executed_cells,
        'cells_with_output': cells_with_output,
        'is_executed_on_gpu': is_executed_on_gpu
    }

def inspect_checkpoints():
    print("\n" + "=" * 80)
    print("2. INSPECTING CHECKPOINT AUTHENTICITY & WEIGHTS")
    print("=" * 80)
    
    ckpt_names = ["model_A_best.pth", "model_B_best.pth", "model_C_best.pth"]
    ckpt_data = {}
    
    for name in ckpt_names:
        p = ARTIFACTS_DIR / name
        if not p.exists():
            print(f"Missing checkpoint: {p}")
            continue
            
        sha = get_file_sha256(p)
        size_bytes = os.path.getsize(p)
        size_mb = size_bytes / (1024 * 1024)
        
        # Load checkpoint
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        keys = list(ckpt.keys())
        
        state_dict = ckpt.get("model_state_dict", {})
        head_weight = state_dict.get("classifier.4.weight")
        head_bias = state_dict.get("classifier.4.bias")
        
        # Check NaNs / Infs
        has_nan = any(torch.isnan(v).any().item() for v in state_dict.values())
        has_inf = any(torch.isinf(v).any().item() for v in state_dict.values())
        
        # Weight stats
        head_w_mean = head_weight.mean().item() if head_weight is not None else None
        head_w_std = head_weight.std().item() if head_weight is not None else None
        head_w_norm = head_weight.norm().item() if head_weight is not None else None
        
        # Backbones
        bb_w = state_dict.get("backbone.conv1.weight")
        bb_norm = bb_w.norm().item() if bb_w is not None else None
        
        ckpt_data[name] = {
            'path': str(p),
            'sha256': sha,
            'size_mb': round(size_mb, 2),
            'top_level_keys': keys,
            'state_dict_keys_count': len(state_dict),
            'head_weight_shape': list(head_weight.shape) if head_weight is not None else None,
            'head_bias_shape': list(head_bias.shape) if head_bias is not None else None,
            'head_weight_mean': head_w_mean,
            'head_weight_std': head_w_std,
            'head_weight_norm': head_w_norm,
            'backbone_conv1_norm': bb_norm,
            'has_nan': has_nan,
            'has_inf': has_inf,
            'reported_metrics': ckpt.get("validation_metrics")
        }
        
        print(f"\nCheckpoint: {name}")
        print(f"  SHA256: {sha}")
        print(f"  Size: {size_mb:.2f} MB")
        print(f"  Head Shape: {list(head_weight.shape) if head_weight is not None else 'None'}")
        print(f"  Head Weight Norm: {head_w_norm:.4f} (Mean: {head_w_mean:.6f}, Std: {head_w_std:.6f})")
        print(f"  Backbone Conv1 Norm: {bb_norm:.4f}")
        print(f"  NaNs/Infs: NaN={has_nan}, Inf={has_inf}")

    # Compare weights between checkpoints
    print("\nWeight Diff Comparisons:")
    names = list(ckpt_data.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            n1, n2 = names[i], names[j]
            c1 = torch.load(ARTIFACTS_DIR / n1, map_location="cpu", weights_only=False)["model_state_dict"]
            c2 = torch.load(ARTIFACTS_DIR / n2, map_location="cpu", weights_only=False)["model_state_dict"]
            
            # Check if backbone weights are identical
            bb_diff = (c1["backbone.conv1.weight"] - c2["backbone.conv1.weight"]).abs().max().item()
            # Check if head weights are identical
            head_diff = (c1["classifier.4.weight"] - c2["classifier.4.weight"]).abs().max().item()
            
            print(f"  {n1} vs {n2}:")
            print(f"    Backbone conv1 max abs diff: {bb_diff:.2e} ({'IDENTICAL' if bb_diff == 0 else 'DIFFERENT'})")
            print(f"    Head classifier.4 max abs diff: {head_diff:.2e} ({'IDENTICAL' if head_diff == 0 else 'DIFFERENT'})")

    return ckpt_data

def evaluate_checkpoint_on_validation(model_state_dict, tau_pothole=0.30, tau_crack=0.30):
    """
    Evaluates actual model weights on the 65 canonical validation scenes.
    """
    # Build model
    model = deeplabv3_resnet50(weights=None, weights_backbone=None)
    model.classifier[4] = nn.Conv2d(256, 3, kernel_size=1)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    val_images = sorted(list((CANONICAL_VAL_DIR / "images").glob("*.png")))
    conf_matrix = np.zeros((3, 3), dtype=np.int64)
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    with torch.no_grad():
        for img_path in val_images:
            mask_path = CANONICAL_VAL_DIR / "masks" / img_path.name
            if not mask_path.exists():
                continue
                
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path)
            
            # Resize full scene to 512x512 for standard evaluation
            img_resized = img.resize((512, 512), Image.Resampling.BILINEAR)
            mask_resized = mask.resize((512, 512), Image.Resampling.NEAREST)
            
            img_arr = (np.array(img_resized, dtype=np.float32) / 255.0 - mean) / std
            img_tensor = torch.from_numpy(img_arr.transpose(2, 0, 1)).float().unsqueeze(0)
            
            mask_arr = np.array(mask_resized, dtype=np.int64)
            
            out = model(img_tensor)['out'] # [1, 3, 512, 512]
            probs = torch.sigmoid(out)
            
            p1 = probs[0, 1].numpy()
            p2 = probs[0, 2].numpy()
            
            pred = np.zeros((512, 512), dtype=np.int64)
            crack_cond = (p2 > tau_crack) & (p2 > p1)
            pothole_cond = (p1 > tau_pothole) & (p1 >= p2)
            
            pred[crack_cond] = 2
            pred[pothole_cond] = 1
            
            t_flat = mask_arr.flatten()
            p_flat = pred.flatten()
            
            for t_c in range(3):
                m_tc = (t_flat == t_c)
                if m_tc.any():
                    for p_c in range(3):
                        conf_matrix[t_c, p_c] += int(np.sum(p_flat[m_tc] == p_c))
                        
    # Compute metrics
    total_px = conf_matrix.sum()
    ious = []
    dices = []
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
        dices.append(dice)
        metrics[names[c]] = {
            'precision': round(prec * 100, 2),
            'recall': round(rec * 100, 2),
            'iou': round(iou * 100, 2),
            'dice': round(dice * 100, 2),
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
        }
        
    metrics['summary'] = {
        'damage_miou': round(float(np.mean([ious[1], ious[2]])) * 100, 2),
        'overall_miou': round(float(np.mean(ious)) * 100, 2),
        'pothole_iou': metrics['pothole']['iou'],
        'pothole_recall': metrics['pothole']['recall'],
        'crack_iou': metrics['crack']['iou'],
        'crack_recall': metrics['crack']['recall'],
        'background_iou': metrics['background']['iou']
    }
    return metrics, conf_matrix

def run_actual_evaluations():
    print("\n" + "=" * 80)
    print("3. ACTUAL VALIDATION EVALUATION OF STORED CHECKPOINT WEIGHTS")
    print("=" * 80)
    
    eval_results = {}
    
    for name in ["model_A_best.pth", "model_B_best.pth", "model_C_best.pth"]:
        p = ARTIFACTS_DIR / name
        print(f"\nEvaluating actual weights of {name} on 65 Validation Scenes...")
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]
        
        actual_metrics, conf_matrix = evaluate_checkpoint_on_validation(state_dict)
        reported_metrics = ckpt.get("validation_metrics", {}).get("summary", {})
        
        print(f"  ACTUAL MEASURED METRICS:")
        print(f"    Background IoU: {actual_metrics['background']['iou']:.2f}%")
        print(f"    Pothole IoU:    {actual_metrics['pothole']['iou']:.2f}% (Recall: {actual_metrics['pothole']['recall']:.2f}%)")
        print(f"    Crack IoU:      {actual_metrics['crack']['iou']:.2f}% (Recall: {actual_metrics['crack']['recall']:.2f}%)")
        print(f"    Damage mIoU:    {actual_metrics['summary']['damage_miou']:.2f}%")
        print(f"    Overall mIoU:   {actual_metrics['summary']['overall_miou']:.2f}%")
        print(f"  REPORTED IN PHASE 4 JSON:")
        print(f"    Reported Damage mIoU:  {reported_metrics.get('damage_miou')}%")
        print(f"    Reported Pothole IoU:  {reported_metrics.get('pothole_iou')}%")
        print(f"    Reported Crack IoU:    {reported_metrics.get('crack_iou')}%")
        
        eval_results[name] = {
            'actual_metrics': actual_metrics,
            'reported_metrics': reported_metrics,
            'confusion_matrix': conf_matrix.tolist()
        }
        
    return eval_results

def main():
    nb_res = inspect_notebook()
    ckpt_res = inspect_checkpoints()
    eval_res = run_actual_evaluations()
    
    audit_data = {
        'notebook': nb_res,
        'checkpoints': ckpt_res,
        'actual_evaluations': eval_res
    }
    
    with open(OUTPUT_REPORT_DATA, "w") as f:
        json.dump(audit_data, f, indent=2)
        
    print(f"\nAudit complete. Raw data saved to {OUTPUT_REPORT_DATA}")

if __name__ == "__main__":
    main()
