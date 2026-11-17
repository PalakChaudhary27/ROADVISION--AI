"""
Inspects Phase 4 checkpoints: SHA256 hashes, tensor shapes, weights, and differences.
"""
import os
import json
import hashlib
import torch

ARTIFACTS_DIR = r"D:\ROADVISION AI\artifacts\phase4_experiments"

def get_file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

ckpt_names = ["model_A_best.pth", "model_B_best.pth", "model_C_best.pth"]
ckpts = {}

print("=" * 80, flush=True)
print("CHECKPOINT INTEGRITY & WEIGHT INSPECTION", flush=True)
print("=" * 80, flush=True)

for name in ckpt_names:
    p = os.path.join(ARTIFACTS_DIR, name)
    sha = get_file_sha256(p)
    size_mb = os.path.getsize(p) / (1024 * 1024)
    
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", {})
    
    head_w = state_dict.get("classifier.4.weight")
    head_b = state_dict.get("classifier.4.bias")
    bb_w = state_dict.get("backbone.conv1.weight")
    
    has_nan = any(torch.isnan(v).any().item() for v in state_dict.values())
    has_inf = any(torch.isinf(v).any().item() for v in state_dict.values())
    
    ckpts[name] = {
        'sha256': sha,
        'size_mb': round(size_mb, 2),
        'keys': list(ckpt.keys()),
        'head_shape': list(head_w.shape) if head_w is not None else None,
        'head_mean': head_w.mean().item() if head_w is not None else None,
        'head_std': head_w.std().item() if head_w is not None else None,
        'head_norm': head_w.norm().item() if head_w is not None else None,
        'bb_norm': bb_w.norm().item() if bb_w is not None else None,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'state_dict': state_dict,
        'reported_metrics': ckpt.get("validation_metrics")
    }
    
    print(f"\nCheckpoint: {name}", flush=True)
    print(f"  SHA256: {sha}", flush=True)
    print(f"  File Size: {size_mb:.2f} MB", flush=True)
    print(f"  Top-level keys: {list(ckpt.keys())}", flush=True)
    print(f"  Head weight shape: {list(head_w.shape) if head_w is not None else 'None'}", flush=True)
    print(f"  Head weight norm: {head_w.norm().item():.6f} (Mean: {head_w.mean().item():.6f}, Std: {head_w.std().item():.6f})", flush=True)
    print(f"  Backbone conv1 norm: {bb_w.norm().item():.6f}", flush=True)
    print(f"  Contains NaN: {has_nan} | Contains Inf: {has_inf}", flush=True)

# Weight Differences Across Experiments
print("\n" + "-" * 80, flush=True)
print("INTER-EXPERIMENT WEIGHT COMPARISONS", flush=True)
print("-" * 80, flush=True)

names = list(ckpts.keys())
comparisons = {}
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        n1, n2 = names[i], names[j]
        s1 = ckpts[n1]['state_dict']
        s2 = ckpts[n2]['state_dict']
        
        # Check every layer
        identical_layers = []
        differing_layers = []
        max_diffs = {}
        
        for k in s1.keys():
            if k in s2:
                diff = (s1[k] - s2[k]).abs().max().item()
                if diff == 0.0:
                    identical_layers.append(k)
                else:
                    differing_layers.append(k)
                    max_diffs[k] = diff
                    
        print(f"\nComparing {n1} vs {n2}:", flush=True)
        print(f"  Total layers in common: {len(identical_layers) + len(differing_layers)}", flush=True)
        print(f"  Identical layers: {len(identical_layers)} ({len(identical_layers)/(len(identical_layers)+len(differing_layers))*100:.1f}%)", flush=True)
        print(f"  Differing layers: {len(differing_layers)}", flush=True)
        if differing_layers:
            print(f"  Differing layer names: {differing_layers[:5]}", flush=True)
            for dk in differing_layers[:3]:
                print(f"    {dk}: max abs diff = {max_diffs[dk]:.6f}", flush=True)
        else:
            print("  ALL WEIGHTS ARE 100% IDENTICAL!", flush=True)
            
        comparisons[f"{n1}_vs_{n2}"] = {
            'identical_count': len(identical_layers),
            'differing_count': len(differing_layers),
            'differing_layers': differing_layers,
            'max_diffs': max_diffs
        }

# Save summary
with open(r"D:\ROADVISION AI\scratch\checkpoint_inspection_summary.json", "w") as f:
    json.dump({
        'checkpoints': {k: {sk: sv for sk, sv in v.items() if sk != 'state_dict'} for k, v in ckpts.items()},
        'comparisons': comparisons
    }, f, indent=2)

print("\nCheckpoint inspection complete. Output saved to scratch/checkpoint_inspection_summary.json.", flush=True)
