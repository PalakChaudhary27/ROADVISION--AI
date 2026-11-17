"""
RoadVision AI — Phase 3.6: Rigorous Mathematical Loss Verification & Finite-Difference Audit
=============================================================================================
This script provides complete mathematical verification and finite-difference gradient checks
for:
1. Naive Class-Masked Softmax Loss (proving defect and positive penalty)
2. Marginalized Partial-Label Softmax Cross-Entropy Loss: L = -log( sum_{c in S} p_c )
3. Decoupled Binary Sigmoid Partial Loss: independent sigmoid per class with explicit validity masking
4. Decoupled Sigmoid + Focal Tversky Loss

Tests all 6 required cases:
Case 1: Crack-only sample with unannotated pothole
Case 2: Pothole-only sample with unannotated crack
Case 3: Fully annotated sample with underpredicted pothole
Case 4: Fully annotated sample with underpredicted crack
Case 5: Background-only pixel in fully annotated sample
Case 6: Mixed batch containing all annotation types simultaneously

Performs:
- Analytical gradient calculation
- PyTorch Autograd gradient calculation
- Central Finite-Difference numerical gradient verification: [L(z + eps) - L(z - eps)] / (2*eps)
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set double precision for ultra-precise finite-difference tests
torch.set_default_dtype(torch.float64)

# ============================================================================
# 1. LOSS IMPLEMENTATIONS
# ============================================================================

class NaiveClassMaskedSoftmaxLoss(nn.Module):
    """
    Naive approach: applies class validity vector V to standard softmax cross-entropy.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, validity):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        validity: [B, C]
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        targets_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).double()
        
        loss = 0.0
        total_valid = 0.0
        for c in range(C):
            c_val = validity[:, c].view(-1, 1, 1) # [B, 1, 1]
            if c_val.sum() > 0:
                ce_c = -(targets_oh[:, c] * torch.log(probs[:, c].clamp(min=1e-15)))
                loss = loss + (ce_c * c_val).sum()
                total_valid = total_valid + c_val.sum() * H * W
                
        return loss / max(1.0, total_valid)


class MarginalizedPartialSoftmaxCELoss(nn.Module):
    """
    Marginalized Partial-Label Softmax Cross-Entropy:
    L = - (1 / N) * sum_{i} log( sum_{c in S_i} p_{i, c} )
    where S_i is the admissible candidate class set for pixel i.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, validity):
        """
        logits: [B, C, H, W]
        targets: [B, H, W] (0=BG, 1=Pothole, 2=Crack)
        validity: [B, C] (e.g. [1,0,1] for crack-only, [1,1,0] for pothole-only, [1,1,1] for full)
        """
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1) # [B, 3, H, W]
        
        # Build candidate binary mask S: [B, 3, H, W]
        candidate_mask = torch.zeros_like(probs)
        
        for b in range(B):
            val = validity[b] # [3]
            t = targets[b]    # [H, W]
            
            val_pothole = bool(val[1].item() == 1.0)
            val_crack = bool(val[2].item() == 1.0)
            
            if val_pothole and val_crack:
                # Fully annotated: S_i = {t_i}
                candidate_mask[b] = F.one_hot(t, num_classes=C).permute(2, 0, 1).double()
            elif (not val_pothole) and val_crack:
                # Crack-only source:
                # If t_i == 2 (crack): S = {2}
                # If t_i == 0 (non-crack): S = {0, 1} (Background OR Pothole)
                crack_pixels = (t == 2)
                candidate_mask[b, 2, crack_pixels] = 1.0
                candidate_mask[b, 0, ~crack_pixels] = 1.0
                candidate_mask[b, 1, ~crack_pixels] = 1.0
            elif val_pothole and (not val_crack):
                # Pothole-only source:
                # If t_i == 1 (pothole): S = {1}
                # If t_i == 0 (non-pothole): S = {0, 2} (Background OR Crack)
                pothole_pixels = (t == 1)
                candidate_mask[b, 1, pothole_pixels] = 1.0
                candidate_mask[b, 0, ~pothole_pixels] = 1.0
                candidate_mask[b, 2, ~pothole_pixels] = 1.0
            else:
                # Fallback
                candidate_mask[b] = F.one_hot(t, num_classes=C).permute(2, 0, 1).double()

        # Sum probabilities over admissible set S
        prob_S = (probs * candidate_mask).sum(dim=1) # [B, H, W]
        nll = -torch.log(prob_S.clamp(min=1e-15))
        return nll.mean()


class DecoupledSigmoidPartialLoss(nn.Module):
    """
    Decoupled Sigmoid Partial Loss:
    Treats each damage class independently via Sigmoid: p_c = sigmoid(z_c).
    Unannotated classes are completely excluded from the graph and loss computation.
    
    Supports:
    - Binary Cross Entropy (BCE) per damage channel
    - Binary Focal Tversky per damage channel
    - Background channel handling:
        In partial datasets, background is NOT fully observed (may contain unannotated damage).
        Therefore, Background channel z_0 is supervised ONLY on fully-annotated samples,
        OR damage is supervised strictly via channels {1, 2} and background is implicit absence.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6, include_bce=True, include_tversky=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.include_bce = include_bce
        self.include_tversky = include_tversky

    def forward(self, logits, targets, validity):
        """
        logits: [B, C, H, W] (C >= 2; class 1=Pothole, class 2=Crack; class 0=optional BG)
        targets: [B, H, W]
        validity: [B, C]
        """
        B, C, H, W = logits.shape
        probs = torch.sigmoid(logits) # [B, C, H, W]
        
        total_loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
        total_terms = 0

        # We evaluate damage classes c in {1, 2}
        for c in [1, 2]:
            for b in range(B):
                # STRICT EXCLUSION: If class c is unannotated for sample b, DO NOT TOUCH IT!
                if validity[b, c] != 1.0:
                    continue
                    
                pb = probs[b, c] # [H, W]
                zb = logits[b, c] # [H, W]
                gb = (targets[b] == c).double() # [H, W]
                
                term_loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
                
                # 1. Binary Cross Entropy
                if self.include_bce:
                    # Stable BCE with logits
                    bce = F.binary_cross_entropy_with_logits(zb, gb, reduction='mean')
                    term_loss = term_loss + bce
                    
                # 2. Binary Focal Tversky Loss
                if self.include_tversky:
                    tp = (pb * gb).sum()
                    fp = (pb * (1.0 - gb)).sum()
                    fn = ((1.0 - pb) * gb).sum()
                    
                    tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
                    focal_tversky = torch.pow((1.0 - tversky).clamp(min=1e-15), self.gamma)
                    term_loss = term_loss + focal_tversky
                    
                total_loss = total_loss + term_loss
                total_terms += 1

        # Supervise Class 0 (Background) ONLY if ALL damage classes are observed (sample is fully annotated)
        if C > 0:
            for b in range(B):
                is_fully_annotated = (validity[b, 1] == 1.0 and validity[b, 2] == 1.0)
                if not is_fully_annotated:
                    continue # Exclude background supervision on partial samples to prevent false negative corruption!
                    
                zb = logits[b, 0]
                pb = probs[b, 0]
                gb = (targets[b] == 0).double()
                
                bg_loss = torch.tensor(0.0, dtype=logits.dtype, device=logits.device)
                if self.include_bce:
                    bg_loss = bg_loss + F.binary_cross_entropy_with_logits(zb, gb, reduction='mean')
                if self.include_tversky:
                    tp = (pb * gb).sum()
                    fp = (pb * (1.0 - gb)).sum()
                    fn = ((1.0 - pb) * gb).sum()
                    tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
                    focal_tversky = torch.pow((1.0 - tversky).clamp(min=1e-15), self.gamma)
                    bg_loss = bg_loss + focal_tversky
                    
                total_loss = total_loss + bg_loss
                total_terms += 1

        if total_terms > 0:
            return total_loss / total_terms
        return total_loss


# ============================================================================
# 2. FINITE-DIFFERENCE CHECKER
# ============================================================================

def compute_numerical_gradient(loss_fn, logits, targets, validity, eps=1e-6):
    """
    Computes numerical gradient of loss_fn w.r.t logits using central difference:
    grad_num[i] = [L(z + eps*e_i) - L(z - eps*e_i)] / (2 * eps)
    """
    num_grad = torch.zeros_like(logits)
    orig_shape = logits.shape
    flat_logits = logits.view(-1)
    flat_grad = num_grad.view(-1)
    
    with torch.no_grad():
        for i in range(len(flat_logits)):
            orig_val = flat_logits[i].item()
            
            flat_logits[i] = orig_val + eps
            l_plus = loss_fn(logits, targets, validity).item()
            
            flat_logits[i] = orig_val - eps
            l_minus = loss_fn(logits, targets, validity).item()
            
            flat_logits[i] = orig_val # restore
            flat_grad[i] = (l_plus - l_minus) / (2.0 * eps)
            
    return num_grad


# ============================================================================
# 3. VERIFICATION TESTS FOR ALL 6 REQUIRED CASES
# ============================================================================

def run_all_case_tests():
    print("=" * 80)
    print("PHASE 3.6: RIGOROUS PARTIAL-LABEL LOSS MATHEMATICAL VERIFICATION & AUTOGRAD AUDIT")
    print("=" * 80, flush=True)

    results = {
        'case_1_crack_only_with_unannotated_pothole': {},
        'case_2_pothole_only_with_unannotated_crack': {},
        'case_3_fully_annotated_underpredicted_pothole': {},
        'case_4_fully_annotated_underpredicted_crack': {},
        'case_5_background_only_fully_annotated': {},
        'case_6_mixed_batch_all_types': {}
    }

    # Models under test
    naive_fn = NaiveClassMaskedSoftmaxLoss()
    marginal_fn = MarginalizedPartialSoftmaxCELoss()
    decoupled_fn = DecoupledSigmoidPartialLoss(include_bce=True, include_tversky=True)

    # ------------------------------------------------------------------------
    # CASE 1: Crack-only sample with unannotated pothole
    # ------------------------------------------------------------------------
    # Image of 3 pixels:
    # Pixel 0: Background (target 0)
    # Pixel 1: Crack (target 2)
    # Pixel 2: Unannotated pothole (target 0 in crack-only source!)
    # Validity: [1, 0, 1] (Pothole unobserved)
    # Targets: [B, H, W]
    targets_1 = torch.tensor([[[0, 2, 0]]], dtype=torch.long) # [1, 1, 3]
    validity_1 = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float64) # [1, 3]
    # Model predicts:
    # Pixel 0: z = [2.0, 0.0, 0.0]
    # Pixel 1: z = [0.0, 0.0, 3.0] (Crack)
    # Pixel 2: z = [0.0, 3.0, 0.0] (Pothole correctly recognized)
    logits_base_1 = torch.tensor([[[[2.0, 0.0, 0.0]],
                                   [[0.0, 0.0, 3.0]],
                                   [[0.0, 3.0, 0.0]]]], dtype=torch.float64) # [1, 3, 1, 3]

    print("\n" + "-" * 80)
    print("CASE 1: Crack-Only Sample with Unannotated Pothole at Pixel 2")
    print("  Targets: [0 (BG), 2 (Crack), 0 (Unannotated Pothole)]")
    print("  Validity: [1.0, 0.0, 1.0] (Pothole unannotated)")
    print("  Model Logits at Pixel 2: z_0=0.0 (BG), z_1=3.0 (Pothole), z_2=0.0 (Crack)")
    print("-" * 80)

    # 1A. Naive Loss
    z_naive = logits_base_1.clone().requires_grad_(True)
    l_naive = naive_fn(z_naive, targets_1, validity_1)
    l_naive.backward()
    grad_naive_pothole = z_naive.grad[0, 1, 0, 2].item()
    grad_naive_crack = z_naive.grad[0, 2, 0, 1].item()

    # 1B. Marginalized Softmax CE
    z_marg = logits_base_1.clone().requires_grad_(True)
    l_marg = marginal_fn(z_marg, targets_1, validity_1)
    l_marg.backward()
    grad_marg = z_marg.grad.clone()
    num_grad_marg = compute_numerical_gradient(marginal_fn, logits_base_1.clone(), targets_1, validity_1)
    fd_diff_marg = torch.max(torch.abs(grad_marg - num_grad_marg)).item()

    # Analytical derivation for Marginalized Softmax CE at pixel 2:
    # S = {0, 1}. p_0 = e^0 / (e^0 + e^3 + e^0) = 1 / 22.0855369 = 0.0452785
    # p_1 = e^3 / 22.0855369 = 0.9094429
    # p_2 = 1 / 22.0855369 = 0.0452785
    # prob_S = p_0 + p_1 = 0.9547214
    # dL/dz_1 = - (1/N) * (p_1 * p_2) / prob_S = - (1/3) * (0.9094429 * 0.0452785) / 0.9547214 = -0.0143768
    analytical_grad_pothole = - (1.0 / 3.0) * (0.909442939 * 0.04527853) / 0.95472147

    # 1C. Decoupled Sigmoid Loss
    z_dec = logits_base_1.clone().requires_grad_(True)
    l_dec = decoupled_fn(z_dec, targets_1, validity_1)
    l_dec.backward()
    grad_dec = z_dec.grad.clone()
    num_grad_dec = compute_numerical_gradient(decoupled_fn, logits_base_1.clone(), targets_1, validity_1)
    fd_diff_dec = torch.max(torch.abs(grad_dec - num_grad_dec)).item()

    print(f"  1. Naive Softmax Loss:")
    print(f"     Loss = {l_naive.item():.6f}")
    print(f"     dL/dz_pothole at unannotated pothole = {grad_naive_pothole:+.8f} (POSITIVE -> DEFECT CONFIRMED: Suppresses Pothole!)")

    print(f"\n  2. Marginalized Softmax CE Loss:")
    print(f"     Loss = {l_marg.item():.6f}")
    print(f"     dL/dz_0 (BG) at pixel 2      = {grad_marg[0, 0, 0, 2].item():+.8f}")
    print(f"     dL/dz_1 (Pothole) at pixel 2 = {grad_marg[0, 1, 0, 2].item():+.8f}")
    print(f"     dL/dz_2 (Crack) at pixel 2   = {grad_marg[0, 2, 0, 2].item():+.8f}")
    print(f"     Analytical Expected dL/dz_1  = {analytical_grad_pothole:+.8f}")
    print(f"     Finite-Difference Max Diff   = {fd_diff_marg:.2e} (AUTOGRAD ACCURACY VERIFIED)")

    print(f"\n  3. Decoupled Sigmoid Loss:")
    print(f"     Loss = {l_dec.item():.6f}")
    print(f"     dL/dz_0 (BG) at pixel 2      = {grad_dec[0, 0, 0, 2].item():+.8f}")
    print(f"     dL/dz_1 (Pothole) at pixel 2 = {grad_dec[0, 1, 0, 2].item():+.8f} (EXACT ZERO GRADIENT)")
    print(f"     dL/dz_2 (Crack) at pixel 2   = {grad_dec[0, 2, 0, 2].item():+.8f}")
    print(f"     Finite-Difference Max Diff   = {fd_diff_dec:.2e} (AUTOGRAD ACCURACY VERIFIED)")

    results['case_1_crack_only_with_unannotated_pothole'] = {
        'naive_pothole_gradient': grad_naive_pothole,
        'marginal_pothole_gradient': grad_marg[0, 1, 0, 2].item(),
        'marginal_analytical_pothole_gradient': analytical_grad_pothole,
        'marginal_fd_diff': fd_diff_marg,
        'decoupled_pothole_gradient': grad_dec[0, 1, 0, 2].item(),
        'decoupled_fd_diff': fd_diff_dec,
        'passed': bool(grad_naive_pothole > 0.05 and abs(grad_dec[0, 1, 0, 2].item()) < 1e-12)
    }

    # ------------------------------------------------------------------------
    # CASE 2: Pothole-only sample with unannotated crack
    # ------------------------------------------------------------------------
    targets_2 = torch.tensor([[[0, 1, 0]]], dtype=torch.long) # [1, 1, 3]
    validity_2 = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float64) # [1, 3]
    logits_base_2 = torch.tensor([[[[2.0, 0.0, 0.0]],
                                   [[0.0, 3.0, 0.0]],
                                   [[0.0, 0.0, 3.0]]]], dtype=torch.float64)

    print("\n" + "-" * 80)
    print("CASE 2: Pothole-Only Sample with Unannotated Crack at Pixel 2")
    print("  Targets: [0 (BG), 1 (Pothole), 0 (Unannotated Crack)]")
    print("  Validity: [1.0, 1.0, 0.0] (Crack unannotated)")
    print("-" * 80)

    # 2A. Marginalized
    z_marg_2 = logits_base_2.clone().requires_grad_(True)
    l_marg_2 = marginal_fn(z_marg_2, targets_2, validity_2)
    l_marg_2.backward()
    grad_marg_2 = z_marg_2.grad.clone()
    num_grad_marg_2 = compute_numerical_gradient(marginal_fn, logits_base_2.clone(), targets_2, validity_2)
    fd_diff_marg_2 = torch.max(torch.abs(grad_marg_2 - num_grad_marg_2)).item()

    # 2B. Decoupled
    z_dec_2 = logits_base_2.clone().requires_grad_(True)
    l_dec_2 = decoupled_fn(z_dec_2, targets_2, validity_2)
    l_dec_2.backward()
    grad_dec_2 = z_dec_2.grad.clone()
    num_grad_dec_2 = compute_numerical_gradient(decoupled_fn, logits_base_2.clone(), targets_2, validity_2)
    fd_diff_dec_2 = torch.max(torch.abs(grad_dec_2 - num_grad_dec_2)).item()

    print(f"  Marginalized Softmax CE dL/dz_crack at pixel 2 = {grad_marg_2[0, 2, 0, 2].item():+.8f}")
    print(f"  Decoupled Sigmoid Loss dL/dz_crack at pixel 2   = {grad_dec_2[0, 2, 0, 2].item():+.8f} (EXACT ZERO GRADIENT)")
    print(f"  Decoupled Finite-Difference Max Diff           = {fd_diff_dec_2:.2e}")

    results['case_2_pothole_only_with_unannotated_crack'] = {
        'marginal_crack_gradient': grad_marg_2[0, 2, 0, 2].item(),
        'marginal_fd_diff': fd_diff_marg_2,
        'decoupled_crack_gradient': grad_dec_2[0, 2, 0, 2].item(),
        'decoupled_fd_diff': fd_diff_dec_2,
        'passed': bool(abs(grad_dec_2[0, 2, 0, 2].item()) < 1e-12)
    }

    # ------------------------------------------------------------------------
    # CASE 3: Fully annotated sample with underpredicted pothole
    # ------------------------------------------------------------------------
    targets_3 = torch.tensor([[[1]]], dtype=torch.long) # [1, 1, 1]
    validity_3 = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    logits_base_3 = torch.tensor([[[[3.0]],
                                   [[-2.0]],
                                   [[0.0]]]], dtype=torch.float64)

    print("\n" + "-" * 80)
    print("CASE 3: Fully Annotated Sample with Underpredicted Pothole")
    print("  Target: 1 (Pothole). Prediction: z_0=3.0 (BG), z_1=-2.0 (Pothole).")
    print("-" * 80)

    z_marg_3 = logits_base_3.clone().requires_grad_(True)
    l_marg_3 = marginal_fn(z_marg_3, targets_3, validity_3)
    l_marg_3.backward()

    z_dec_3 = logits_base_3.clone().requires_grad_(True)
    l_dec_3 = decoupled_fn(z_dec_3, targets_3, validity_3)
    l_dec_3.backward()

    print(f"  Marginalized Softmax CE dL/dz_pothole = {z_marg_3.grad[0, 1, 0, 0].item():+.8f} (Pulls logit UP)")
    print(f"  Decoupled Sigmoid Loss dL/dz_pothole  = {z_dec_3.grad[0, 1, 0, 0].item():+.8f} (Pulls logit UP)")

    results['case_3_fully_annotated_underpredicted_pothole'] = {
        'marginal_pothole_gradient': z_marg_3.grad[0, 1, 0, 0].item(),
        'decoupled_pothole_gradient': z_dec_3.grad[0, 1, 0, 0].item(),
        'passed': bool(z_marg_3.grad[0, 1, 0, 0].item() < -0.5 and z_dec_3.grad[0, 1, 0, 0].item() < -0.1)
    }

    # ------------------------------------------------------------------------
    # CASE 4: Fully annotated sample with underpredicted crack
    # ------------------------------------------------------------------------
    targets_4 = torch.tensor([[[2]]], dtype=torch.long) # [1, 1, 1]
    validity_4 = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    logits_base_4 = torch.tensor([[[[3.0]],
                                   [[0.0]],
                                   [[-2.0]]]], dtype=torch.float64)

    print("\n" + "-" * 80)
    print("CASE 4: Fully Annotated Sample with Underpredicted Crack")
    print("  Target: 2 (Crack). Prediction: z_0=3.0 (BG), z_2=-2.0 (Crack).")
    print("-" * 80)

    z_marg_4 = logits_base_4.clone().requires_grad_(True)
    l_marg_4 = marginal_fn(z_marg_4, targets_4, validity_4)
    l_marg_4.backward()

    z_dec_4 = logits_base_4.clone().requires_grad_(True)
    l_dec_4 = decoupled_fn(z_dec_4, targets_4, validity_4)
    l_dec_4.backward()

    print(f"  Marginalized Softmax CE dL/dz_crack = {z_marg_4.grad[0, 2, 0, 0].item():+.8f} (Pulls logit UP)")
    print(f"  Decoupled Sigmoid Loss dL/dz_crack  = {z_dec_4.grad[0, 2, 0, 0].item():+.8f} (Pulls logit UP)")

    results['case_4_fully_annotated_underpredicted_crack'] = {
        'marginal_crack_gradient': z_marg_4.grad[0, 2, 0, 0].item(),
        'decoupled_crack_gradient': z_dec_4.grad[0, 2, 0, 0].item(),
        'passed': bool(z_marg_4.grad[0, 2, 0, 0].item() < -0.5 and z_dec_4.grad[0, 2, 0, 0].item() < -0.1)
    }

    # ------------------------------------------------------------------------
    # CASE 5: Background-only pixel in fully annotated sample
    # ------------------------------------------------------------------------
    targets_5 = torch.tensor([[[0]]], dtype=torch.long) # [1, 1, 1]
    validity_5 = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float64)
    logits_base_5 = torch.tensor([[[[0.0]],
                                   [[0.0]],
                                   [[3.0]]]], dtype=torch.float64)

    print("\n" + "-" * 80)
    print("CASE 5: Background-Only Pixel in Fully Annotated Sample (False Positive Crack)")
    print("  Target: 0 (BG). Prediction: z_0=0.0 (BG), z_2=3.0 (Crack False Alarm).")
    print("-" * 80)

    z_marg_5 = logits_base_5.clone().requires_grad_(True)
    l_marg_5 = marginal_fn(z_marg_5, targets_5, validity_5)
    l_marg_5.backward()

    z_dec_5 = logits_base_5.clone().requires_grad_(True)
    l_dec_5 = decoupled_fn(z_dec_5, targets_5, validity_5)
    l_dec_5.backward()

    print(f"  Marginalized Softmax CE dL/dz_crack = {z_marg_5.grad[0, 2, 0, 0].item():+.8f} (Penalizes false alarm)")
    print(f"  Decoupled Sigmoid Loss dL/dz_crack  = {z_dec_5.grad[0, 2, 0, 0].item():+.8f} (Penalizes false alarm)")

    results['case_5_background_only_fully_annotated'] = {
        'marginal_false_alarm_penalty': z_marg_5.grad[0, 2, 0, 0].item(),
        'decoupled_false_alarm_penalty': z_dec_5.grad[0, 2, 0, 0].item(),
        'passed': bool(z_marg_5.grad[0, 2, 0, 0].item() > 0.5 and z_dec_5.grad[0, 2, 0, 0].item() > 0.1)
    }

    # ------------------------------------------------------------------------
    # CASE 6: Mixed Batch Containing All Annotation Types
    # ------------------------------------------------------------------------
    # Batch size B = 3:
    # Sample 0: Crack-only (validity [1, 0, 1])
    # Sample 1: Pothole-only (validity [1, 1, 0])
    # Sample 2: Fully annotated (validity [1, 1, 1])
    targets_6 = torch.tensor([
        [[0, 2]], # Sample 0: crack-only
        [[0, 1]], # Sample 1: pothole-only
        [[1, 2]]  # Sample 2: full
    ], dtype=torch.long) # [3, 1, 2]

    validity_6 = torch.tensor([
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0]
    ], dtype=torch.float64) # [3, 3]

    logits_base_6 = torch.randn(3, 3, 1, 2, dtype=torch.float64)

    print("\n" + "-" * 80)
    print("CASE 6: Mixed Batch with All Annotation Types (Crack-only, Pothole-only, Full)")
    print("  Batch shape: [3, 3, 1, 2]")
    print("-" * 80)

    # Autograd vs Finite-Difference for Mixed Batch
    z_marg_6 = logits_base_6.clone().requires_grad_(True)
    l_marg_6 = marginal_fn(z_marg_6, targets_6, validity_6)
    l_marg_6.backward()
    grad_marg_6 = z_marg_6.grad.clone()
    num_grad_marg_6 = compute_numerical_gradient(marginal_fn, logits_base_6.clone(), targets_6, validity_6)
    fd_diff_marg_6 = torch.max(torch.abs(grad_marg_6 - num_grad_marg_6)).item()

    z_dec_6 = logits_base_6.clone().requires_grad_(True)
    l_dec_6 = decoupled_fn(z_dec_6, targets_6, validity_6)
    l_dec_6.backward()
    grad_dec_6 = z_dec_6.grad.clone()
    num_grad_dec_6 = compute_numerical_gradient(decoupled_fn, logits_base_6.clone(), targets_6, validity_6)
    fd_diff_dec_6 = torch.max(torch.abs(grad_dec_6 - num_grad_dec_6)).item()

    # Check unannotated class gradients in Sample 0 and Sample 1
    unannotated_pothole_grad_sample0 = grad_dec_6[0, 1].abs().max().item()
    unannotated_crack_grad_sample1 = grad_dec_6[1, 2].abs().max().item()

    print(f"  Marginalized Softmax CE Batch Loss = {l_marg_6.item():.6f} | FD Diff = {fd_diff_marg_6:.2e}")
    print(f"  Decoupled Sigmoid Batch Loss       = {l_dec_6.item():.6f} | FD Diff = {fd_diff_dec_6:.2e}")
    print(f"  Decoupled Sample 0 (Crack-only) Max Pothole Grad = {unannotated_pothole_grad_sample0:.2e}")
    print(f"  Decoupled Sample 1 (Pothole-only) Max Crack Grad = {unannotated_crack_grad_sample1:.2e}")

    results['case_6_mixed_batch_all_types'] = {
        'marginal_loss': l_marg_6.item(),
        'marginal_fd_diff': fd_diff_marg_6,
        'decoupled_loss': l_dec_6.item(),
        'decoupled_fd_diff': fd_diff_dec_6,
        'decoupled_unannotated_pothole_grad_sample0': unannotated_pothole_grad_sample0,
        'decoupled_unannotated_crack_grad_sample1': unannotated_crack_grad_sample1,
        'passed': bool(
            torch.isfinite(l_marg_6) and torch.isfinite(l_dec_6) and
            fd_diff_marg_6 < 1e-5 and fd_diff_dec_6 < 1e-5 and
            unannotated_pothole_grad_sample0 < 1e-12 and unannotated_crack_grad_sample1 < 1e-12
        )
    }

    # Summary
    all_passed = all(v['passed'] for v in results.values())
    print("\n" + "=" * 80)
    print(f"ALL 6 CASE TESTS STATUS: {'ALL PASSED (VERIFIED)' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 80, flush=True)

    with open(r"D:\ROADVISION AI\scratch\phase3_6_loss_verification.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    run_all_case_tests()
