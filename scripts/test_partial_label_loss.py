"""
RoadVision AI — Phase 3.5: Rigorous Partial-Label Loss Verification & Unit Tests
================================================================================
Investigates:
1. The Softmax Coupling Problem:
   Proves mathematically and empirically that naive class-masked loss on Softmax outputs
   causes background supervision to penalize unobserved damage classes.
2. Mathematically Correct Marginalized Partial-Label Softmax Objective:
   L_partial = -log( sum_{c in S} p_c )
   where S is the candidate set:
     - Crack pixel: S = {2}
     - Non-crack pixel in crack-only source: S = {0, 1} (Background OR Pothole)
     - Pothole pixel: S = {1}
     - Non-pothole pixel in pothole-only source: S = {0, 2} (Background OR Crack)
     - Fully annotated pixel: S = {target}
3. Decoupled Multi-Label Binary Tversky Loss:
   Independent sigmoid per class, where unobserved classes have zero loss and zero gradient.
4. Unit Tests for:
   - Case A: Crack-only sample with an unannotated pothole.
   - Case B: Pothole-only sample with an unannotated crack.
   - Case C: Fully annotated 3-class sample.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NaiveClassMaskedLoss(nn.Module):
    """
    The naive approach from Phase 3:
    Applies class-level validity vector to Softmax cross-entropy or Tversky.
    Exposes the Softmax Coupling Defect.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, validity):
        """
        logits: [B, C, H, W]
        targets: [B, H, W]
        validity: [B, C]
        """
        probs = F.softmax(logits, dim=1) # [B, C, H, W]
        num_classes = logits.shape[1]
        targets_oh = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        loss = 0.0
        for c in range(num_classes):
            c_val = validity[:, c].view(-1, 1, 1) # [B, 1, 1]
            # Cross entropy component
            ce_c = - (targets_oh[:, c] * torch.log(probs[:, c] + 1e-7))
            loss = loss + (ce_c * c_val).mean()
        return loss

class MarginalizedPartialCrossEntropyLoss(nn.Module):
    """
    Mathematically Correct Partial-Label Cross-Entropy:
    L = -log( sum_{c in S_{i}} p_c )
    
    Candidate set S:
    - If sample is fully annotated (validity == [1, 1, 1]):
        S = {y_i}
    - If crack-only (validity == [1, 0, 1]):
        y_i == 2 (crack) -> S = {2}
        y_i == 0 (non-crack) -> S = {0, 1} (could be BG or Pothole)
    - If pothole-only (validity == [1, 1, 0]):
        y_i == 1 (pothole) -> S = {1}
        y_i == 0 (non-pothole) -> S = {0, 2} (could be BG or Crack)
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets, validity):
        B, C, H, W = logits.shape
        probs = F.softmax(logits, dim=1) # [B, 3, H, W]
        
        # Build candidate binary mask S: [B, 3, H, W]
        candidate_mask = torch.zeros_like(probs)
        
        for b in range(B):
            val = validity[b] # [3]
            t = targets[b]    # [H, W]
            
            if val[1] == 1 and val[2] == 1:
                # Fully annotated
                candidate_mask[b] = F.one_hot(t, num_classes=C).permute(2, 0, 1).float()
            elif val[1] == 0 and val[2] == 1:
                # Crack-only source
                # At crack: only crack {2}
                crack_mask = (t == 2)
                candidate_mask[b, 2, crack_mask] = 1.0
                # At non-crack: {0, 1} (BG or Pothole)
                candidate_mask[b, 0, ~crack_mask] = 1.0
                candidate_mask[b, 1, ~crack_mask] = 1.0
            elif val[1] == 1 and val[2] == 0:
                # Pothole-only source
                # At pothole: only pothole {1}
                pothole_mask = (t == 1)
                candidate_mask[b, 1, pothole_mask] = 1.0
                # At non-pothole: {0, 2} (BG or Crack)
                candidate_mask[b, 0, ~pothole_mask] = 1.0
                candidate_mask[b, 2, ~pothole_mask] = 1.0
            else:
                # Default to one-hot target
                candidate_mask[b] = F.one_hot(t, num_classes=C).permute(2, 0, 1).float()
                
        # Probability mass assigned to admissible candidate set S
        prob_S = (probs * candidate_mask).sum(dim=1) # [B, H, W]
        loss = -torch.log(prob_S.clamp(min=1e-7))
        return loss.mean()

class DecoupledPartialFocalTverskyLoss(nn.Module):
    """
    Decoupled Partial Focal Tversky Loss:
    Uses independent Sigmoids for damage classes, entirely removing softmax cross-talk.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets, validity):
        """
        logits: [B, C, H, W] (class 1: Pothole, class 2: Crack)
        targets: [B, H, W]
        validity: [B, C]
        """
        B, C, H, W = logits.shape
        # Independent sigmoids
        probs = torch.sigmoid(logits)
        
        total_loss = 0.0
        active_classes = 0
        
        # Supervise damage classes c in {1, 2}
        for c in [1, 2]:
            c_valid = validity[:, c] # [B]
            if c_valid.sum() == 0:
                continue
                
            p = probs[:, c] # [B, H, W]
            g = (targets == c).float() # [B, H, W]
            
            # For samples where class c is valid:
            for b in range(B):
                if c_valid[b] == 1.0:
                    pb = p[b]
                    gb = g[b]
                    tp = (pb * gb).sum()
                    fp = (pb * (1 - gb)).sum()
                    fn = ((1 - pb) * gb).sum()
                    
                    tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
                    focal_tversky = torch.pow((1.0 - tversky), self.gamma)
                    total_loss = total_loss + focal_tversky
                    active_classes += 1

        if active_classes > 0:
            return total_loss / active_classes
        return total_loss

def run_coupling_demonstration():
    print("=" * 70)
    print("PARTIAL-LABEL LOSS AUDIT: SOFTMAX COUPLING VERIFICATION")
    print("=" * 70, flush=True)

    # Simulate an image with 3 pixels:
    # Pixel 0: Background
    # Pixel 1: Crack
    # Pixel 2: UNANNOTATED Pothole (labeled as 0 in a crack-only dataset)
    
    # Target in crack-only dataset:
    # Pixel 0 -> 0 (BG)
    # Pixel 1 -> 2 (Crack)
    # Pixel 2 -> 0 (BG) [False Negative ground truth because pothole was unannotated!]
    targets = torch.tensor([[0, 2, 0]], dtype=torch.long) # [1, 3]
    validity = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float) # Crack-only: pothole is unobserved!

    # Model predicts:
    # Pixel 0: high BG [3.0, 0.0, 0.0]
    # Pixel 1: high Crack [0.0, 0.0, 3.0]
    # Pixel 2: high POTHOLE [0.0, 4.0, 0.0] (the model correctly recognized the pothole!)
    
    logits_naive = torch.tensor([[[3.0, 0.0, 0.0],
                                  [0.0, 0.0, 4.0],
                                  [0.0, 3.0, 0.0]]], dtype=torch.float, requires_grad=True)

    logits_marginal = logits_naive.clone().detach().requires_grad_(True)
    logits_decoupled = logits_naive.clone().detach().requires_grad_(True)

    # 1. Test Naive Class-Masked Loss
    naive_loss_fn = NaiveClassMaskedLoss()
    loss_naive = naive_loss_fn(logits_naive.unsqueeze(2), targets.unsqueeze(1), validity)
    loss_naive.backward()
    grad_pothole_naive = logits_naive.grad[0, 1, 2].item() # Gradient on pothole logit at unannotated pothole pixel

    # 2. Test Marginalized Partial Cross-Entropy Loss
    marginal_loss_fn = MarginalizedPartialCrossEntropyLoss()
    loss_marginal = marginal_loss_fn(logits_marginal.unsqueeze(2), targets.unsqueeze(1), validity)
    loss_marginal.backward()
    grad_pothole_marginal = logits_marginal.grad[0, 1, 2].item()

    # 3. Test Decoupled Partial Focal Tversky Loss
    decoupled_loss_fn = DecoupledPartialFocalTverskyLoss()
    loss_decoupled = decoupled_loss_fn(logits_decoupled.unsqueeze(2), targets.unsqueeze(1), validity)
    loss_decoupled.backward()
    grad_pothole_decoupled = logits_decoupled.grad[0, 1, 2].item()

    print(f"\n[Case A] Crack-Only Image with Unannotated Pothole at Pixel 2:")
    print(f"  Target at Pixel 2: 0 (Assigned BG by naive dataset)")
    print(f"  Model Logit for Pothole: z_1 = 4.0 (Model correctly predicts pothole)")
    print(f"  Validity: {validity.tolist()[0]} (Pothole class marked unobserved)")
    print("-" * 70)
    print(f"  1. Naive Class-Masked Loss:")
    print(f"     Loss = {loss_naive.item():.4f}")
    print(f"     dL/dz_pothole at unannotated pothole = {grad_pothole_naive:+.6f}")
    if grad_pothole_naive > 0.01:
        print("     [DEFECT CONFIRMED] POSITIVE gradient penalizes and suppresses pothole prediction!")
    else:
        print("     Gradient does not penalize.")

    print(f"\n  2. Marginalized Partial Cross-Entropy Loss (Candidate Set S={{0, 1}}):")
    print(f"     Loss = {loss_marginal.item():.4f}")
    print(f"     dL/dz_pothole at unannotated pothole = {grad_pothole_marginal:+.6f}")
    if grad_pothole_marginal <= 0.0001:
        print("     [VERIFIED PASSED] Zero positive penalty! Model is NOT penalized for predicting pothole.")

    print(f"\n  3. Decoupled Partial Focal Tversky Loss (Independent Sigmoid):")
    print(f"     Loss = {loss_decoupled.item():.4f}")
    print(f"     dL/dz_pothole at unannotated pothole = {grad_pothole_decoupled:+.6f}")
    if abs(grad_pothole_decoupled) <= 0.0001:
        print("     [VERIFIED PASSED] Exact zero gradient! Class 1 is completely unpenalized.")

    return {
        'grad_pothole_naive': grad_pothole_naive,
        'grad_pothole_marginal': grad_pothole_marginal,
        'grad_pothole_decoupled': grad_pothole_decoupled,
    }

def run_comprehensive_unit_tests():
    print("\n" + "=" * 70)
    print("EXECUTING FORMAL UNIT TESTS (CASES A, B, C)")
    print("=" * 70, flush=True)

    marginal_fn = MarginalizedPartialCrossEntropyLoss()
    decoupled_fn = DecoupledPartialFocalTverskyLoss()
    
    results = {}

    # -------------------------------------------------------------
    # TEST CASE A: Crack-Only sample with unannotated pothole
    # -------------------------------------------------------------
    # Target: pixel 0=BG, pixel 1=Crack, pixel 2=unannotated pothole (labeled 0)
    targets_a = torch.tensor([[0, 2, 0]], dtype=torch.long)
    validity_a = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float)
    logits_a = torch.tensor([[[2.0, 0.0, 0.0],
                              [0.0, 0.0, 3.0],
                              [0.0, 2.0, 0.0]]], dtype=torch.float, requires_grad=True)
    
    loss_a = marginal_fn(logits_a.unsqueeze(2), targets_a.unsqueeze(1), validity_a)
    loss_a.backward()
    grad_pothole_a = logits_a.grad[0, 1, 2].item()
    # At pixel 2, pothole prediction must NOT receive positive gradient penalty
    test_a_passed = (grad_pothole_a <= 0.0001)
    results['Case_A'] = {
        'description': 'Crack-only with unannotated pothole',
        'pothole_grad': grad_pothole_a,
        'passed': test_a_passed
    }
    print(f"Test Case A (Crack-only, unannotated pothole): {'PASSED' if test_a_passed else 'FAILED'} (grad={grad_pothole_a:+.6f})")

    # -------------------------------------------------------------
    # TEST CASE B: Pothole-Only sample with unannotated crack
    # -------------------------------------------------------------
    # Target: pixel 0=BG, pixel 1=Pothole, pixel 2=unannotated crack (labeled 0)
    targets_b = torch.tensor([[0, 1, 0]], dtype=torch.long)
    validity_b = torch.tensor([[1.0, 1.0, 0.0]], dtype=torch.float)
    logits_b = torch.tensor([[[2.0, 0.0, 0.0],
                              [0.0, 2.0, 0.0],
                              [0.0, 0.0, 3.0]]], dtype=torch.float, requires_grad=True)
    
    loss_b = marginal_fn(logits_b.unsqueeze(2), targets_b.unsqueeze(1), validity_b)
    loss_b.backward()
    grad_crack_b = logits_b.grad[0, 2, 2].item()
    test_b_passed = (grad_crack_b <= 0.0001)
    results['Case_B'] = {
        'description': 'Pothole-only with unannotated crack',
        'crack_grad': grad_crack_b,
        'passed': test_b_passed
    }
    print(f"Test Case B (Pothole-only, unannotated crack): {'PASSED' if test_b_passed else 'FAILED'} (grad={grad_crack_b:+.6f})")

    # -------------------------------------------------------------
    # TEST CASE C: Fully annotated 3-class sample
    # -------------------------------------------------------------
    # Target: pixel 0=BG, pixel 1=Pothole, pixel 2=Crack
    targets_c = torch.tensor([[0, 1, 2]], dtype=torch.long)
    validity_c = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float)
    # Model predicts wrong on pixel 2 (predicts BG instead of Crack)
    logits_c = torch.tensor([[[2.0, 0.0, 2.0],
                              [0.0, 2.0, 0.0],
                              [0.0, 0.0, 0.0]]], dtype=torch.float, requires_grad=True)
    
    loss_c = marginal_fn(logits_c.unsqueeze(2), targets_c.unsqueeze(1), validity_c)
    loss_c.backward()
    # At pixel 2, crack is true positive target, logit z_2 is too low, so gradient dL/dz_2 must be NEGATIVE (to pull z_2 up)
    grad_crack_c = logits_c.grad[0, 2, 2].item()
    test_c_passed = (grad_crack_c < -0.1)
    results['Case_C'] = {
        'description': 'Fully annotated 3-class with underpredicted crack',
        'crack_grad': grad_crack_c,
        'passed': test_c_passed
    }
    print(f"Test Case C (Fully annotated 3-class): {'PASSED' if test_c_passed else 'FAILED'} (grad={grad_crack_c:+.6f})")

    return results

if __name__ == "__main__":
    coupling_results = run_coupling_demonstration()
    unit_test_results = run_comprehensive_unit_tests()
    
    # Save results to scratch
    import json
    with open(r"D:\ROADVISION AI\scratch\partial_label_audit_results.json", "w") as f:
        json.dump({
            'coupling': coupling_results,
            'unit_tests': unit_test_results
        }, f, indent=2)
    print("\nAll partial-label tests completed and saved to scratch/partial_label_audit_results.json.", flush=True)
