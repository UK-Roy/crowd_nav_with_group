"""
GRAM-v2 Phase 1 — offline training of group detector.

Trains PedestrianEncoder + PairwiseEdgeNetwork on collected rollout data
using a weighted binary cross-entropy loss (L_group).

Success criterion: test F1 >= 0.85  (see GRAM_V2_DESIGN.md, Phase 1)

Usage:
  python gram_v2_train_phase1.py
  python gram_v2_train_phase1.py --data gram_v2_data --epochs 60 --batch 256

Checkpoints:
  trained_models/gram_v2/phase1/best.pt   — best val-F1 checkpoint
  trained_models/gram_v2/phase1/last.pt   — final epoch checkpoint
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from crowd_nav.gram_v2.models import GroupDetector
# Phase 1 uses n_gnn_layers=0 (Encoder + EdgeNet only)

try:
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not found. Metrics will be basic.")


# ── Dataset ───────────────────────────────────────────────────────────────────

class GroupPairDataset(Dataset):
    def __init__(self, path: str):
        data = np.load(path)
        self.feats  = torch.from_numpy(data['feats'])           # (T, 20, 7)
        self.masks  = torch.from_numpy(data['masks'].astype(np.float32)).bool()  # (T, 20)
        self.labels = torch.from_numpy(data['labels'])          # (T, 20, 20)

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        return self.feats[idx], self.masks[idx], self.labels[idx]

    def pos_weight(self) -> float:
        """Compute recommended pos_weight = neg_pairs / pos_pairs."""
        total_pos = self.labels.sum().item()
        total = (self.masks.unsqueeze(2).float() *
                 self.masks.unsqueeze(1).float()).sum().item()
        # Subtract self-loops (diagonal: T * N)
        total -= len(self) * self.masks.shape[1]
        total_neg = total - total_pos
        return total_neg / (total_pos + 1e-6)


# ── Loss ──────────────────────────────────────────────────────────────────────

def weighted_bce(W: torch.Tensor, labels: torch.Tensor,
                 pair_valid: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """
    Weighted BCE on valid pairs only.
    W, labels, pair_valid: (B, N, N)
    """
    W = W.clamp(1e-6, 1.0 - 1e-6)
    bce = -(pos_weight * labels * torch.log(W) +
            (1.0 - labels) * torch.log(1.0 - W))
    loss = (bce * pair_valid).sum() / (pair_valid.sum() + 1e-6)
    return loss


def pair_valid_mask(masks: torch.Tensor, device) -> torch.Tensor:
    """Return (B, N, N) float mask: 1 if both humans visible and i != j."""
    B, N = masks.shape
    vi = masks.unsqueeze(2).float()
    vj = masks.unsqueeze(1).float()
    pv = vi * vj
    eye = torch.eye(N, device=device).unsqueeze(0)
    return pv * (1.0 - eye)


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, pos_weight: float, device: str):
    model.train()
    total_loss = 0.0
    for feats, masks, labels in loader:
        feats, masks, labels = feats.to(device), masks.to(device), labels.to(device)
        pv = pair_valid_mask(masks, device)

        W, _, _, _ = model(feats, masks)
        loss = weighted_bce(W, labels, pv, pos_weight)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device: str, threshold: float = 0.5) -> dict:
    model.eval()
    all_probs, all_preds, all_gt = [], [], []

    for feats, masks, labels in loader:
        feats, masks = feats.to(device), masks.to(device)
        W, _, _, _ = model(feats, masks)           # (B, 20, 20)

        pv = pair_valid_mask(masks, device).bool()  # (B, 20, 20)
        probs = W[pv].cpu().numpy()
        preds = (probs >= threshold).astype(int)
        gt    = labels.to(device)[pv].cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_gt.extend(gt.tolist())

    all_gt    = np.array(all_gt)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if HAS_SKLEARN:
        f1  = f1_score(all_gt, all_preds, zero_division=0)
        pre = precision_score(all_gt, all_preds, zero_division=0)
        rec = recall_score(all_gt, all_preds, zero_division=0)
        auc = (roc_auc_score(all_gt, all_probs)
               if len(np.unique(all_gt)) > 1 else 0.0)
    else:
        tp = ((all_preds == 1) & (all_gt == 1)).sum()
        fp = ((all_preds == 1) & (all_gt == 0)).sum()
        fn = ((all_preds == 0) & (all_gt == 1)).sum()
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1  = 2 * pre * rec / (pre + rec + 1e-6)
        auc = 0.0

    return {'f1': f1, 'precision': pre, 'recall': rec, 'auroc': auc,
            'probs': all_probs, 'gt': all_gt}


def find_best_threshold(probs: np.ndarray, gt: np.ndarray) -> tuple:
    """Sweep thresholds [0.1, 0.9] and return (best_threshold, best_f1)."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.91, 0.05):
        preds = (probs >= t).astype(int)
        if HAS_SKLEARN:
            f1 = f1_score(gt, preds, zero_division=0)
        else:
            tp = ((preds == 1) & (gt == 1)).sum()
            fp = ((preds == 1) & (gt == 0)).sum()
            fn = ((preds == 0) & (gt == 1)).sum()
            p  = tp / (tp + fp + 1e-6)
            r  = tp / (tp + fn + 1e-6)
            f1 = 2 * p * r / (p + r + 1e-6)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',    default='gram_v2_data')
    parser.add_argument('--save',    default='trained_models/gram_v2/phase1')
    parser.add_argument('--epochs',  type=int,   default=60)
    parser.add_argument('--batch',   type=int,   default=256)
    parser.add_argument('--lr',      type=float, default=1e-3)
    parser.add_argument('--workers', type=int,   default=4)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    os.makedirs(args.save, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = GroupPairDataset(os.path.join(args.data, 'train.npz'))
    val_ds   = GroupPairDataset(os.path.join(args.data, 'val.npz'))
    test_ds  = GroupPairDataset(os.path.join(args.data, 'test.npz'))

    pw = train_ds.pos_weight()
    print(f"Positive-pair rate: {1/(pw+1):.3f}  →  pos_weight = {pw:.1f}")
    print(f"Train: {len(train_ds):,}   Val: {len(val_ds):,}   Test: {len(test_ds):,} samples")
    print(f"Device: {device}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False,
                              num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    model = GroupDetector(n_gnn_layers=0).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GroupDetector  params: {n_params:,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    header = f"{'Epoch':>5}  {'Loss':>8}  {'valF1':>7}  {'valAUC':>7}  {'P':>6}  {'R':>6}"
    print(header)
    print('-' * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss   = train_epoch(model, train_loader, optimizer, pw, device)
        val_metrics  = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"{epoch:5d}  {train_loss:8.4f}  "
              f"{val_metrics['f1']:7.3f}  {val_metrics['auroc']:7.3f}  "
              f"{val_metrics['precision']:6.3f}  {val_metrics['recall']:6.3f}",
              end='')

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'val_metrics': val_metrics, 'pos_weight': pw},
                       os.path.join(args.save, 'best.pt'))
            print('  ← best', end='')
        print()

    torch.save({'epoch': args.epochs, 'model_state': model.state_dict(),
                'val_metrics': val_metrics},
               os.path.join(args.save, 'last.pt'))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val F1={best_val_f1:.3f})…")
    ckpt = torch.load(os.path.join(args.save, 'best.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # Find optimal threshold on val set, apply to test set
    val_m  = evaluate(model, val_loader, device)
    best_t, _ = find_best_threshold(val_m['probs'], val_m['gt'])
    print(f"Optimal threshold (val): {best_t:.2f}")

    test_m = evaluate(model, test_loader, device, threshold=best_t)

    print(f"\nTEST RESULTS  (threshold={best_t:.2f})")
    print(f"  F1        = {test_m['f1']:.4f}   (criterion: >= 0.85)")
    print(f"  AUROC     = {test_m['auroc']:.4f}")
    print(f"  Precision = {test_m['precision']:.4f}")
    print(f"  Recall    = {test_m['recall']:.4f}")

    passed = test_m['f1'] >= 0.85
    status = 'PASSED ✓  → proceed to Phase 2' if passed else \
             'NOT YET — tune pos_weight, epochs, or architecture'
    print(f"\nPhase 1 criterion (test F1 >= 0.85):  {status}")

    torch.save({'test_metrics': test_m, 'best_val_f1': best_val_f1,
                'best_threshold': best_t},
               os.path.join(args.save, 'phase1_results.pt'))


if __name__ == '__main__':
    main()
