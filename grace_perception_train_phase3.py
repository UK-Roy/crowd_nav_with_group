"""
GRAM-v2 Phase 3 — Slot Attention group pooling.

Architecture
------------
  Phase 2 GroupDetector (frozen)  →  g : (B, N, 64)  GNN-refined embeddings
  SlotAttention (trained here)    →  slots : (B, K=3, 64)  group prototypes
                                     attn  : (B, K, N)      slot assignment weights

Loss
----
  L_total = L_ca + λ_div * L_div
    L_ca  : co-assignment BCE  — same-group pairs should share a slot
    L_div : slot diversity (entropy) — prevents all humans collapsing into one slot

Success criterion
-----------------
  Hungarian-matched purity > 0.85 on the test set.
  Purity = fraction of grouped humans assigned to the correct (Hungarian-matched) slot.

Usage
-----
  # Standard: train against Phase 2 v1 checkpoint
  python grace_perception_train_phase3.py

  # With v2-trained Phase 2 checkpoint (no code changes needed):
  python grace_perception_train_phase3.py --phase2 trained_models/gram_v2/phase2_v2/best.pt

  # Quick smoke-test (5 epochs):
  python grace_perception_train_phase3.py --epochs 5 --no-cuda

Checkpoints
-----------
  trained_models/gram_v2/phase3/best.pt    ← highest val purity
  trained_models/gram_v2/phase3/last.pt
  trained_models/gram_v2/phase3/phase3_results.pt
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from crowd_nav.grace_perception.models import GroupDetector
from crowd_nav.grace_perception.slot_attention import (
    SlotAttention, co_assignment_loss, slot_diversity_loss, compute_purity)

# Reuse dataset helper from Phase 1
from grace_perception_train_phase1 import GroupPairDataset


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_purity(detector, slot_attn, loader, device):
    """
    Compute mean Hungarian-matched purity over the full dataset.
    Phase 2 detector runs in eval+no_grad mode; slot_attn also in eval mode.
    Returns mean purity over samples that have at least one grouped human.
    """
    detector.eval()
    slot_attn.eval()
    purities = []

    for feats, masks, labels in loader:
        feats_d = feats.to(device)
        masks_d = masks.to(device)

        _, _, g, _ = detector(feats_d, masks_d)
        _, attn    = slot_attn(g, masks_d)          # attn : (B, K, N)

        attn_np   = attn.cpu().numpy()
        labels_np = labels.numpy()
        masks_np  = masks.numpy().astype(bool)

        B = feats.shape[0]
        for b in range(B):
            p = compute_purity(attn_np[b], labels_np[b], masks_np[b])
            if p is not None:
                purities.append(p)

    return float(np.mean(purities)) if purities else 0.0


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(detector, slot_attn, loader, optimizer,
                pos_weight, lambda_div, device):
    """
    One training epoch.
    Detector is frozen (eval + no_grad); only slot_attn parameters are updated.
    """
    detector.eval()
    slot_attn.train()
    total_loss = total_ca = total_div = 0.0

    for feats, masks, labels in loader:
        feats  = feats.to(device)
        masks  = masks.to(device)
        labels = labels.to(device)

        # Phase 2 inference (frozen — no gradients needed here)
        with torch.no_grad():
            _, _, g, _ = detector(feats, masks)

        # Slot attention forward (gradients flow through this)
        _, attn = slot_attn(g, masks)                  # attn : (B, K, N)

        loss_ca  = co_assignment_loss(attn, labels, masks, pos_weight)
        loss_div = slot_diversity_loss(attn, masks)
        loss     = loss_ca + lambda_div * loss_div

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(slot_attn.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_ca   += loss_ca.item()
        total_div  += loss_div.item()

    n = len(loader)
    return total_loss / n, total_ca / n, total_div / n


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='GRAM-v2 Phase 3 — Slot Attention training')
    parser.add_argument('--data',       default='gram_v2_data',
                        help='Directory with train/val/test.npz')
    parser.add_argument('--phase2',     default='trained_models/gram_v2/phase2_v1/best.pt',
                        help='Phase 2 GroupDetector checkpoint (frozen backbone)')
    parser.add_argument('--save',       default='trained_models/gram_v2/phase3',
                        help='Output directory for checkpoints')
    parser.add_argument('--epochs',     type=int,   default=40)
    parser.add_argument('--batch',      type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--lambda-div', type=float, default=0.1,
                        help='Weight on slot diversity (entropy) loss')
    parser.add_argument('--K',          type=int,   default=3,
                        help='Number of slots (should match env num_groups)')
    parser.add_argument('--n-iters',    type=int,   default=3,
                        help='Slot attention iterations per forward pass')
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--no-cuda',    action='store_true')
    args = parser.parse_args()

    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    os.makedirs(args.save, exist_ok=True)

    print('=' * 60)
    print('  GRAM-v2 Phase 3 — Slot Attention')
    print(f'  K={args.K} slots  n_iters={args.n_iters}  λ_div={args.lambda_div}')
    print('=' * 60)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = GroupPairDataset(os.path.join(args.data, 'train.npz'))
    val_ds   = GroupPairDataset(os.path.join(args.data, 'val.npz'))
    test_ds  = GroupPairDataset(os.path.join(args.data, 'test.npz'))

    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              shuffle=True,  num_workers=args.workers,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch,
                              shuffle=False, num_workers=args.workers)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch,
                              shuffle=False, num_workers=args.workers)

    pw = train_ds.pos_weight()
    print(f'Positive-pair rate: {1/(pw+1):.3f}  →  pos_weight = {pw:.1f}')
    print(f'Train: {len(train_ds):,}   Val: {len(val_ds):,}   Test: {len(test_ds):,} samples')
    print(f'Device: {device}')

    # ── Phase 2 backbone (frozen) ─────────────────────────────────────────────
    if not os.path.exists(args.phase2):
        print(f'ERROR: Phase 2 checkpoint not found at {args.phase2}')
        sys.exit(1)

    detector = GroupDetector(n_gnn_layers=3).to(device)
    ckpt     = torch.load(args.phase2, map_location=device)
    detector.load_state_dict(ckpt['model_state'])
    detector.eval()
    for p in detector.parameters():
        p.requires_grad_(False)
    print(f'Phase 2 backbone loaded from {args.phase2}  (frozen)')

    # ── Slot attention ────────────────────────────────────────────────────────
    slot_attn = SlotAttention(K=args.K, n_iters=args.n_iters).to(device)
    n_params  = sum(p.numel() for p in slot_attn.parameters())
    print(f'SlotAttention params: {n_params:,}\n')

    optimizer = torch.optim.Adam(slot_attn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f'{"Epoch":>6}  {"Loss":>8}  {"L_ca":>8}  {"L_div":>8}  '
          f'{"valPurity":>10}')
    print('-' * 52)

    best_purity = 0.0
    best_epoch  = 0

    for epoch in range(1, args.epochs + 1):
        loss, l_ca, l_div = train_epoch(
            detector, slot_attn, train_loader, optimizer,
            pw, args.lambda_div, device)

        val_purity = evaluate_purity(detector, slot_attn, val_loader, device)
        scheduler.step()

        print(f'{epoch:>6}  {loss:>8.4f}  {l_ca:>8.4f}  {l_div:>8.4f}  '
              f'{val_purity:>10.4f}')

        if val_purity > best_purity:
            best_purity = val_purity
            best_epoch  = epoch
            torch.save({
                'epoch':       epoch,
                'model_state': slot_attn.state_dict(),
                'val_purity':  val_purity,
                'K':           args.K,
                'n_iters':     args.n_iters,
                'phase2_ckpt': args.phase2,
            }, os.path.join(args.save, 'best.pt'))

    torch.save({
        'epoch':       args.epochs,
        'model_state': slot_attn.state_dict(),
        'K':           args.K,
        'n_iters':     args.n_iters,
        'phase2_ckpt': args.phase2,
    }, os.path.join(args.save, 'last.pt'))

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f'\nBest val purity = {best_purity:.4f} at epoch {best_epoch}')
    print('Loading best checkpoint for test evaluation…')
    best_ckpt = torch.load(os.path.join(args.save, 'best.pt'), map_location=device)
    slot_attn.load_state_dict(best_ckpt['model_state'])

    test_purity = evaluate_purity(detector, slot_attn, test_loader, device)
    criterion   = '✅' if test_purity > 0.85 else '❌'

    print()
    print('=' * 60)
    print(f'  Test  Purity: {test_purity:.4f}   {criterion} (criterion: > 0.85)')
    print('=' * 60)

    torch.save({
        'test_purity': test_purity,
        'best_val_purity': best_purity,
        'best_epoch': best_epoch,
        'K': args.K,
        'n_iters': args.n_iters,
        'phase2_ckpt': args.phase2,
    }, os.path.join(args.save, 'phase3_results.pt'))

    print(f'\nCheckpoints saved to {args.save}/')


if __name__ == '__main__':
    main()
