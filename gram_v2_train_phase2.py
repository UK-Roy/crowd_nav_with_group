"""
GRAM-v2 Phase 2 — offline training of GroupDetectorV2 (Encoder + EdgeNet + GNN).

Loss: L_total = L_group(W_final) + λ_aux * L_group(W0)
  W_final = post-GNN prediction (main target)
  W0      = pre-GNN prediction  (auxiliary; keeps encoder grounded)

Success criterion: test F1 >= 0.90  AND  test ARI > 0.70

Usage:
  # With Phase 1 pretrained weights (recommended):
  python gram_v2_train_phase2.py --phase1 trained_models/gram_v2/phase1/best.pt

  # From random init (slower convergence but valid):
  python gram_v2_train_phase2.py

  # Quick test run:
  python gram_v2_train_phase2.py --epochs 5 --phase1 trained_models/gram_v2/phase1/best.pt

Checkpoints:
  trained_models/gram_v2/phase2/best.pt
  trained_models/gram_v2/phase2/last.pt
  trained_models/gram_v2/phase2/phase2_results.pt
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from crowd_nav.gram_v2.models import GroupDetector

# Reuse dataset and helpers from Phase 1 training
from gram_v2_train_phase1 import (GroupPairDataset, weighted_bce,
                                   pair_valid_mask, evaluate,
                                   find_best_threshold)

try:
    from scipy.sparse.csgraph import connected_components
    from sklearn.metrics import adjusted_rand_score, f1_score
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy/sklearn not found — ARI metric will be skipped.")


# ── ARI evaluation ────────────────────────────────────────────────────────────

def labels_from_pair_matrix(pair_gt: np.ndarray) -> np.ndarray:
    """
    Reconstruct per-human group labels from symmetric pair GT matrix.
    Uses connected components: connected humans share a label.
    Isolated humans (no positive pairs) get unique singleton labels.
    Returns: (N,) int array
    """
    N = len(pair_gt)
    # Connected components on the binary GT adjacency
    _, labels = connected_components(pair_gt, directed=False, connection='weak')
    return labels


def compute_batch_ari(W_pred: np.ndarray, labels_gt: np.ndarray,
                      masks: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute mean ARI over a batch.
    W_pred    : (B, N, N)  predicted probabilities
    labels_gt : (B, N, N)  GT pair matrix (symmetric binary)
    masks     : (B, N)     bool
    """
    if not HAS_SCIPY:
        return 0.0
    aris = []
    B = len(W_pred)
    for b in range(B):
        vis = masks[b].astype(bool)
        if vis.sum() < 2:
            continue
        # GT labels for visible humans
        gt_pairs_vis = labels_gt[b][np.ix_(vis, vis)]
        gt_labels    = labels_from_pair_matrix(gt_pairs_vis)

        # Predicted labels via connected components on thresholded W
        pred_adj_vis = (W_pred[b][np.ix_(vis, vis)] > threshold).astype(float)
        _, pred_labels = connected_components(pred_adj_vis, directed=False,
                                              connection='weak')

        if len(np.unique(gt_labels)) < 2:
            continue   # all same label or all singletons — ARI undefined
        aris.append(adjusted_rand_score(gt_labels, pred_labels))

    return float(np.mean(aris)) if aris else 0.0


@torch.no_grad()
def evaluate_v2(model, loader, device: str, threshold: float = 0.5) -> dict:
    """Evaluate W_final: F1, AUROC, ARI."""
    model.eval()
    all_probs, all_preds, all_gt = [], [], []
    ari_scores = []

    for feats, masks, labels in loader:
        feats, masks_gpu = feats.to(device), masks.to(device)
        W_final, _, _, _ = model(feats, masks_gpu)

        pv = pair_valid_mask(masks_gpu, device).bool()
        probs = W_final[pv].cpu().numpy()
        preds = (probs >= threshold).astype(int)
        gt    = labels.to(device)[pv].cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_gt.extend(gt.tolist())

        # ARI over this batch
        ari = compute_batch_ari(
            W_final.cpu().numpy(),
            labels.numpy(),
            masks.numpy(),
            threshold
        )
        if ari > 0:
            ari_scores.append(ari)

    all_gt    = np.array(all_gt)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    if HAS_SCIPY:
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        f1  = f1_score(all_gt, all_preds, zero_division=0)
        pre = precision_score(all_gt, all_preds, zero_division=0)
        rec = recall_score(all_gt, all_preds, zero_division=0)
        auc = roc_auc_score(all_gt, all_probs) if len(np.unique(all_gt)) > 1 else 0.0
    else:
        tp  = ((all_preds == 1) & (all_gt == 1)).sum()
        fp  = ((all_preds == 1) & (all_gt == 0)).sum()
        fn  = ((all_preds == 0) & (all_gt == 1)).sum()
        pre = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1  = 2 * pre * rec / (pre + rec + 1e-6)
        auc = 0.0

    ari = float(np.mean(ari_scores)) if ari_scores else 0.0
    return {'f1': f1, 'precision': pre, 'recall': rec, 'auroc': auc, 'ari': ari}


# ── Train loop ────────────────────────────────────────────────────────────────

def train_epoch_v2(model, loader, optimizer, pos_weight: float,
                   lambda_aux: float, device: str) -> float:
    model.train()
    total_loss = 0.0
    for feats, masks, labels in loader:
        feats, masks, labels = feats.to(device), masks.to(device), labels.to(device)
        pv = pair_valid_mask(masks, device)

        W_final, W0, _, _ = model(feats, masks)

        loss_main = weighted_bce(W_final, labels, pv, pos_weight)
        loss_aux  = weighted_bce(W0,      labels, pv, pos_weight)
        loss = loss_main + lambda_aux * loss_aux

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',       default='gram_v2_data')
    parser.add_argument('--save',       default='trained_models/gram_v2/phase2')
    parser.add_argument('--phase1',     default='trained_models/gram_v2/phase1/B/best.pt',
                        help='Phase 1 checkpoint to initialise encoder + edge_net')
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--batch',      type=int,   default=256)
    parser.add_argument('--lr',         type=float, default=5e-4)
    parser.add_argument('--lambda-aux', type=float, default=0.3,
                        help='Weight on W0 auxiliary loss')
    parser.add_argument('--gnn-layers', type=int,   default=3)
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--no-cuda',    action='store_true')
    args = parser.parse_args()

    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'
    os.makedirs(args.save, exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = GroupPairDataset(os.path.join(args.data, 'train.npz'))
    val_ds   = GroupPairDataset(os.path.join(args.data, 'val.npz'))
    test_ds  = GroupPairDataset(os.path.join(args.data, 'test.npz'))

    pw = train_ds.pos_weight()
    print(f"pos_weight = {pw:.1f}")
    print(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    print(f"Device: {device}\n")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers,
                              pin_memory=(device == 'cuda'))
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=2)
    test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                              num_workers=2)

    # ── Model ────────────────────────────────────────────────────────────────
    model = GroupDetector(n_gnn_layers=args.gnn_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"GroupDetectorV2  params: {n_params:,}")

    # Load Phase 1 weights into encoder + edge_net
    if os.path.exists(args.phase1):
        ckpt = torch.load(args.phase1, map_location=device)
        # Phase 1 and Phase 2 use the same GroupDetector class.
        # encoder.* and edge_net.* keys match directly; gnn.* keys are missing (new).
        result = model.load_state_dict(ckpt['model_state'], strict=False)
        print(f"Loaded Phase 1 weights from {args.phase1}")
        print(f"  Missing (GNN, expected): {result.missing_keys}")
    else:
        print(f"Phase 1 checkpoint not found at {args.phase1} — training from random init")

    print()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_f1 = 0.0
    header = (f"{'Epoch':>5}  {'Loss':>8}  {'valF1':>7}  {'valAUC':>7}  "
              f"{'valARI':>7}  {'P':>6}  {'R':>6}")
    print(header)
    print('-' * len(header))

    for epoch in range(1, args.epochs + 1):
        train_loss  = train_epoch_v2(model, train_loader, optimizer,
                                     pw, args.lambda_aux, device)
        val_metrics = evaluate_v2(model, val_loader, device)
        scheduler.step()

        print(f"{epoch:5d}  {train_loss:8.4f}  "
              f"{val_metrics['f1']:7.3f}  {val_metrics['auroc']:7.3f}  "
              f"{val_metrics['ari']:7.3f}  "
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

    # ── Final test ────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint (val F1={best_val_f1:.3f})…")
    ckpt = torch.load(os.path.join(args.save, 'best.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state'])

    # Find optimal threshold on val, apply to test
    from gram_v2_train_phase1 import evaluate as evaluate_p1
    val_m  = evaluate_p1(model, val_loader, device, variant='B')
    best_t, _ = find_best_threshold(val_m['probs'], val_m['gt'])
    print(f"Optimal threshold (val): {best_t:.2f}")

    test_m = evaluate_v2(model, test_loader, device, threshold=best_t)

    print(f"\nTEST RESULTS (Phase 2, threshold={best_t:.2f})")
    print(f"  F1        = {test_m['f1']:.4f}   (criterion: >= 0.90)")
    print(f"  ARI       = {test_m['ari']:.4f}   (criterion: >  0.70)")
    print(f"  AUROC     = {test_m['auroc']:.4f}")
    print(f"  Precision = {test_m['precision']:.4f}")
    print(f"  Recall    = {test_m['recall']:.4f}")

    f1_ok  = test_m['f1']  >= 0.90
    ari_ok = test_m['ari'] >  0.70
    if f1_ok and ari_ok:
        verdict = 'PASSED ✓  → proceed to Phase 3 (slot attention)'
    elif f1_ok:
        verdict = 'F1 PASSED but ARI too low — check GNN layers or threshold'
    elif ari_ok:
        verdict = 'ARI PASSED but F1 too low — check pos_weight or epochs'
    else:
        verdict = 'NOT YET — tune and retry'
    print(f"\nPhase 2 criteria: {verdict}")

    torch.save({'test_metrics': test_m, 'best_val_f1': best_val_f1,
                'best_threshold': best_t},
               os.path.join(args.save, 'phase2_results.pt'))


if __name__ == '__main__':
    main()
