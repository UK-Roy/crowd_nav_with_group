"""
GRAM-v2 — DBSCAN baseline evaluation.

Runs DBSCAN group detection on the same test dataset used for GRAM-v2
and reports pairwise F1 and ARI so results are directly comparable.

Two DBSCAN modes:
  position  — clusters on (p_rel_x, p_rel_y) only
  pos+vel   — clusters on (p_rel_x, p_rel_y, v_rel_x, v_rel_y), velocity
              scaled by vel_scale to balance units

eps is swept on the val set; optimal eps is applied to the test set.

Also loads GRAM-v2 Phase 1 and Phase 2 results (if checkpoints exist)
and prints a single comparison table.

Usage:
  python gram_v2_eval_dbscan.py
  python gram_v2_eval_dbscan.py --data gram_v2_data_v2   # v2 dataset
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import f1_score, adjusted_rand_score, precision_score, recall_score
    from scipy.sparse.csgraph import connected_components
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: sklearn and scipy are required. pip install scikit-learn scipy")
    sys.exit(1)


FEAT_DIM   = 7
MAX_HUMANS = 20


# ── DBSCAN helpers ────────────────────────────────────────────────────────────

def dbscan_predict(feats_sample: np.ndarray, mask: np.ndarray,
                   eps: float, mode: str, vel_scale: float) -> np.ndarray:
    """
    Run DBSCAN on one sample and return (MAX_HUMANS, MAX_HUMANS) predicted pair matrix.
    feats_sample : (20, 21)  full feature vector
    mask         : (20,)     bool, True = visible
    Returns pred : (20, 20)  1.0 if pair predicted in same group, else 0.0
    """
    vis_idx = np.where(mask)[0]
    pred = np.zeros((MAX_HUMANS, MAX_HUMANS), dtype=np.float32)
    if len(vis_idx) < 2:
        return pred

    curr = feats_sample[vis_idx, -FEAT_DIM:]          # (n_vis, 7) — current frame
    pos  = curr[:, :2]                                 # p_rel_x, p_rel_y

    if mode == 'pos+vel':
        vel = curr[:, 2:4] * vel_scale
        X   = np.concatenate([pos, vel], axis=1)
    else:
        X = pos

    labels = DBSCAN(eps=eps, min_samples=2).fit_predict(X)

    for a, i in enumerate(vis_idx):
        for b, j in enumerate(vis_idx):
            if i == j:
                continue
            li, lj = labels[a], labels[b]
            if li != -1 and li == lj:
                pred[i, j] = 1.0

    return pred


def evaluate_dbscan(feats: np.ndarray, masks: np.ndarray, labels_gt: np.ndarray,
                    eps: float, mode: str, vel_scale: float) -> dict:
    """Evaluate DBSCAN over all samples. Returns F1, precision, recall, ARI."""
    all_pred, all_gt = [], []
    ari_scores = []

    for t in range(len(feats)):
        mask = masks[t]
        pred = dbscan_predict(feats[t], mask, eps, mode, vel_scale)

        vis = mask.astype(bool)
        eye = np.eye(MAX_HUMANS, dtype=bool)
        valid = (vis[:, None] & vis[None, :]) & ~eye

        all_pred.extend(pred[valid].tolist())
        all_gt.extend(labels_gt[t][valid].tolist())

        # ARI: per-human cluster labels via connected components
        vis_idx = np.where(vis)[0]
        if len(vis_idx) < 2:
            continue
        gt_vis  = labels_gt[t][np.ix_(vis_idx, vis_idx)]
        pred_vis = pred[np.ix_(vis_idx, vis_idx)]

        _, gt_lbl   = connected_components(gt_vis,   directed=False, connection='weak')
        _, pred_lbl = connected_components(pred_vis, directed=False, connection='weak')

        if len(np.unique(gt_lbl)) >= 2:
            ari_scores.append(adjusted_rand_score(gt_lbl, pred_lbl))

    all_gt   = np.array(all_gt,   dtype=int)
    all_pred = np.array(all_pred, dtype=int)

    f1  = f1_score(all_gt, all_pred, zero_division=0)
    pre = precision_score(all_gt, all_pred, zero_division=0)
    rec = recall_score(all_gt, all_pred, zero_division=0)
    ari = float(np.mean(ari_scores)) if ari_scores else 0.0

    return {'f1': f1, 'precision': pre, 'recall': rec, 'ari': ari}


def sweep_eps(feats, masks, labels_gt, mode, vel_scale,
              eps_values=None) -> tuple:
    """Sweep eps on val set; return (best_eps, best_f1)."""
    if eps_values is None:
        eps_values = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.8, 2.0, 2.5, 3.0]
    best_eps, best_f1 = 1.0, 0.0
    for eps in eps_values:
        m = evaluate_dbscan(feats, masks, labels_gt, eps, mode, vel_scale)
        if m['f1'] > best_f1:
            best_f1, best_eps = m['f1'], eps
    return best_eps, best_f1


# ── Load GRAM-v2 saved results ─────────────────────────────────────────────────

def load_gram_results(phase1_dir: str, phase2_dir: str) -> dict:
    """Load saved phase1 and phase2 test results if available."""
    results = {}
    try:
        import torch
        for variant in ['A', 'B', 'C']:
            path = os.path.join(phase1_dir, variant, 'phase1_results.pt')
            if os.path.exists(path):
                r = torch.load(path, map_location='cpu')
                results[f'GRAM-v2 Phase1 Var-{variant}'] = r['test_metrics']
        p2 = os.path.join(phase2_dir, 'phase2_results.pt')
        if os.path.exists(p2):
            r = torch.load(p2, map_location='cpu')
            results['GRAM-v2 Phase2 (GNN)'] = r['test_metrics']
    except ImportError:
        pass
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default='gram_v2_data')
    parser.add_argument('--phase1',    default='trained_models/gram_v2/phase1')
    parser.add_argument('--phase2',    default='trained_models/gram_v2/phase2')
    parser.add_argument('--vel-scale', type=float, default=0.5,
                        help='Scale factor for velocity features in pos+vel mode '
                             '(balances position/velocity units)')
    args = parser.parse_args()

    print("Loading data…")
    val_data  = np.load(os.path.join(args.data, 'val.npz'))
    test_data = np.load(os.path.join(args.data, 'test.npz'))

    val_feats,  val_masks,  val_labels  = (val_data['feats'],
                                           val_data['masks'].astype(bool),
                                           val_data['labels'])
    test_feats, test_masks, test_labels = (test_data['feats'],
                                           test_data['masks'].astype(bool),
                                           test_data['labels'])

    print(f"Val:  {len(val_feats):,} samples   Test: {len(test_feats):,} samples\n")

    dbscan_results = {}

    for mode in ['position', 'pos+vel']:
        print(f"── DBSCAN ({mode}) — sweeping eps on val set…")
        best_eps, best_val_f1 = sweep_eps(val_feats, val_masks, val_labels,
                                          mode, args.vel_scale)
        print(f"   Best eps = {best_eps:.2f}  (val F1 = {best_val_f1:.3f})")

        m = evaluate_dbscan(test_feats, test_masks, test_labels,
                            best_eps, mode, args.vel_scale)
        m['best_eps'] = best_eps
        dbscan_results[f'DBSCAN ({mode})'] = m
        print(f"   Test  →  F1={m['f1']:.3f}  P={m['precision']:.3f}  "
              f"R={m['recall']:.3f}  ARI={m['ari']:.3f}\n")

    # ── Load GRAM-v2 results ──────────────────────────────────────────────────
    gram_results = load_gram_results(args.phase1, args.phase2)

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "="*72)
    print("  GROUP DETECTION — METHOD COMPARISON (test set)")
    print("="*72)
    header = f"  {'Method':<35}  {'F1':>6}  {'Prec':>6}  {'Rec':>6}  {'ARI':>6}"
    print(header)
    print("  " + "-" * 68)

    # DBSCAN rows
    for name, m in dbscan_results.items():
        eps_str = f"  [eps={m['best_eps']:.1f}]"
        print(f"  {name+eps_str:<35}  {m['f1']:6.3f}  {m['precision']:6.3f}  "
              f"{m['recall']:6.3f}  {m['ari']:6.3f}")

    if gram_results:
        print("  " + "·" * 68)
    for name, m in gram_results.items():
        f1  = m.get('f1',  0)
        pre = m.get('precision', 0)
        rec = m.get('recall', 0)
        ari = m.get('ari', 0)
        print(f"  {name:<35}  {f1:6.3f}  {pre:6.3f}  {rec:6.3f}  {ari:6.3f}")

    print("="*72)
    print("\nNote: DBSCAN has no AUROC (hard assignment, not probabilistic).")
    print("      ARI measures whole-scene cluster quality; F1 measures pairwise.")


if __name__ == '__main__':
    main()
