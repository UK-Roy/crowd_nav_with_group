"""
eval_detection_comparison.py — Group detection: DBSCAN vs GRACE GroupDetector.

Evaluates on the test split of gram_v2_data and prints a comparison table
in both ASCII and LaTeX formats suitable for a CoRL paper.

Methods evaluated:
  1. DBSCAN (position only)         — classical baseline
  2. DBSCAN (position + velocity)   — classical + velocity
  3. GroupDetector Phase1-B (enc.)  — our encoder only (no GNN)
  4. GroupDetector Phase2   (GNN)   — our full backbone (= GRACE perception)

Usage:
  # Full comparison — auto eps sweep + model eval (main paper table):
  python eval_detection_comparison.py

  # Run DBSCAN for ONE specific eps value (fill in one appendix table row):
  python eval_detection_comparison.py --fixed-eps 1.0 --dbscan-only
  python eval_detection_comparison.py --fixed-eps 0.5 --dbscan-only
  python eval_detection_comparison.py --fixed-eps 0.8 --mode pos+vel --dbscan-only

  # Full DBSCAN eps sweep table (all eps, both modes — for appendix):
  python eval_detection_comparison.py --eps-sweep-only

  # Re-run model inference from scratch (ignores saved results.pt):
  python eval_detection_comparison.py --force-eval

  # Save fresh results.pt alongside the checkpoints:
  python eval_detection_comparison.py --force-eval --save-results

Output:
  ASCII comparison table (stdout)
  LaTeX table snippet (stdout, unless --no-latex)
  Optional: updated phase1_results.pt / phase2_results.pt
"""

import os, sys, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("WARNING: PyTorch not found — model evaluation disabled, DBSCAN only.")

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                  adjusted_rand_score, roc_auc_score)
    from scipy.sparse.csgraph import connected_components
    HAS_SKLEARN = True
except ImportError:
    print("ERROR: scikit-learn and scipy required.  pip install scikit-learn scipy")
    sys.exit(1)


MAX_HUMANS = 20
FEAT_DIM   = 7   # current-frame features (last 7 dims of 21-d vector)


# ─────────────────────────────────────────────────────────────────────────────
# DBSCAN helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fast_group_predict(pos_vis, eps):
    """
    Fast DBSCAN-equivalent with min_samples=2 via pairwise distance threshold.
    Uses connected components on the eps-neighborhood graph — O(n²) but no Python
    loop overhead beyond the distance computation.
    Returns labels array (n_vis,) where isolated humans get label -1.
    """
    if len(pos_vis) < 2:
        return np.full(len(pos_vis), -1, dtype=int)
    D = np.linalg.norm(pos_vis[:, None] - pos_vis[None, :], axis=2)  # (n, n)
    adj = (D <= eps)
    np.fill_diagonal(adj, False)
    is_core = adj.any(axis=1)      # has >= 1 neighbour ← DBSCAN min_samples=2
    reachable = adj & is_core[:, None]
    sym = reachable | reachable.T
    _, labels = connected_components(sym, directed=False, connection='weak')
    noise = ~is_core & ~(adj & is_core[None, :]).any(axis=1)
    labels = labels.astype(int)
    labels[noise] = -1
    return labels


def _dbscan_eval_batch(feats, masks, labels_gt, eps, mode, vel_scale):
    """
    Vectorised DBSCAN evaluation over all samples.
    Returns flat (all_pred, all_gt, ari_scores) lists.
    """
    all_pred, all_gt, ari_scores = [], [], []
    eye_full = np.eye(MAX_HUMANS, dtype=bool)

    for t in range(len(feats)):
        mask    = masks[t]
        vis_idx = np.where(mask)[0]
        valid   = (mask[:, None] & mask[None, :]) & ~eye_full

        if len(vis_idx) < 2:
            all_pred.extend([0] * int(valid.sum()))
            all_gt.extend(labels_gt[t][valid].astype(int).tolist())
            continue

        curr = feats[t][vis_idx, -FEAT_DIM:]
        if mode == 'pos+vel':
            X = np.concatenate([curr[:, :2], curr[:, 2:4] * vel_scale], axis=1)
        else:
            X = curr[:, :2]

        db_labels = _fast_group_predict(X[:, :2] if mode != 'pos+vel' else X, eps)

        same = ((db_labels[:, None] == db_labels[None, :])
                & (db_labels[:, None] != -1))
        np.fill_diagonal(same, False)

        pred = np.zeros((MAX_HUMANS, MAX_HUMANS), dtype=np.float32)
        pred[np.ix_(vis_idx, vis_idx)] = same.astype(np.float32)

        all_pred.extend(pred[valid].tolist())
        all_gt.extend(labels_gt[t][valid].astype(int).tolist())

        gt_sub = labels_gt[t][np.ix_(vis_idx, vis_idx)]
        _, gt_lbl   = connected_components(gt_sub, directed=False, connection='weak')
        _, pred_lbl = connected_components(same.astype(float), directed=False, connection='weak')
        if len(np.unique(gt_lbl)) >= 2:
            ari_scores.append(adjusted_rand_score(gt_lbl, pred_lbl))

    return all_pred, all_gt, ari_scores


def _evaluate_dbscan(feats, masks, labels_gt, eps, mode, vel_scale):
    all_pred, all_gt, ari_scores = _dbscan_eval_batch(feats, masks, labels_gt,
                                                       eps, mode, vel_scale)
    all_gt   = np.array(all_gt,   dtype=int)
    all_pred = np.array(all_pred, dtype=int)
    return {
        'f1':        f1_score(all_gt, all_pred, zero_division=0),
        'precision': precision_score(all_gt, all_pred, zero_division=0),
        'recall':    recall_score(all_gt, all_pred, zero_division=0),
        'ari':       float(np.mean(ari_scores)) if ari_scores else 0.0,
        'auroc':     None,
    }


def _sweep_eps(feats, masks, labels_gt, mode, vel_scale,
               eps_list=None, n_sweep=300):
    """Sweep eps on a fixed subsample of the val set."""
    if eps_list is None:
        eps_list = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
    if n_sweep and len(feats) > n_sweep:
        rng  = np.random.default_rng(42)
        idx  = rng.choice(len(feats), n_sweep, replace=False)
        feats, masks, labels_gt = feats[idx], masks[idx], labels_gt[idx]
    best_eps, best_f1 = 1.0, 0.0
    for eps in eps_list:
        m = _evaluate_dbscan(feats, masks, labels_gt, eps, mode, vel_scale)
        if m['f1'] > best_f1:
            best_f1, best_eps = m['f1'], eps
    return best_eps


# ─────────────────────────────────────────────────────────────────────────────
# Model eval helpers (Phase 1 & Phase 2)
# ─────────────────────────────────────────────────────────────────────────────

def _pair_valid_mask(masks_t, device):
    """(B, N, N) float: 1 if both visible and i≠j."""
    B, N = masks_t.shape
    pv  = masks_t.unsqueeze(2).float() * masks_t.unsqueeze(1).float()
    eye = torch.eye(N, device=device).unsqueeze(0)
    return pv * (1.0 - eye)


def _compute_batch_ari(W_np, labels_np, masks_np, threshold=0.5):
    """ARI averaged over a batch. Returns list of per-sample ARI values."""
    from scipy.sparse.csgraph import connected_components as cc
    ari_vals = []
    for b in range(len(W_np)):
        vis = masks_np[b].astype(bool)
        vis_idx = np.where(vis)[0]
        if len(vis_idx) < 2:
            continue
        gt  = labels_np[b][np.ix_(vis_idx, vis_idx)]
        pr  = (W_np[b][np.ix_(vis_idx, vis_idx)] >= threshold).astype(float)
        _, gt_lbl   = cc(gt, directed=False, connection='weak')
        _, pred_lbl = cc(pr, directed=False, connection='weak')
        if len(np.unique(gt_lbl)) >= 2:
            ari_vals.append(adjusted_rand_score(gt_lbl, pred_lbl))
    return ari_vals


@torch.no_grad()
def _eval_model(model, loader, device, threshold=0.5):
    """Evaluate a GroupDetector on a DataLoader. Returns metrics dict."""
    model.eval()
    all_probs, all_preds, all_gt, ari_scores = [], [], [], []

    for feats, masks, labels in loader:
        feats_g, masks_g = feats.to(device), masks.to(device)
        W_final, _, _, _ = model(feats_g, masks_g)

        pv    = _pair_valid_mask(masks_g, device).bool()
        probs = W_final[pv].cpu().numpy()
        preds = (probs >= threshold).astype(int)
        gt    = labels.to(device)[pv].cpu().numpy().astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_gt.extend(gt.tolist())

        ari_scores.extend(_compute_batch_ari(
            W_final.cpu().numpy(), labels.numpy(), masks.numpy(), threshold))

    all_gt    = np.array(all_gt)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    f1  = f1_score(all_gt, all_preds, zero_division=0)
    pre = precision_score(all_gt, all_preds, zero_division=0)
    rec = recall_score(all_gt, all_preds, zero_division=0)
    auc = roc_auc_score(all_gt, all_probs) if len(np.unique(all_gt)) > 1 else 0.0
    ari = float(np.mean(ari_scores)) if ari_scores else 0.0

    return {'f1': f1, 'precision': pre, 'recall': rec, 'auroc': auc, 'ari': ari}


def _load_phase1(ckpt_path, device):
    """Load GroupDetector in Phase1 mode (no GNN layers)."""
    from crowd_nav.gram_v2.models import GroupDetector
    ckpt     = torch.load(ckpt_path, map_location=device)
    use_pt   = ckpt.get('vcfg', {}).get('use_pairwise_temporal', False)
    model    = GroupDetector(n_gnn_layers=0, use_pairwise_temporal=use_pt).to(device)
    model.load_state_dict(ckpt['model_state'])
    return model


def _load_phase2(ckpt_path, device):
    """Load GroupDetector in Phase2 mode (with GNN layers)."""
    from crowd_nav.gram_v2.models import GroupDetector
    ckpt   = torch.load(ckpt_path, map_location=device)
    use_pt = ckpt.get('use_pairwise_temporal', False)
    # Infer number of GNN layers from state dict
    gnn_keys   = [k for k in ckpt['model_state'] if k.startswith('gnn_layers.')]
    n_gnn      = max((int(k.split('.')[1]) for k in gnn_keys), default=-1) + 1
    n_gnn      = max(n_gnn, 1)   # at least 1 if any gnn key exists
    model      = GroupDetector(n_gnn_layers=n_gnn, use_pairwise_temporal=use_pt).to(device)
    model.load_state_dict(ckpt['model_state'])
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Table printers
# ─────────────────────────────────────────────────────────────────────────────

def _record_result(txt_path, mode, eps, metrics):
    """
    Write one DBSCAN result into the TBD row of perception_detection_results.txt.
    Finds the correct section (position / pos+vel) and the matching eps line,
    then replaces 'TBD    TBD    TBD      TBD' with actual values.
    """
    import re
    if not os.path.exists(txt_path):
        print(f"[record] {txt_path} not found — skipping.")
        return

    with open(txt_path) as f:
        lines = f.readlines()

    pos_hdr    = '[DBSCAN — position only]'
    posvel_hdr = '[DBSCAN — position + velocity'
    ours_hdr   = '[Ours — GroupDetector'
    target_hdr = pos_hdr if mode == 'position' else posvel_hdr
    eps_str    = f"{eps:.1f}"

    f1  = f"{metrics['f1']:.3f}"
    pre = f"{metrics['precision']:.3f}"
    rec = f"{metrics['recall']:.3f}"
    ari = f"{metrics['ari']:.3f}"

    in_section = False
    updated    = False
    new_lines  = []

    for line in lines:
        stripped = line.rstrip('\n')
        if target_hdr in stripped:
            in_section = True
        elif (pos_hdr in stripped or posvel_hdr in stripped
              or ours_hdr in stripped) and target_hdr not in stripped:
            in_section = False

        if (in_section and not updated
                and re.search(rf'\beps\s*=\s*{re.escape(eps_str)}\b', stripped)
                and 'TBD' in stripped):
            stripped = re.sub(r'TBD\s+TBD\s+TBD\s+TBD',
                               f"{f1:<6} {pre:<6} {rec:<8} {ari}", stripped)
            updated = True

        new_lines.append(stripped + '\n')

    with open(txt_path, 'w') as f:
        f.writelines(new_lines)

    if updated:
        print(f"[record] {txt_path} updated: "
              f"DBSCAN {mode} eps={eps_str} → F1={f1} Prec={pre} Rec={rec} ARI={ari}")
    else:
        print(f"[record] Row already filled or not found: DBSCAN {mode} eps={eps_str}")


def _ascii_table(rows):
    """
    rows : list of (name, metrics_dict, note)
    metrics_dict keys: f1, precision, recall, ari, auroc
    """
    W = 75
    hdr = f"  {'Method':<38} {'F1':>5}  {'Prec':>5}  {'Rec':>5}  {'ARI':>5}  {'AUROC':>5}"
    sep = "  " + "─" * (W - 2)

    print()
    print("=" * W)
    print("  GROUP DETECTION COMPARISON  (test set)")
    print("=" * W)
    print(hdr)
    print(sep)

    prev_group = None
    for name, m, note, group in rows:
        if group != prev_group and prev_group is not None:
            print("  " + "·" * (W - 2))
        prev_group = group

        f1  = f"{m['f1']:.3f}"
        pre = f"{m['precision']:.3f}"
        rec = f"{m['recall']:.3f}"
        ari = f"{m['ari']:.3f}"
        auc = f"{m['auroc']:.3f}" if m['auroc'] is not None else "  —  "

        suffix = f"  ← {note}" if note else ""
        print(f"  {name:<38} {f1:>5}  {pre:>5}  {rec:>5}  {ari:>5}  {auc:>5}{suffix}")

    print("=" * W)
    print()
    print("  F1 / Prec / Rec: pairwise (same-group pair classification)")
    print("  ARI: Adjusted Rand Index over whole-scene cluster assignments")
    print("  AUROC: ranking quality (hard-assignment methods have no AUROC)")
    print()


def _latex_table(rows):
    """Print a LaTeX tabular snippet."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Group detection performance on the GRACE test set.}",
        r"\label{tab:group_detection}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & F1 & Precision & Recall & ARI \\",
        r"\midrule",
    ]

    prev_group = None
    for name, m, note, group in rows:
        if group != prev_group and prev_group is not None:
            lines.append(r"\midrule")
        prev_group = group
        f1  = f"{m['f1']:.3f}"
        pre = f"{m['precision']:.3f}"
        rec = f"{m['recall']:.3f}"
        ari = f"{m['ari']:.3f}"
        tex_name = name.replace("_", r"\_").replace("★", r"$\star$")
        lines.append(f"{tex_name} & {f1} & {pre} & {rec} & {ari} \\\\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]

    print("\n── LaTeX table ──────────────────────────────────────────────────────\n")
    print("\n".join(lines))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DBSCAN vs GRACE GroupDetector — group detection comparison")
    parser.add_argument('--data',         default='gram_v2_data',
                        help='Directory with train.npz / val.npz / test.npz')
    parser.add_argument('--phase1-ckpt',  default='trained_models/gram_v2/phase1_v2/B/best.pt',
                        help='Phase1 (encoder-only) checkpoint')
    parser.add_argument('--phase1-save',  default='trained_models/gram_v2/phase1_v2/B',
                        help='Directory where phase1_results.pt is/will be saved')
    parser.add_argument('--phase2-ckpt',  default='trained_models/gram_v2/phase2_v2/best.pt',
                        help='Phase2 (encoder+GNN) checkpoint — GRACE backbone')
    parser.add_argument('--phase2-save',  default='trained_models/gram_v2/phase2_v2',
                        help='Directory where phase2_results.pt is/will be saved')
    parser.add_argument('--vel-scale',    type=float, default=0.5,
                        help='Velocity scale for DBSCAN pos+vel mode')
    parser.add_argument('--n-sweep',      type=int,   default=300,
                        help='Val samples for eps sweep (0 = all, slow)')
    parser.add_argument('--n-test',       type=int,   default=2000,
                        help='Test samples for DBSCAN final eval (0 = all, slow)')
    parser.add_argument('--batch',        type=int,   default=512)
    parser.add_argument('--workers',      type=int,   default=4)
    parser.add_argument('--force-eval',   action='store_true',
                        help='Re-run model inference even if results.pt exists')
    parser.add_argument('--save-results', action='store_true',
                        help='Overwrite results.pt files with fresh eval results')
    parser.add_argument('--fixed-eps',    type=float, default=None,
                        help='Skip eps sweep; use this exact eps for DBSCAN')
    parser.add_argument('--mode',         default='both',
                        choices=['position', 'pos+vel', 'both'],
                        help='DBSCAN feature mode (default: both)')
    parser.add_argument('--eps-sweep-only', action='store_true',
                        help='Print full eps sweep table for all values, skip model eval')
    parser.add_argument('--dbscan-only',  action='store_true',
                        help='Run DBSCAN only, skip model eval (use with --fixed-eps)')
    parser.add_argument('--record-file',  default=None,
                        metavar='PATH',
                        help='After each result, write it into this text file '
                             '(replaces TBD rows). Default: perception_detection_results.txt '
                             'when --record-file is given without a value.')
    parser.add_argument('--no-latex',     action='store_true',
                        help='Skip LaTeX table output')
    parser.add_argument('--no-cuda',      action='store_true')
    args = parser.parse_args()

    device = 'cpu'
    if HAS_TORCH and not args.no_cuda and torch.cuda.is_available():
        device = 'cuda'
    print(f"[eval] Device: {device}")

    # ── Load data ─────────────────────────────────────────────────────────────
    val_data  = np.load(os.path.join(args.data, 'val.npz'))
    test_data = np.load(os.path.join(args.data, 'test.npz'))

    val_feats,  val_masks,  val_labels  = (val_data['feats'],
                                            val_data['masks'].astype(bool),
                                            val_data['labels'])
    test_feats, test_masks, test_labels = (test_data['feats'],
                                            test_data['masks'].astype(bool),
                                            test_data['labels'])
    print(f"[eval] Val: {len(val_feats):,} samples   Test: {len(test_feats):,} samples\n")

    rows = []   # (display_name, metrics, note, group_id)

    # ── DBSCAN baselines ──────────────────────────────────────────────────────
    n_sw   = args.n_sweep or None
    n_te   = args.n_test  or None
    rng    = np.random.default_rng(0)
    te_idx = (rng.choice(len(test_feats), min(n_te, len(test_feats)), replace=False)
              if n_te else np.arange(len(test_feats)))
    te_f, te_m, te_l = test_feats[te_idx], test_masks[te_idx], test_labels[te_idx]

    n_test_str = f"{len(te_idx)} test samples" if n_te else "all test samples"

    modes_to_run = (['position', 'pos+vel'] if args.mode == 'both'
                    else [args.mode])

    rec_path = args.record_file   # None if not given

    # ── Full eps sweep table (appendix mode) ──────────────────────────────────
    if args.eps_sweep_only:
        EPS_LIST = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
        W = 72
        print()
        print("=" * W)
        print("  DBSCAN FULL EPS SWEEP  (test set)      [for CoRL appendix]")
        print("=" * W)
        hdr = f"  {'Method':<36} {'F1':>5}  {'Prec':>5}  {'Rec':>5}  {'ARI':>5}"
        print(hdr);  print("  " + "─" * (W - 2))
        for mode in modes_to_run:
            for eps in EPS_LIST:
                m = _evaluate_dbscan(te_f, te_m, te_l, eps, mode, args.vel_scale)
                label = (f"DBSCAN (pos)  eps={eps:.1f}"
                         if mode == 'position'
                         else f"DBSCAN (p+v)  eps={eps:.1f}")
                print(f"  {label:<36} {m['f1']:5.3f}  {m['precision']:5.3f}  "
                      f"{m['recall']:5.3f}  {m['ari']:5.3f}")
                if rec_path:
                    _record_result(rec_path, mode, eps, m)
            print("  " + "·" * (W - 2))
        print("=" * W)
        print(f"\n  Evaluated on {n_test_str}.")
        if rec_path:
            print(f"  Results recorded in {rec_path}.")
        else:
            print("  Copy these values into perception_detection_results.txt → Appendix table.")
        return

    # ── Single fixed-eps run (one appendix row at a time) ─────────────────────
    if args.fixed_eps is not None:
        for mode in modes_to_run:
            m = _evaluate_dbscan(te_f, te_m, te_l, args.fixed_eps, mode, args.vel_scale)
            label = (f"DBSCAN (position)   [eps={args.fixed_eps:.1f}]"
                     if mode == 'position'
                     else f"DBSCAN (pos+vel)    [eps={args.fixed_eps:.1f}]")
            print(f"\n[DBSCAN {mode}] eps={args.fixed_eps}  ({n_test_str})")
            print(f"  F1={m['f1']:.3f}  Prec={m['precision']:.3f}  "
                  f"Rec={m['recall']:.3f}  ARI={m['ari']:.3f}")
            if rec_path:
                _record_result(rec_path, mode, args.fixed_eps, m)
            rows.append((label, m, '', 'dbscan'))
        if args.dbscan_only:
            _ascii_table(rows)
            if not args.no_latex:
                _latex_table(rows)
            return

    # ── Auto eps sweep (normal / main paper mode) ─────────────────────────────
    elif not args.dbscan_only:
        n_sweep_str = f"{n_sw} val samples" if n_sw else "all val samples"
        for mode in modes_to_run:
            print(f"[DBSCAN {mode}] Sweeping eps ({n_sweep_str}) …")
            best_eps = _sweep_eps(val_feats, val_masks, val_labels, mode,
                                  args.vel_scale, n_sweep=n_sw)
            print(f"  Best eps = {best_eps:.2f}")
            m = _evaluate_dbscan(te_f, te_m, te_l, best_eps, mode, args.vel_scale)
            m['best_eps'] = best_eps
            label = (f"DBSCAN (position)   [eps={best_eps:.1f}]"
                     if mode == 'position'
                     else f"DBSCAN (pos+vel)    [eps={best_eps:.1f}]")
            print(f"  Test ({n_test_str}) → F1={m['f1']:.3f}  P={m['precision']:.3f}  "
                  f"R={m['recall']:.3f}  ARI={m['ari']:.3f}\n")
            rows.append((label, m, '', 'dbscan'))

    # ── GroupDetector evaluation ───────────────────────────────────────────────
    if args.dbscan_only:
        _ascii_table(rows)
        if not args.no_latex:
            _latex_table(rows)
        return

    if not HAS_TORCH:
        print("WARNING: PyTorch not available — skipping model evaluation.")
    else:
        from torch.utils.data import DataLoader, TensorDataset

        def _make_loader(feats_np, masks_np, labels_np, batch):
            ds = TensorDataset(
                torch.from_numpy(feats_np.astype(np.float32)),
                torch.from_numpy(masks_np.astype(np.float32)).bool(),
                torch.from_numpy(labels_np.astype(np.float32)),
            )
            return DataLoader(ds, batch_size=batch, shuffle=False,
                              num_workers=args.workers, pin_memory=(device == 'cuda'))

        test_loader = _make_loader(test_feats, test_masks, test_labels, args.batch)

        # ── Phase 1 (encoder only) ─────────────────────────────────────────────
        p1_results_path = os.path.join(args.phase1_save, 'phase1_results.pt')
        p1_metrics = None

        ckpt_valid = args.phase1_ckpt not in ('__skip__', '') and os.path.exists(args.phase1_ckpt)

        if not ckpt_valid:
            if args.phase1_ckpt != '__skip__':
                print(f"[Phase1] Checkpoint not found: {args.phase1_ckpt} — skipping.")
        elif not args.force_eval and os.path.exists(p1_results_path):
            print(f"[Phase1] Loading saved results from {p1_results_path}")
            r = torch.load(p1_results_path, map_location='cpu')
            m = r['test_metrics']
            # Saved results may lack ARI — re-compute from checkpoint
            if 'ari' not in m:
                print(f"[Phase1] ARI missing in saved results — re-evaluating …")
                model      = _load_phase1(args.phase1_ckpt, device)
                p1_metrics = _eval_model(model, test_loader, device)
            else:
                p1_metrics = {
                    'f1':        m.get('f1', 0),
                    'precision': m.get('precision', 0),
                    'recall':    m.get('recall', 0),
                    'ari':       m.get('ari', 0),
                    'auroc':     m.get('auroc', None),
                }
        else:
            print(f"[Phase1] Running evaluation from {args.phase1_ckpt} …")
            model      = _load_phase1(args.phase1_ckpt, device)
            p1_metrics = _eval_model(model, test_loader, device)

        if p1_metrics:
            if args.save_results and os.path.exists(args.phase1_ckpt):
                ckpt = torch.load(args.phase1_ckpt, map_location='cpu')
                out  = {'test_metrics': p1_metrics,
                        'best_val_f1': ckpt.get('val_metrics', {}).get('f1', 0),
                        'best_threshold': 0.5,
                        'variant': 'B',
                        'vcfg': ckpt.get('vcfg', {})}
                torch.save(out, p1_results_path)
                print(f"[Phase1] Results saved → {p1_results_path}")
            print(f"[Phase1] F1={p1_metrics['f1']:.3f}  P={p1_metrics['precision']:.3f}  "
                  f"R={p1_metrics['recall']:.3f}  ARI={p1_metrics['ari']:.3f}  "
                  f"AUROC={p1_metrics.get('auroc') or 0:.3f}\n")
            rows.append(('GroupDetector Phase1 (encoder)',
                          p1_metrics, '', 'ours'))

        # ── Phase 2 (encoder + GNN = GRACE backbone) ───────────────────────────
        p2_results_path = os.path.join(args.phase2_save, 'phase2_results.pt')
        p2_metrics = None
        ckpt2_valid = args.phase2_ckpt not in ('__skip__', '') and os.path.exists(args.phase2_ckpt)

        if not ckpt2_valid:
            if args.phase2_ckpt != '__skip__':
                print(f"[Phase2] Checkpoint not found: {args.phase2_ckpt} — skipping.")
        elif not args.force_eval and os.path.exists(p2_results_path):
            print(f"[Phase2] Loading saved results from {p2_results_path}")
            r = torch.load(p2_results_path, map_location='cpu')
            m = r['test_metrics']
            p2_metrics = {
                'f1':        m.get('f1', 0),
                'precision': m.get('precision', 0),
                'recall':    m.get('recall', 0),
                'ari':       m.get('ari', 0),
                'auroc':     m.get('auroc', None),
            }
        else:
            print(f"[Phase2] Running evaluation from {args.phase2_ckpt} …")
            model      = _load_phase2(args.phase2_ckpt, device)
            p2_metrics = _eval_model(model, test_loader, device)

        if p2_metrics:
            if args.save_results and os.path.exists(args.phase2_ckpt):
                torch.save({'test_metrics': p2_metrics, 'best_val_f1': 0,
                             'best_threshold': 0.5}, p2_results_path)
                print(f"[Phase2] Results saved → {p2_results_path}")
            print(f"[Phase2] F1={p2_metrics['f1']:.3f}  P={p2_metrics['precision']:.3f}  "
                  f"R={p2_metrics['recall']:.3f}  ARI={p2_metrics['ari']:.3f}  "
                  f"AUROC={p2_metrics.get('auroc') or 0:.3f}\n")
            rows.append(('GroupDetector Phase2 (GNN) ★',
                          p2_metrics, 'GRACE backbone', 'ours'))

    # ── Print tables ──────────────────────────────────────────────────────────
    _ascii_table(rows)
    if not args.no_latex:
        _latex_table(rows)


if __name__ == '__main__':
    main()
