"""
GRAM-v2 — group detection comparison visualiser.

Renders a 3-row × 3-column grid on the same test sample:

  Rows    : method  (DBSCAN | Phase 1 | Phase 2)
  Columns : temporal frame  (t-2 oldest | t-1 | t current)

Each cell shows the same scene at that point in time with prediction edges
overlaid on GT-coloured circles.  Temporal columns let you see co-movement:
group members track together across frames; strangers drift apart.

DBSCAN is run independently per frame (shows proximity instability over time).
Phase 1 / Phase 2 W-matrices are computed from the full 3-frame input and
overlaid on each frame's positions (shows temporal-aware stable predictions).

Edge colouring (all three method rows):
  ─── green  : True Positive  (predicted pair = real group pair)
  ─── red    : False Positive (predicted pair ≠ real group pair)
  ╌╌╌ orange : False Negative (real group pair that was missed)

Circle colouring: GT group membership (blue / orange / green / purple per group,
gray for individuals).  ★ = robot at origin (robot-centric coordinates).

Three frame-selection modes:
  random           : pick a random test sample (default)
  dbscan_vs_gram   : auto-find a frame where DBSCAN F1 < 0.40 AND Phase 2 F1 > 0.75
  phase_transition : auto-find a frame where Phase 1 misses ≥1 pair in a 3-person
                     group that Phase 2 catches (GNN transitivity story)

Usage:
  python gram_v2_visualize_detection.py
  python gram_v2_visualize_detection.py --mode dbscan_vs_gram --out dbscan_fail.png
  python gram_v2_visualize_detection.py --mode phase_transition --out p1_fail.png
  python gram_v2_visualize_detection.py --idx 42 --out sample_42.png
  python gram_v2_visualize_detection.py --mode dbscan_vs_gram --gif --n-frames 6 --out compare.gif
  python gram_v2_visualize_detection.py --mode dbscan_vs_gram --split-gif --n-frames 6 --out split
  # → split_dbscan.gif  split_phase1.gif  split_phase2.gif  (1-row × 3-col each)
"""

import os, sys, argparse, random
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    print("ERROR: scikit-learn required. pip install scikit-learn"); sys.exit(1)

try:
    import torch
except ImportError:
    print("ERROR: torch required."); sys.exit(1)

try:
    from scipy.sparse.csgraph import connected_components
except ImportError:
    print("ERROR: scipy required. pip install scipy"); sys.exit(1)


# ── Constants ─────────────────────────────────────────────────────────────────
FEAT_DIM   = 7
T_WINDOW   = 3               # temporal frames stacked per sample (oldest → newest)
MAX_HUMANS = 20
THRESHOLD  = 0.65            # Phase 1/2 decision boundary (optimal from eval)
VEL_SCALE  = 0.5             # pos+vel DBSCAN scale (matches gram_v2_eval_dbscan.py)
DBSCAN_EPS = 2.0             # optimal eps from gram_v2_eval_dbscan.py (pos+vel)

# Column labels for the 3 temporal frames
FRAME_LABELS = ['t−2  (oldest)', 't−1', 't  (current)']
# Row labels for the 3 methods
ROW_LABELS   = ['DBSCAN\n(pos+vel)', 'Phase 1\n(Encoder + EdgeNet)', 'Phase 2\n(+ 3-layer GNN)']

GROUP_COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0', '#FF9800']
INDIV_COLOR  = '#B0BEC5'


# ── GT group IDs from pairwise label matrix ───────────────────────────────────

def gt_group_ids(labels_mat, vis_idx):
    """
    Returns dict: human_index → group_id (0,1,2,…) or -1 if individual.
    Runs connected-components on the visible sub-matrix.
    """
    if len(vis_idx) < 2:
        return {vi: -1 for vi in vis_idx}
    sub = labels_mat[np.ix_(vis_idx, vis_idx)]
    _, comp = connected_components(sub, directed=False, connection='weak')
    has_partner = sub.sum(axis=1) > 0
    comp_to_gid, gid_counter, result = {}, 0, {}
    for local_i, (vi, hp) in enumerate(zip(vis_idx, has_partner)):
        if hp:
            c = int(comp[local_i])
            if c not in comp_to_gid:
                comp_to_gid[c] = gid_counter
                gid_counter += 1
            result[vi] = comp_to_gid[c]
        else:
            result[vi] = -1
    return result


# ── DBSCAN inference ───────────────────────────────────────────────────────────

def dbscan_predict(feats_sample, mask):
    """DBSCAN on the most-recent frame (frame_k=2). Used by find_frame for scoring."""
    return dbscan_predict_frame(feats_sample, mask, frame_k=T_WINDOW - 1)


def dbscan_predict_frame(feats_sample, mask, frame_k):
    """
    Run DBSCAN (pos+vel) on a single temporal frame.
    frame_k : 0 = oldest (t-2), 1 = middle (t-1), 2 = current (t)
    Returns (20, 20) float32 pairwise groupness matrix (0.0 or 1.0).
    """
    vis_idx = np.where(mask)[0]
    pred = np.zeros((MAX_HUMANS, MAX_HUMANS), dtype=np.float32)
    if len(vis_idx) < 2:
        return pred
    frame = feats_sample[vis_idx, frame_k * FEAT_DIM : (frame_k + 1) * FEAT_DIM]
    pos   = frame[:, :2]
    vel   = frame[:, 2:4] * VEL_SCALE
    X     = np.concatenate([pos, vel], axis=1)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=2).fit_predict(X)
    for a, i in enumerate(vis_idx):
        for b, j in enumerate(vis_idx):
            if i == j:
                continue
            if labels[a] != -1 and labels[a] == labels[b]:
                pred[i, j] = 1.0
    return pred


# ── Neural model loading ───────────────────────────────────────────────────────

def load_model(path, n_gnn_layers, device):
    from crowd_nav.gram_v2.models import GroupDetector
    model = GroupDetector(n_gnn_layers=n_gnn_layers).to(device)
    ckpt  = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model


def model_predict(model, feats_sample, mask_sample, device):
    """Run GroupDetector on one sample. Returns (20, 20) W_final as numpy array."""
    x = torch.from_numpy(feats_sample).float().unsqueeze(0).to(device)
    m = torch.from_numpy(mask_sample).bool().unsqueeze(0).to(device)
    with torch.no_grad():
        W_final, _, _, _ = model(x, m)
    return W_final.squeeze(0).cpu().numpy()


# ── Per-sample metrics ─────────────────────────────────────────────────────────

def sample_metrics(pred_mat, gt_mat, mask, threshold=THRESHOLD):
    """F1, TP, FP, FN over all valid visible off-diagonal pairs."""
    vis   = mask.astype(bool)
    eye   = np.eye(MAX_HUMANS, dtype=bool)
    valid = (vis[:, None] & vis[None, :]) & ~eye
    pred  = (pred_mat[valid] > threshold).astype(bool)
    gt    = (gt_mat[valid]   > 0.5      ).astype(bool)
    tp = int(( pred &  gt).sum())
    fp = int(( pred & ~gt).sum())
    fn = int((~pred &  gt).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    return float(f1), tp, fp, fn


# ── Transitivity failure detection ────────────────────────────────────────────

def has_transitivity_failure(w_p1, gt_mat, mask, threshold=THRESHOLD):
    """
    Returns True when Phase 1 has a transitive miss in a 3+ member group:
    W1[A,B] > threshold AND W1[B,C] > threshold BUT W1[A,C] <= threshold,
    while GT[A,C] = 1. This is the GNN-fixes-transitivity story.
    """
    vis_idx = np.where(mask)[0]
    gid_map = gt_group_ids(gt_mat, vis_idx)
    groups  = defaultdict(list)
    for vi, gid in gid_map.items():
        if gid >= 0:
            groups[gid].append(vi)
    for members in groups.values():
        if len(members) < 3:
            continue
        for a in members:
            for b in members:
                if a == b: continue
                for c in members:
                    if c == b or c == a: continue
                    if (w_p1[a, b] > threshold and w_p1[b, c] > threshold
                            and gt_mat[a, c] > 0.5 and w_p1[a, c] <= threshold):
                        return True
    return False


# ── Rendering ─────────────────────────────────────────────────────────────────

def render_panel(ax, feats_sample, mask, gt_mat, pred_mat, title,
                 threshold=THRESHOLD, frame_k=T_WINDOW - 1):
    """
    Render one cell of the temporal grid onto ax.

    frame_k  : which temporal frame to use for positions/velocities
               (0 = t-2 oldest, 1 = t-1, 2 = t current)
    pred_mat : (20,20) groupness matrix for this cell (DBSCAN per-frame, or
               Phase 1/2 W fixed across all columns)
    """
    vis_idx = np.where(mask)[0]
    frame   = feats_sample[:, frame_k * FEAT_DIM : (frame_k + 1) * FEAT_DIM]
    pos     = frame[:, :2]     # p_rel_x, p_rel_y  (robot-centric)
    vel     = frame[:, 2:4]    # v_rel_x, v_rel_y
    gid_map = gt_group_ids(gt_mat, vis_idx)

    # ── prediction edges ──────────────────────────────────────────────────────
    drawn = set()
    for i in vis_idx:
        for j in vis_idx:
            if i >= j or (i, j) in drawn:
                continue
            drawn.add((i, j))
            is_gt   = gt_mat[i, j]   > 0.5
            is_pred = pred_mat[i, j] > threshold
            if   is_gt and     is_pred: color, ls, lw = '#4CAF50', '-',  2.0
            elif not is_gt and is_pred: color, ls, lw = '#F44336', '-',  1.5
            elif is_gt and not is_pred: color, ls, lw = '#FF9800', '--', 1.5
            else: continue
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]],
                    color=color, ls=ls, lw=lw, alpha=0.85, zorder=1)

    # ── human circles + velocity arrows ──────────────────────────────────────
    for vi in vis_idx:
        gid   = gid_map.get(vi, -1)
        color = GROUP_COLORS[gid % len(GROUP_COLORS)] if gid >= 0 else INDIV_COLOR
        ax.add_patch(mpatches.Circle(
            (pos[vi, 0], pos[vi, 1]), 0.28, color=color, alpha=0.88, zorder=2))
        v = vel[vi]
        if np.linalg.norm(v) > 0.05:
            ax.annotate('', xy=(pos[vi, 0] + v[0] * 0.4, pos[vi, 1] + v[1] * 0.4),
                        xytext=(pos[vi, 0], pos[vi, 1]),
                        arrowprops=dict(arrowstyle='->', color='#37474F', lw=1.2),
                        zorder=3)

    # ── robot at origin ───────────────────────────────────────────────────────
    ax.plot(0, 0, marker='*', ms=14, color='black', zorder=4, label='Robot')

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    ax.grid(True, alpha=0.15, lw=0.5)
    ax.tick_params(labelsize=7)


def add_shared_legend(fig):
    handles = [
        Line2D([0], [0], marker='*', color='black', ms=11, ls='None',
               label='★  Robot (origin)'),
        Line2D([0], [0], color='#4CAF50', lw=2,         label='True Positive'),
        Line2D([0], [0], color='#F44336', lw=2,         label='False Positive'),
        Line2D([0], [0], color='#FF9800', lw=2, ls='--', label='False Negative'),
    ]
    for k, c in enumerate(GROUP_COLORS[:3]):
        handles.append(mpatches.Patch(color=c, label=f'GT Group {k + 1}'))
    handles.append(mpatches.Patch(color=INDIV_COLOR, label='Individual'))
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), framealpha=0.95)


def _fill_temporal_grid(axes, feats, mask, gt, db_frames, p1_mat, p2_mat, threshold):
    """
    Fill a pre-created (3, 3) axes array with the temporal comparison grid.
    Called by both render_temporal_grid (static PNG) and GIF update loop.

    axes[row, col]:
      row 0 = DBSCAN, row 1 = Phase 1, row 2 = Phase 2
      col 0 = t-2,    col 1 = t-1,     col 2 = t (current)
    """
    f1_db, tp_db, fp_db, fn_db = sample_metrics(db_frames[2], gt, mask, threshold)
    f1_p1, tp_p1, fp_p1, fn_p1 = sample_metrics(p1_mat,       gt, mask, threshold)
    f1_p2, tp_p2, fp_p2, fn_p2 = sample_metrics(p2_mat,       gt, mask, threshold)

    # pred_mats[row] is a list of 3 matrices — one per temporal column
    pred_mats = [
        db_frames,           # DBSCAN: different matrix per frame
        [p1_mat] * T_WINDOW, # Phase 1: same W across all frames
        [p2_mat] * T_WINDOW, # Phase 2: same W across all frames
    ]
    metrics_by_row = [
        (f1_db, tp_db, fp_db, fn_db),
        (f1_p1, tp_p1, fp_p1, fn_p1),
        (f1_p2, tp_p2, fp_p2, fn_p2),
    ]

    for row, (row_label, preds, mets) in enumerate(
            zip(ROW_LABELS, pred_mats, metrics_by_row)):
        f1, tp, fp, fn = mets
        for col, frame_k in enumerate(range(T_WINDOW)):
            ax = axes[row, col]
            # Column title: frame time label; append metrics on current frame
            if col == T_WINDOW - 1:
                title = f'{FRAME_LABELS[col]}\nF1={f1:.3f}  TP={tp} FP={fp} FN={fn}'
            else:
                title = FRAME_LABELS[col]
            render_panel(ax, feats, mask, gt, preds[col], title, threshold, frame_k)
            # Row label on leftmost column only
            if col == 0:
                ax.set_ylabel(row_label, fontsize=10, fontweight='bold', labelpad=6)


def render_temporal_grid(feats, mask, gt, p1_mat, p2_mat, out_path, threshold=THRESHOLD):
    """
    Build the 3×3 temporal grid, save to out_path.
    DBSCAN is run per-frame internally.
    """
    db_frames = [dbscan_predict_frame(feats, mask, k) for k in range(T_WINDOW)]

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.set_facecolor('#FAFAFA')

    _fill_temporal_grid(axes, feats, mask, gt, db_frames, p1_mat, p2_mat, threshold)

    add_shared_legend(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    f1_db = sample_metrics(db_frames[2], gt, mask, threshold)[0]
    f1_p1 = sample_metrics(p1_mat,       gt, mask, threshold)[0]
    f1_p2 = sample_metrics(p2_mat,       gt, mask, threshold)[0]
    print(f"Saved → {out_path}")
    print(f"  DBSCAN F1={f1_db:.3f}  Phase1 F1={f1_p1:.3f}  Phase2 F1={f1_p2:.3f}")


def render_method_grid(feats, mask, gt, pred_list, row_label, out_path, threshold=THRESHOLD):
    """
    1-row × 3-column temporal grid for a single detection method.
    Used by --split-gif to produce one GIF per method.

    pred_list : [pred_t2, pred_t1, pred_t]  — one (20,20) matrix per frame
    row_label : method name shown as figure title
    """
    f1, tp, fp, fn = sample_metrics(pred_list[T_WINDOW - 1], gt, mask, threshold)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.set_facecolor('#FAFAFA')
    fig.suptitle(row_label, fontsize=13, fontweight='bold', y=1.01)

    for col, frame_k in enumerate(range(T_WINDOW)):
        if col == T_WINDOW - 1:
            title = f'{FRAME_LABELS[col]}\nF1={f1:.3f}  TP={tp} FP={fp} FN={fn}'
        else:
            title = FRAME_LABELS[col]
        render_panel(axes[col], feats, mask, gt, pred_list[col], title, threshold, frame_k)

    add_shared_legend(fig)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_triangle_frame(feats, mask, gt, db_frames, p1_mat, p2_mat,
                          out_path, threshold=THRESHOLD):
    """
    Triangular 3-screen layout for one scene:
      Screen 1 (top-left)     — DBSCAN       t-2 | t-1 | t
      Screen 2 (top-right)    — Phase 1      t-2 | t-1 | t
      Screen 3 (bottom-center)— Phase 2      t-2 | t-1 | t

    GridSpec(2, 12): each temporal panel spans 2 columns.
      DBSCAN  → cols 0-1, 2-3, 4-5
      Phase 1 → cols 6-7, 8-9, 10-11
      Phase 2 → cols 3-4, 5-6, 7-8   (centred under the Screen-1/Screen-2 join at col 6)
    """
    f1_db, tp_db, fp_db, fn_db = sample_metrics(db_frames[2], gt, mask, threshold)
    f1_p1, tp_p1, fp_p1, fn_p1 = sample_metrics(p1_mat,        gt, mask, threshold)
    f1_p2, tp_p2, fp_p2, fn_p2 = sample_metrics(p2_mat,        gt, mask, threshold)

    fig = plt.figure(figsize=(18, 12))
    fig.set_facecolor('#FAFAFA')

    gs = GridSpec(2, 12, figure=fig, hspace=0.55, wspace=0.30)
    ax_db = [fig.add_subplot(gs[0, k * 2 : k * 2 + 2])     for k in range(3)]
    ax_p1 = [fig.add_subplot(gs[0, 6 + k * 2 : 8 + k * 2]) for k in range(3)]
    ax_p2 = [fig.add_subplot(gs[1, 3 + k * 2 : 5 + k * 2]) for k in range(3)]

    screen_rows = [
        (ax_db, ROW_LABELS[0], db_frames,           (f1_db, tp_db, fp_db, fn_db)),
        (ax_p1, ROW_LABELS[1], [p1_mat] * T_WINDOW, (f1_p1, tp_p1, fp_p1, fn_p1)),
        (ax_p2, ROW_LABELS[2], [p2_mat] * T_WINDOW, (f1_p2, tp_p2, fp_p2, fn_p2)),
    ]
    for axes_row, row_label, pred_list, (f1, tp, fp, fn) in screen_rows:
        for col, frame_k in enumerate(range(T_WINDOW)):
            title = (f'{FRAME_LABELS[col]}\nF1={f1:.3f}  TP={tp} FP={fp} FN={fn}'
                     if col == T_WINDOW - 1 else FRAME_LABELS[col])
            render_panel(axes_row[col], feats, mask, gt, pred_list[col],
                         title, threshold, frame_k)
        axes_row[0].set_ylabel(row_label, fontsize=10, fontweight='bold', labelpad=6)

    # Screen header labels, centred above each 1×3 strip via annotation on middle panel
    for mid_ax, header, bg in [
        (ax_db[1], 'Screen 1 — DBSCAN (pos+vel)',            '#E3F2FD'),
        (ax_p1[1], 'Screen 2 — Phase 1 (Encoder + EdgeNet)', '#E8F5E9'),
        (ax_p2[1], 'Screen 3 — Phase 2 (+ 3-layer GNN)',     '#FFF3E0'),
    ]:
        mid_ax.annotate(
            header, xy=(0.5, 1.0), xycoords='axes fraction',
            xytext=(0, 28), textcoords='offset points',
            ha='center', va='bottom', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor=bg, alpha=0.85),
            annotation_clip=False)

    add_shared_legend(fig)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Auto-find interesting frames ───────────────────────────────────────────────

def find_frame(feats, masks, labels, p1_model, p2_model, device, mode,
               max_scan=1000, threshold=THRESHOLD):
    """
    Scan up to max_scan random samples; return index of the best frame.

    dbscan_vs_gram   : maximise (Phase2 F1 - DBSCAN F1) subject to
                       DBSCAN F1 < 0.40 and Phase2 F1 > 0.75
    phase_transition : Phase 1 has a transitive miss AND Phase2 > Phase1 + 0.05
    """
    n = len(feats)
    indices = random.sample(range(n), min(max_scan, n))
    best_idx, best_score = None, -1.0

    for idx in indices:
        f, m, gt = feats[idx], masks[idx], labels[idx]
        if m.sum() < 4 or gt.sum() < 2:
            continue
        db = dbscan_predict(f, m)        # current frame only for scoring
        p1 = model_predict(p1_model, f, m, device)
        p2 = model_predict(p2_model, f, m, device)

        f1_db            = sample_metrics(db, gt, m, threshold)[0]
        f1_p1, _, _, fn1 = sample_metrics(p1, gt, m, threshold)
        f1_p2            = sample_metrics(p2, gt, m, threshold)[0]

        if mode == 'dbscan_vs_gram':
            score = (f1_p2 - f1_db) if (f1_db < 0.40 and f1_p2 > 0.75) else -1.0
        elif mode == 'phase_transition':
            if fn1 == 0 or not has_transitivity_failure(p1, gt, m, threshold):
                score = -1.0
            else:
                score = (f1_p2 - f1_p1) if f1_p2 > f1_p1 + 0.05 else -1.0
        else:
            score = -1.0

        if score > best_score:
            best_score, best_idx = score, idx

    return best_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',      default='gram_v2_data',
                        help='Directory containing test.npz')
    parser.add_argument('--phase1',    default='trained_models/gram_v2/phase1_v1/B/best.pt')
    parser.add_argument('--phase2',    default='trained_models/gram_v2/phase2_v1/best.pt')
    parser.add_argument('--mode',      default='random',
                        choices=['random', 'dbscan_vs_gram', 'phase_transition'])
    parser.add_argument('--idx',       type=int, default=None,
                        help='Fixed test sample index (overrides --mode)')
    parser.add_argument('--gif',        action='store_true',
                        help='Produce animated GIF (N scenes, each a full 3×3 grid). '
                             'Default output: detection_<mode>.gif')
    parser.add_argument('--split-gif', action='store_true',
                        help='Produce 3 separate method GIFs (1 row × 3 temporal columns each): '
                             '<base>_dbscan.gif  <base>_phase1.gif  <base>_phase2.gif. '
                             'Useful as supplementary video clips. '
                             'Can be combined with --gif to get all 4 outputs at once.')
    parser.add_argument('--tri-gif', action='store_true',
                        help='Produce triangular-layout GIF: DBSCAN top-left, Phase 1 top-right, '
                             'Phase 2 bottom-center (centred under the join). '
                             'Output: <base>_tri.gif. Can be combined with --gif / --split-gif.')
    parser.add_argument('--out',       default=None,
                        help='Output path. PNG: detection_<mode>.png. '
                             'GIF: detection_<mode>.gif. '
                             'Split-GIF: used as base prefix (e.g. --out figures/split '
                             '→ figures/split_dbscan.gif etc.)')
    parser.add_argument('--n-frames',  type=int, default=6,
                        help='Number of scenes in GIF')
    parser.add_argument('--max-scan',  type=int, default=1000,
                        help='Max samples to scan in auto-find modes')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--seed',      type=int, default=0)
    parser.add_argument('--no-cuda',   action='store_true')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    device = 'cpu' if args.no_cuda or not torch.cuda.is_available() else 'cuda'

    # ── Check inputs ──────────────────────────────────────────────────────────
    test_path = os.path.join(args.data, 'test.npz')
    for p, lbl in [(test_path, 'test.npz'),
                   (args.phase1, '--phase1'),
                   (args.phase2, '--phase2')]:
        if not os.path.exists(p):
            print(f"ERROR: {lbl} not found at {p}"); sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading test data…")
    td     = np.load(test_path)
    feats  = td['feats']
    masks  = td['masks'].astype(bool)
    labels = td['labels']
    print(f"  {len(feats):,} test samples loaded")

    # ── Load models ────────────────────────────────────────────────────────────
    print("Loading Phase 1 model (n_gnn_layers=0)…")
    p1_model = load_model(args.phase1, n_gnn_layers=0, device=device)
    print("Loading Phase 2 model (n_gnn_layers=3)…")
    p2_model = load_model(args.phase2, n_gnn_layers=3, device=device)

    gif_mode       = args.gif or (args.out or '').endswith('.gif')
    split_gif_mode = args.split_gif
    tri_gif_mode   = args.tri_gif
    need_multi     = gif_mode or split_gif_mode or tri_gif_mode

    if args.out is None:
        args.out = f'detection_{args.mode}.gif' if gif_mode else f'detection_{args.mode}.png'

    # Base path for split-GIF outputs (strip .gif suffix if present)
    split_base = (args.out[:-4] if args.out.endswith('.gif') else args.out) \
                 if split_gif_mode else None

    # Output path for tri-GIF (<base>_tri.gif); computed always, used only when tri_gif_mode
    _tb = args.out
    if _tb.endswith('.gif'): _tb = _tb[:-4]
    elif _tb.endswith('.png'): _tb = _tb[:-4]
    tri_out = _tb + '_tri.gif'

    # ── Select sample index/indices ────────────────────────────────────────────
    if args.idx is not None:
        indices = [args.idx]
    elif need_multi:
        n_needed = args.n_frames
        if args.mode == 'random':
            indices = random.sample(range(len(feats)), n_needed)
        else:
            print(f"Scanning for {n_needed} frames (mode={args.mode})…")
            found, attempts = [], 0
            while len(found) < n_needed and attempts < args.max_scan:
                idx = find_frame(feats, masks, labels, p1_model, p2_model,
                                 device, args.mode, max_scan=50,
                                 threshold=args.threshold)
                attempts += 50
                if idx is not None and idx not in found:
                    found.append(idx)
            indices = found or random.sample(range(len(feats)), n_needed)
    elif args.mode == 'random':  # single PNG, random
        indices = [random.randint(0, len(feats) - 1)]
    else:
        print(f"Scanning for best frame (mode={args.mode}, max_scan={args.max_scan})…")
        idx = find_frame(feats, masks, labels, p1_model, p2_model,
                         device, args.mode, args.max_scan, args.threshold)
        if idx is None:
            print("No matching frame found — falling back to random sample.")
            idx = random.randint(0, len(feats) - 1)
        indices = [idx]

    print(f"Selected index/indices: {indices}")

    # ── Helper: predictions for one sample ────────────────────────────────────
    def get_preds(idx):
        f, m = feats[idx], masks[idx]
        p1 = model_predict(p1_model, f, m, device)
        p2 = model_predict(p2_model, f, m, device)
        return f, m, labels[idx], p1, p2

    # ── Render ─────────────────────────────────────────────────────────────────
    if gif_mode:
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: pillow required for GIF output. pip install pillow")
            sys.exit(1)

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='gram_v2_gif_')
        frame_paths = []

        for fi, sample_idx in enumerate(indices):
            f, m, gt, p1, p2 = get_preds(sample_idx)
            tmp_path = os.path.join(tmp_dir, f'frame_{fi:04d}.png')
            render_temporal_grid(f, m, gt, p1, p2, tmp_path, args.threshold)
            frame_paths.append(tmp_path)
            print(f"  GIF frame {fi + 1}/{len(indices)} rendered")

        imgs = [Image.open(p).convert('RGB') for p in frame_paths]
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        imgs[0].save(args.out, save_all=True, append_images=imgs[1:],
                     loop=0, duration=1800)
        for p in frame_paths:
            os.unlink(p)
        os.rmdir(tmp_dir)
        print(f"\nSaved GIF ({len(imgs)} frames) → {args.out}")

    elif not split_gif_mode and not tri_gif_mode:
        # Single PNG
        f, m, gt, p1, p2 = get_preds(indices[0])
        render_temporal_grid(f, m, gt, p1, p2, args.out, args.threshold)

    # ── Split-GIF: one 1×3 temporal GIF per method ────────────────────────────
    if split_gif_mode:
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: pillow required for GIF output. pip install pillow")
            sys.exit(1)

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='gram_v2_split_')

        # Precompute all predictions so each model runs only once per sample
        print("\nPrecomputing predictions for split GIFs…")
        all_items = []
        for sample_idx in indices:
            f, m, gt, p1, p2 = get_preds(sample_idx)
            db_frames = [dbscan_predict_frame(f, m, k) for k in range(T_WINDOW)]
            all_items.append((f, m, gt, db_frames, p1, p2))

        method_configs = [
            ('dbscan', ROW_LABELS[0], lambda item: item[3]),
            ('phase1', ROW_LABELS[1], lambda item: [item[4]] * T_WINDOW),
            ('phase2', ROW_LABELS[2], lambda item: [item[5]] * T_WINDOW),
        ]

        for method_key, method_label, get_pred_list in method_configs:
            out_gif  = f'{split_base}_{method_key}.gif'
            frame_paths = []

            for fi, item in enumerate(all_items):
                f, m, gt = item[0], item[1], item[2]
                pred_list = get_pred_list(item)
                tmp_path  = os.path.join(tmp_dir, f'{method_key}_{fi:04d}.png')
                render_method_grid(f, m, gt, pred_list, method_label,
                                   tmp_path, args.threshold)
                frame_paths.append(tmp_path)
                print(f"  {method_key} frame {fi + 1}/{len(all_items)} rendered")

            imgs = [Image.open(p).convert('RGB') for p in frame_paths]
            os.makedirs(os.path.dirname(out_gif) or '.', exist_ok=True)
            imgs[0].save(out_gif, save_all=True, append_images=imgs[1:],
                         loop=0, duration=1800)
            for p in frame_paths:
                os.unlink(p)
            print(f"Saved {method_key} GIF ({len(imgs)} frames) → {out_gif}")

        os.rmdir(tmp_dir)

    # ── Tri-GIF: triangular layout ────────────────────────────────────────────
    if tri_gif_mode:
        try:
            from PIL import Image
        except ImportError:
            print("ERROR: pillow required for GIF output. pip install pillow")
            sys.exit(1)

        import tempfile
        tmp_dir = tempfile.mkdtemp(prefix='gram_v2_tri_')
        frame_paths = []

        print("\nRendering triangular layout GIF…")
        for fi, sample_idx in enumerate(indices):
            f, m, gt, p1, p2 = get_preds(sample_idx)
            db_frames_tri = [dbscan_predict_frame(f, m, k) for k in range(T_WINDOW)]
            tmp_path = os.path.join(tmp_dir, f'tri_{fi:04d}.png')
            render_triangle_frame(f, m, gt, db_frames_tri, p1, p2,
                                  tmp_path, args.threshold)
            frame_paths.append(tmp_path)
            print(f"  Tri-GIF frame {fi + 1}/{len(indices)} rendered")

        imgs = [Image.open(p).convert('RGB') for p in frame_paths]
        os.makedirs(os.path.dirname(tri_out) or '.', exist_ok=True)
        imgs[0].save(tri_out, save_all=True, append_images=imgs[1:],
                     loop=0, duration=1800)
        for p in frame_paths:
            os.unlink(p)
        os.rmdir(tmp_dir)
        print(f"\nSaved triangular GIF ({len(imgs)} frames) → {tri_out}")


if __name__ == '__main__':
    main()
