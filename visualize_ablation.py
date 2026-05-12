"""
visualize_ablation.py — Ablation study figures and videos for GRACE.

Generates publication-quality figures comparing GRACE (baseline) against
any ablation variant. Keeps visualize_cost_map.py untouched — that file
is for the final GRACE result only.

Plot types
----------
  training_curves  : reward vs updates for all ablations (reads progress.csv, no model)
  results_bar      : grouped SR / GCR bar chart from test logs (no model)
  trajectory       : robot path overlay — baseline vs selected ablation
  cost_map         : 9-channel cost stack — baseline vs selected ablation
  all              : all four of the above

Usage examples
--------------
  # Training curves for all ablations (no model needed)
  python visualize_ablation.py --plot training_curves

  # Results bar chart (populated from test logs as ablations complete)
  python visualize_ablation.py --plot results_bar

  # Trajectory: baseline vs C1, static figure
  python visualize_ablation.py --ablation C1 --plot trajectory --seed 42

  # Cost map comparison: baseline vs C3 (traj channels will be dark)
  python visualize_ablation.py --ablation C3 --plot cost_map --seed 42

  # Trajectory video for C1
  python visualize_ablation.py --ablation C1 --plot trajectory --seed 42 --video

  # Full report for C1 — all plots + video
  python visualize_ablation.py --ablation C1 --plot all --seed 42 --video

  # All ablations, all plots, no video (batch)
  python visualize_ablation.py --ablation all --plot all --seed 42
"""

import sys, os, re, csv, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter

# ── Registry ──────────────────────────────────────────────────────────────────

BASE_DIR = 'trained_models/gram_map'

ABLATIONS = {
    'baseline': {
        'dir':   f'{BASE_DIR}/stageC',
        'label': 'GRACE (full)',
        'color': '#2166ac',
    },
    'C1': {
        'dir':   f'{BASE_DIR}/ablation_C1_no_group',
        'label': 'C1: No group layers',
        'color': '#d73027',
    },
    'C2': {
        'dir':   f'{BASE_DIR}/ablation_C2_K1',
        'label': 'C2: K=1 slot',
        'color': '#f46d43',
    },
    'C3': {
        'dir':   f'{BASE_DIR}/ablation_C3_no_traj',
        'label': 'C3: No traj layers',
        'color': '#fdae61',
    },
    'C4': {
        'dir':   f'{BASE_DIR}/ablation_C4_no_aux',
        'label': 'C4: No aux loss',
        'color': '#74add1',
    },
    'C5': {
        'dir':   f'{BASE_DIR}/ablation_C5_uniform',
        'label': 'C5: Uniform alpha',
        'color': '#4dac26',
    },
}

CH_NAMES = ['Individual\n(current)', 'Traj 0.3 s', 'Traj 0.7 s',
            'Traj 1.0 s', 'Traj 1.5 s', 'Group\nCohesion',
            'Group\nRepulsion', 'Goal\nAttractor', 'Boundary']
CH_CMAPS = ['Reds', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd',
            'Purples', 'RdPu', 'Greens', 'Blues']

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
parser.add_argument('--ablation',   default='all',
                    choices=['C1','C2','C3','C4','C5','baseline','all'],
                    help='Which ablation to visualize (default: all)')
parser.add_argument('--plot',       default='all',
                    choices=['training_curves','results_bar','trajectory','cost_map','all'],
                    help='Which plot type to generate (default: all)')
parser.add_argument('--seed',       type=int, default=42)
parser.add_argument('--max_steps',  type=int, default=200)
parser.add_argument('--smooth',     type=int, default=50,
                    help='Moving-average window for training curves')
parser.add_argument('--out_dir',    default='ablation_figures',
                    help='Output directory for figures and videos')
parser.add_argument('--dpi',        type=int, default=150)
parser.add_argument('--fps',        type=int, default=8)
parser.add_argument('--video',      action='store_true',
                    help='Generate MP4 video for trajectory / cost_map plots')
parser.add_argument('--device',     default='auto')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

ablation_keys = list(ABLATIONS.keys()) if args.ablation == 'all' else [args.ablation]
plot_types    = ['training_curves', 'results_bar', 'trajectory', 'cost_map'] \
                if args.plot == 'all' else [args.plot]

if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)

print(f"Device: {device} | Ablations: {ablation_keys} | Plots: {plot_types}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _moving_avg(arr, w):
    if w <= 1 or len(arr) < w:
        return np.array(arr, dtype=float)
    return np.convolve(arr, np.ones(w) / w, mode='valid')

def _read_progress(model_dir):
    path = os.path.join(model_dir, 'progress.csv')
    if not os.path.exists(path):
        return None, None
    updates, rewards = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                updates.append(int(row['misc/nupdates']))
                rewards.append(float(row['eprewmean']))
            except (KeyError, ValueError):
                pass
    return np.array(updates), np.array(rewards)

def _read_test_log(model_dir, checkpoint='best.pt'):
    log = os.path.join(model_dir, 'test', f'test_{checkpoint}.log')
    if not os.path.exists(log):
        return None
    with open(log) as f:
        text = f.read()
    m = re.search(
        r'success rate:\s*([\d.]+).*?collision rate:\s*([\d.]+).*?'
        r'timeout rate:\s*([\d.]+).*?GCR\):\s*([\d.]+)%',
        text, re.DOTALL)
    if m:
        return {'SR': float(m.group(1)), 'CR': float(m.group(2)),
                'TR': float(m.group(3)), 'GCR': float(m.group(4))}
    return None

def _load_model(model_dir, device):
    from importlib import import_module
    from rl.networks.model import Policy
    from rl.networks.envs import make_vec_envs
    from crowd_sim import *

    mdir = model_dir.rstrip('/')
    try:
        Config   = getattr(import_module(mdir.replace('/', '.') + '.configs.config'), 'Config')
        get_args = getattr(import_module(mdir.replace('/', '.') + '.arguments'), 'get_args')
    except Exception:
        from crowd_nav.configs.config import Config
        from arguments import get_args

    algo_args = get_args()
    algo_args.num_processes  = 1
    algo_args.num_mini_batch = 1

    config = Config()
    config.sim.render      = False
    config.env.use_wrapper = False
    config.sim.human_num     = 20
    config.sim.circle_radius = 8.5
    config.sim.arena_size    = 8.5
    config.group.num_groups  = 3
    config.group.num_on_path = 2
    config.group.types       = ['static_f', 'dynamic_lf', 'dynamic_free']
    r = getattr(config, 'realistic', None)
    if r: r.enabled = True

    policy_name = config.robot.policy
    if policy_name in ('grace', 'gram_map'):
        grace_cfg = getattr(config, 'grace', None) or getattr(config, 'gram_map', None)
        algo_args._grace_cfg         = grace_cfg
        algo_args.grace_use_aux_loss = False

    env_name_file = os.path.join(mdir, 'env_name.txt')
    env_name = open(env_name_file).read().strip() \
               if os.path.exists(env_name_file) else 'CrowdSimVarNum-v0'

    torch.manual_seed(args.seed)
    envs = make_vec_envs(env_name, args.seed, 1, algo_args.gamma, None, device,
                         False, config=config, ax=None, pretext_wrapper=False)

    actor_critic = Policy(envs.observation_space.spaces, envs.action_space,
                          base_kwargs=algo_args, base=policy_name)

    ckpt = os.path.join(mdir, 'checkpoints', 'best.pt')
    if not os.path.exists(ckpt):
        print(f"  [SKIP] No best.pt found in {mdir}/checkpoints/")
        envs.close()
        return None, None, None, None

    actor_critic.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
    actor_critic.to(device).eval()

    grid_range = getattr(grace_cfg if policy_name in ('grace','gram_map') else config,
                         'grid_range', 6.0)

    rnn_hxs = {
        'human_node_rnn':       torch.zeros(1, 1,
                                            actor_critic.base.human_node_rnn_size, device=device),
        'human_human_edge_rnn': torch.zeros(1, actor_critic.base.human_num + 1,
                                            actor_critic.base.human_human_edge_rnn_size, device=device),
    }
    return actor_critic, envs, rnn_hxs, grid_range

def _run_episode(actor_critic, envs, rnn_hxs, device, max_steps):
    obs   = envs.reset()
    masks = torch.zeros(1, 1, device=device)

    traj, cost_frames, human_init, goal_pos = [], [], None, None

    for step in range(max_steps):
        rn = obs['robot_node'][0, 0].cpu().numpy()
        sp = obs['spatial_edges'][0].cpu().numpy()
        robot_pos = rn[0:2].copy()
        traj.append(robot_pos)

        if human_init is None:
            human_init = robot_pos + sp        # absolute human positions at step 0
            goal_pos   = rn[3:5].copy()

        with torch.no_grad():
            _, action, _, rnn_hxs = actor_critic.act(obs, rnn_hxs, masks, deterministic=True)

        cost_map = None
        if hasattr(actor_critic.base, '_last_cost_stack') and \
                actor_critic.base._last_cost_stack is not None:
            cost_map = actor_critic.base._last_cost_stack[0].numpy()
        cost_frames.append(cost_map)

        obs, _, done, _ = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done[0] else [1.0]]).to(device)
        if done[0]:
            break

    return np.array(traj), cost_frames, human_init, goal_pos

# ── Plot: Training Curves ─────────────────────────────────────────────────────

def plot_training_curves():
    print("\n[training_curves] Reading progress.csv files...")
    fig, ax = plt.subplots(figsize=(9, 5))

    any_data = False
    for key, cfg in ABLATIONS.items():
        updates, rewards = _read_progress(cfg['dir'])
        if updates is None:
            print(f"  {key}: no progress.csv — skipped")
            continue
        sm = _moving_avg(rewards, args.smooth)
        x  = updates[len(updates) - len(sm):]
        lw = 2.5 if key == 'baseline' else 1.5
        ls = '-'  if key == 'baseline' else '--'
        ax.plot(x, sm, label=cfg['label'], color=cfg['color'],
                lw=lw, linestyle=ls)
        any_data = True
        print(f"  {key}: {len(updates)} updates, peak reward {rewards.max():.2f}")

    if not any_data:
        ax.text(0.5, 0.5, 'No progress.csv files found yet.\nRun ablation training first.',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)

    ax.set_xlabel('Training updates', fontsize=11)
    ax.set_ylabel(f'Mean episode reward (smooth={args.smooth})', fontsize=11)
    ax.set_title('GRACE Ablation — Training Curves', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = os.path.join(args.out_dir, 'training_curves.png')
    fig.savefig(out, dpi=args.dpi)
    plt.close(fig)
    print(f"  Saved {out}")

# ── Plot: Results Bar Chart ───────────────────────────────────────────────────

def plot_results_bar():
    print("\n[results_bar] Reading test logs...")
    keys, sr_vals, gcr_vals, labels, colors = [], [], [], [], []

    for key, cfg in ABLATIONS.items():
        result = _read_test_log(cfg['dir'])
        if result:
            sr_vals.append(result['SR'])
            gcr_vals.append(result['GCR'])
            print(f"  {key}: SR={result['SR']:.2f}  GCR={result['GCR']:.2f}%")
        else:
            sr_vals.append(None)
            gcr_vals.append(None)
            print(f"  {key}: no test log yet — placeholder")
        keys.append(key)
        labels.append(cfg['label'])
        colors.append(cfg['color'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(keys))
    w = 0.6

    for i, (key, sr, gcr, lbl, col) in enumerate(zip(keys, sr_vals, gcr_vals, labels, colors)):
        # SR bar
        if sr is not None:
            bar = ax1.bar(i, sr, w, color=col, alpha=0.85, edgecolor='black', lw=0.6)
            ax1.text(i, sr + 0.005, f'{sr:.2f}', ha='center', va='bottom', fontsize=8)
        else:
            ax1.bar(i, 0.5, w, color='#dddddd', alpha=0.5, edgecolor='black', lw=0.6, linestyle='--')
            ax1.text(i, 0.52, 'TBD', ha='center', va='bottom', fontsize=8, color='gray')

        # GCR bar
        if gcr is not None:
            ax2.bar(i, gcr, w, color=col, alpha=0.85, edgecolor='black', lw=0.6)
            ax2.text(i, gcr + 0.05, f'{gcr:.2f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax2.bar(i, 5, w, color='#dddddd', alpha=0.5, edgecolor='black', lw=0.6, linestyle='--')
            ax2.text(i, 5.2, 'TBD', ha='center', va='bottom', fontsize=8, color='gray')

    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax1.set_ylabel('Success Rate (SR)', fontsize=11); ax1.set_ylim(0, 1.05)
    ax1.set_title('Success Rate — Ablation Comparison', fontsize=12)
    ax1.axhline(sr_vals[0] if sr_vals[0] else 0.92, color=colors[0],
                linestyle=':', lw=1.5, label='GRACE baseline')
    ax1.grid(axis='y', alpha=0.3); ax1.legend(fontsize=8)

    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=8)
    ax2.set_ylabel('Group Crossing Rate GCR (%)', fontsize=11)
    ax2.set_title('GCR — Ablation Comparison (lower is better)', fontsize=12)
    ax2.axhline(gcr_vals[0] if gcr_vals[0] else 0.07, color=colors[0],
                linestyle=':', lw=1.5, label='GRACE baseline')
    ax2.grid(axis='y', alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle('GRACE Ablation Study Results', fontsize=14, y=1.01)
    fig.tight_layout()

    out = os.path.join(args.out_dir, 'results_bar.png')
    fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")

# ── Plot: Trajectory ──────────────────────────────────────────────────────────

def _draw_trajectory_panel(ax, traj, human_init, goal_pos, title, color, arena=8.5):
    ax.set_xlim(-arena, arena); ax.set_ylim(-arena, arena)
    ax.set_aspect('equal'); ax.set_facecolor('#f8f8f8')
    ax.set_title(title, fontsize=10, pad=4)

    # arena boundary
    ax.add_patch(mpatches.Circle((0, 0), arena, fill=False,
                                  edgecolor='#aaaaaa', lw=1, linestyle='--'))

    # humans at episode start
    if human_init is not None:
        for hp in human_init:
            if np.linalg.norm(hp) < arena * 1.5:
                ax.add_patch(mpatches.Circle(hp, 0.3, color='#888888', alpha=0.5, zorder=3))

    # robot trajectory
    if len(traj) > 1:
        traj = np.array(traj)
        ax.plot(traj[:, 0], traj[:, 1], '-', color=color, lw=1.5, zorder=4, alpha=0.9)
        ax.plot(*traj[0], 'o', color=color, markersize=7, zorder=5, label='start')
        ax.plot(*traj[-1], 's', color=color, markersize=7, zorder=5, label='end')

    # start and goal
    ax.plot(0, 0, 'D', color='royalblue', markersize=8, zorder=6)
    if goal_pos is not None:
        ax.plot(*goal_pos, '*', color='green', markersize=12, zorder=6)

    ax.set_xticks([]); ax.set_yticks([])


def plot_trajectory_for(abl_key):
    print(f"\n[trajectory] Loading baseline vs {abl_key} ...")

    cols = ['baseline', abl_key] if abl_key != 'baseline' else ['baseline']
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, key in zip(axes, cols):
        cfg = ABLATIONS[key]
        ac, envs, rnn_hxs, grid_range = _load_model(cfg['dir'], device)
        if ac is None:
            ax.text(0.5, 0.5, f'{cfg["label"]}\n(best.pt not found)',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        traj, _, human_init, goal_pos = _run_episode(ac, envs, rnn_hxs, device, args.max_steps)
        envs.close()

        _draw_trajectory_panel(ax, traj, human_init, goal_pos,
                               cfg['label'], cfg['color'])

    fig.suptitle(f'Robot Trajectory — Seed {args.seed}', fontsize=13)
    fig.tight_layout()

    tag = abl_key if abl_key != 'baseline' else 'baseline'
    if args.video:
        _save_trajectory_video(abl_key, fig, axes, cols)
    else:
        out = os.path.join(args.out_dir, f'trajectory_{tag}.png')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


def _save_trajectory_video(abl_key, fig, axes, cols):
    from matplotlib.animation import FFMpegWriter

    all_data = {}
    for key in cols:
        cfg = ABLATIONS[key]
        ac, envs, rnn_hxs, _ = _load_model(cfg['dir'], device)
        if ac is None:
            all_data[key] = ([], None, None)
            continue
        traj, _, human_init, goal_pos = _run_episode(ac, envs, rnn_hxs, device, args.max_steps)
        envs.close()
        all_data[key] = (traj, human_init, goal_pos)

    out = os.path.join(args.out_dir, f'trajectory_{abl_key}.mp4')
    writer = FFMpegWriter(fps=args.fps, metadata={'title': f'Trajectory {abl_key}'})
    max_len = max(len(d[0]) for d in all_data.values())

    with writer.saving(fig, out, dpi=args.dpi):
        for step in range(max_len):
            for ax, key in zip(axes, cols):
                traj, human_init, goal_pos = all_data[key]
                partial = traj[:step+1] if step < len(traj) else traj
                _draw_trajectory_panel(ax, partial, human_init, goal_pos,
                                       ABLATIONS[key]['label'], ABLATIONS[key]['color'])
            writer.grab_frame()

    plt.close(fig)
    print(f"  Saved {out}")

# ── Plot: Cost Map ────────────────────────────────────────────────────────────

def plot_cost_map_for(abl_key):
    print(f"\n[cost_map] Loading baseline vs {abl_key} ...")
    cols  = ['baseline', abl_key] if abl_key != 'baseline' else ['baseline']
    n_col = len(cols)

    # 9 channels per column, 3 rows × 3 cols per model
    fig = plt.figure(figsize=(6 * n_col, 7))
    outer = fig.add_gridspec(1, n_col, wspace=0.35)

    model_data = {}
    for key in cols:
        cfg = ABLATIONS[key]
        ac, envs, rnn_hxs, grid_range = _load_model(cfg['dir'], device)
        if ac is None:
            model_data[key] = (None, grid_range or 6.0)
            continue
        _, cost_frames, _, _ = _run_episode(ac, envs, rnn_hxs, device, args.max_steps)
        envs.close()
        mid = len(cost_frames) // 2
        chosen = next((c for c in cost_frames[mid:] if c is not None), None) or \
                 next((c for c in cost_frames if c is not None), None)
        model_data[key] = (chosen, grid_range)

    for col_idx, key in enumerate(cols):
        cost_map, gr = model_data[key]
        inner = outer[col_idx].subgridspec(3, 3, wspace=0.3, hspace=0.45)
        title_ax = fig.add_subplot(outer[col_idx])
        title_ax.set_title(ABLATIONS[key]['label'], fontsize=11, pad=20)
        title_ax.axis('off')

        for ch in range(9):
            ax = fig.add_subplot(inner[ch // 3, ch % 3])
            if cost_map is not None and ch < cost_map.shape[0]:
                ax.imshow(cost_map[ch], origin='lower',
                          extent=[-gr, gr, -gr, gr],
                          cmap=CH_CMAPS[ch], vmin=0, vmax=1,
                          aspect='equal', interpolation='nearest')
                ax.plot(0, 0, 'b^', markersize=4, zorder=5)
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=ax.transAxes, color='gray')
            ax.set_title(CH_NAMES[ch], fontsize=7, pad=2)
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f'GRACE Cost Map Channels — Seed {args.seed}', fontsize=13, y=1.01)

    tag = abl_key if abl_key != 'baseline' else 'baseline'
    if args.video:
        _save_cost_map_video(abl_key, cols)
        plt.close(fig)
    else:
        out = os.path.join(args.out_dir, f'cost_map_{tag}.png')
        fig.savefig(out, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {out}")


def _save_cost_map_video(abl_key, cols):
    all_frames, grid_ranges = {}, {}
    for key in cols:
        cfg = ABLATIONS[key]
        ac, envs, rnn_hxs, gr = _load_model(cfg['dir'], device)
        grid_ranges[key] = gr or 6.0
        if ac is None:
            all_frames[key] = []
            continue
        _, cost_frames, _, _ = _run_episode(ac, envs, rnn_hxs, device, args.max_steps)
        envs.close()
        all_frames[key] = cost_frames

    n_col   = len(cols)
    fig     = plt.figure(figsize=(6 * n_col, 7))
    outer   = fig.add_gridspec(1, n_col, wspace=0.35)
    max_len = max(len(f) for f in all_frames.values())
    out     = os.path.join(args.out_dir, f'cost_map_{abl_key}.mp4')
    writer  = FFMpegWriter(fps=args.fps, metadata={'title': f'Cost map {abl_key}'})

    with writer.saving(fig, out, dpi=args.dpi):
        for step in range(max_len):
            fig.clf()
            outer = fig.add_gridspec(1, n_col, wspace=0.35)
            for col_idx, key in enumerate(cols):
                frames = all_frames[key]
                gr     = grid_ranges[key]
                cost_map = frames[step] if step < len(frames) else None
                inner = outer[col_idx].subgridspec(3, 3, wspace=0.3, hspace=0.45)
                title_ax = fig.add_subplot(outer[col_idx])
                title_ax.set_title(f'{ABLATIONS[key]["label"]} — step {step}', fontsize=10, pad=18)
                title_ax.axis('off')
                for ch in range(9):
                    ax = fig.add_subplot(inner[ch // 3, ch % 3])
                    if cost_map is not None and ch < cost_map.shape[0]:
                        ax.imshow(cost_map[ch], origin='lower',
                                  extent=[-gr, gr, -gr, gr],
                                  cmap=CH_CMAPS[ch], vmin=0, vmax=1,
                                  aspect='equal', interpolation='nearest')
                        ax.plot(0, 0, 'b^', markersize=3, zorder=5)
                    else:
                        ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                                transform=ax.transAxes, color='gray')
                    ax.set_title(CH_NAMES[ch], fontsize=6, pad=1)
                    ax.set_xticks([]); ax.set_yticks([])
            writer.grab_frame()

    plt.close(fig)
    print(f"  Saved {out}")

# ── Main dispatch ─────────────────────────────────────────────────────────────

if 'training_curves' in plot_types:
    plot_training_curves()

if 'results_bar' in plot_types:
    plot_results_bar()

if 'trajectory' in plot_types:
    for key in ablation_keys:
        if key == 'baseline' and len(ablation_keys) > 1:
            continue   # baseline only shown alongside an ablation
        plot_trajectory_for(key)

if 'cost_map' in plot_types:
    for key in ablation_keys:
        if key == 'baseline' and len(ablation_keys) > 1:
            continue
        plot_cost_map_for(key)

print(f"\nAll figures saved to: {args.out_dir}/")
