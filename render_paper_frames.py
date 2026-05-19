"""
Render paper-quality snapshot frames of a GRACE navigation episode.

Saves individual PNG files at specified timesteps AND an optional
multi-panel figure suitable for a paper.

Usage examples:
    # Default: GRACE, seed 0, frames at steps 0,20,40,60,80
    python render_paper_frames.py

    # Custom steps and seed
    python render_paper_frames.py --seed 3 --steps 0,15,30,50,80

    # Show robot trajectory trail
    python render_paper_frames.py --trail

    # Multi-panel figure only (no individual files)
    python render_paper_frames.py --panel-only

    # Change output directory
    python render_paper_frames.py --out-dir figures/episode_frames
"""
import sys, os, argparse
sys.path.insert(0, '.')
sys.path.insert(0, 'crowd_nav')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import torch

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--seed',       type=int,   default=0)
parser.add_argument('--steps',      default='0,20,40,60,80',
                    help='Comma-separated timestep indices to capture')
parser.add_argument('--model-dir',  default='trained_models/gram_map/stageC')
parser.add_argument('--checkpoint', default='best.pt')
parser.add_argument('--out-dir',    default='results/paper_frames')
parser.add_argument('--dpi',        type=int,   default=300,
                    help='Output DPI (300 = print quality)')
parser.add_argument('--trail',      action='store_true',
                    help='Draw robot trajectory trail across frames')
parser.add_argument('--panel-only', action='store_true',
                    help='Only save the multi-panel figure, skip individual PNGs')
parser.add_argument('--max-steps',  type=int,   default=200)
args = parser.parse_args()

CAPTURE_STEPS = sorted(int(s) for s in args.steps.split(','))
os.makedirs(args.out_dir, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
import importlib
cfg_mod = importlib.import_module(args.model_dir.replace('/', '.') + '.configs.config')
config  = cfg_mod.Config()

# ── Environment ───────────────────────────────────────────────────────────────
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.robot import Robot

env = CrowdSimVarNum()
env.configure(config)
robot = Robot(config, 'robot')
env.set_robot(robot)
env.thisSeed = args.seed
env.nenv = 1
env.phase = 'test'

# ── Load GRACE model ──────────────────────────────────────────────────────────
device = torch.device('cpu')

arg_mod  = importlib.import_module(args.model_dir.replace('/', '.') + '.arguments')
get_args = getattr(arg_mod, 'get_args')
algo_args = get_args()

from rl.networks.model import Policy
obs_space = env.observation_space
actor_critic = Policy(
    obs_space.spaces,
    env.action_space,
    base_kwargs={'config': config, 'base': config.robot.policy})
ckpt_path = os.path.join(args.model_dir, 'checkpoints', args.checkpoint)
actor_critic.load_state_dict(
    torch.load(ckpt_path, map_location=device), strict=False)
actor_critic.eval()

base     = actor_critic.base
node_num = 1
edge_num = base.human_num + 1
hidden = {
    'human_node_rnn':       torch.zeros(1, node_num, base.human_node_rnn_size),
    'human_human_edge_rnn': torch.zeros(1, edge_num, base.human_human_edge_rnn_size),
}

def obs_to_tensor(obs):
    out = {}
    for key, val in obs.items():
        if key == 'group_members':
            out[key] = val; continue
        if key not in obs_space.spaces: continue
        expected = obs_space.spaces[key].shape
        arr = np.array(val).reshape(expected)
        t   = torch.from_numpy(arr).float()
        out[key] = t.unsqueeze(0)
    return out

# ── Drawing constants ─────────────────────────────────────────────────────────
GROUP_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
TYPE_HATCH   = {'static_f': '///', 'dynamic_lf': '', 'dynamic_free': '...'}
TYPE_LABEL   = {'static_f': 'Static F-form', 'dynamic_lf': 'Dynamic LF', 'dynamic_free': 'Dynamic Free'}
R = config.sim.circle_radius

def draw_frame(ax, env, robot, step, trail_pts=None):
    """Draw one snapshot onto ax."""
    ax.cla()
    ax.set_facecolor('#f8f8f8')
    ax.set_xlim(-R - 0.5, R + 0.5)
    ax.set_ylim(-R - 0.5, R + 0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle='--', alpha=0.35, linewidth=0.5)
    ax.set_title(f't = {step * config.env.time_step:.1f} s', fontsize=10, fontweight='bold')

    # robot trajectory trail
    if trail_pts and len(trail_pts) > 1:
        xs, ys = zip(*trail_pts)
        ax.plot(xs, ys, color='gold', linewidth=1.2, alpha=0.6, zorder=3, linestyle='-')

    # goal
    ax.plot(robot.gx, robot.gy, marker='*', color='red',
            markersize=13, linestyle='None', zorder=10)

    # humans
    for h in env.humans:
        gid = getattr(h, 'group_id', None)
        if gid is not None:
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            grp   = env.grp[gid] if gid < len(env.grp) else None
            hatch = TYPE_HATCH.get(getattr(grp, 'group_type', ''), '') if grp else ''
            c = plt.Circle((h.px, h.py), h.radius, facecolor=color, alpha=0.75,
                           hatch=hatch, edgecolor='black', linewidth=0.7, zorder=5)
        else:
            c = plt.Circle((h.px, h.py), h.radius, facecolor='lightgray',
                           edgecolor='black', linewidth=0.7, zorder=5)
        ax.add_patch(c)
        if np.hypot(h.vx, h.vy) > 0.05 and not getattr(h, 'isObstacle', False):
            th = np.arctan2(h.vy, h.vx)
            ax.annotate('', xy=(h.px + 0.3*np.cos(th), h.py + 0.3*np.sin(th)),
                        xytext=(h.px, h.py),
                        arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.8), zorder=6)

    # group hull boundaries
    for gid, hull in (getattr(env, 'group_hulls', None) or {}).items():
        color = GROUP_COLORS[gid % len(GROUP_COLORS)]
        if getattr(hull, '_kind', '') == 'polygon':
            poly = plt.Polygon(hull._verts, fill=False, edgecolor=color,
                               linestyle='--', linewidth=1.6, zorder=7)
            ax.add_patch(poly)
        else:
            hc = plt.Circle(hull.centroid, hull.bounding_radius, fill=False,
                            edgecolor=color, linestyle='--', linewidth=1.6, zorder=7)
            ax.add_patch(hc)

    # robot (drawn last so it's on top)
    rc = plt.Circle((robot.px, robot.py), robot.radius,
                    facecolor='gold', edgecolor='black', linewidth=1.2, zorder=8)
    ax.add_patch(rc)
    spd = np.hypot(robot.vx, robot.vy)
    if spd > 0.05:
        th = np.arctan2(robot.vy, robot.vx)
        ax.annotate('', xy=(robot.px + 0.4*np.cos(th), robot.py + 0.4*np.sin(th)),
                    xytext=(robot.px, robot.py),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=9)

def make_legend(ax, env):
    handles = [
        mpatches.Patch(facecolor='gold',      edgecolor='black', label='Robot'),
        mlines.Line2D([], [], marker='*', color='red', linestyle='None',
                      markersize=10, label='Goal'),
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Individual'),
    ]
    for g in env.grp:
        if g.members:
            col = GROUP_COLORS[g.id % len(GROUP_COLORS)]
            lbl = f"Grp {g.id}: {TYPE_LABEL.get(g.group_type, g.group_type)}"
            handles.append(mpatches.Patch(facecolor=col, alpha=0.75,
                hatch=TYPE_HATCH.get(g.group_type, ''), edgecolor='black', label=lbl))
    ax.legend(handles=handles, loc='upper right', fontsize=7, framealpha=0.85, edgecolor='gray')

# ── Run episode ───────────────────────────────────────────────────────────────
obs  = env.reset()
done = False
step = 0
trail = []  # robot (px, py) at each step

captured = {}   # step → snapshot data (env state, robot state)

print(f"Seed {args.seed} | {env.human_num} humans")
for g in env.grp:
    if g.members:
        print(f"  Group {g.id}: {g.group_type}, {len(g.members)} members")

while not done and step <= args.max_steps:
    trail.append((robot.px, robot.py))

    if step in CAPTURE_STEPS:
        # Save figure for this step
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('white')
        draw_frame(ax, env, robot, step, trail if args.trail else None)
        make_legend(ax, fig.axes[0])
        fig.tight_layout()
        if not args.panel_only:
            out_path = os.path.join(args.out_dir, f'frame_step{step:04d}.png')
            fig.savefig(out_path, dpi=args.dpi, bbox_inches='tight')
            print(f"Saved: {out_path}")
        captured[step] = fig

    # step environment
    obs_t = obs_to_tensor(obs)
    masks = torch.ones(1, 1)
    with torch.no_grad():
        _, action_t, _, hidden = actor_critic.act(obs_t, hidden, masks, deterministic=True)
    action_np = action_t.squeeze().cpu().numpy()
    obs, rew, done, info = env.step(action_np)
    step += 1

    if done:
        outcome = info.get('info', 'done')
        print(f"Episode ended at step {step}: {outcome}")

# capture any remaining requested steps that weren't reached (episode ended early)
# also capture final state
if step - 1 not in captured and step - 1 in CAPTURE_STEPS:
    fig, ax = plt.subplots(figsize=(5, 5))
    fig.patch.set_facecolor('white')
    draw_frame(ax, env, robot, step - 1, trail if args.trail else None)
    make_legend(ax, ax)
    fig.tight_layout()
    captured[step - 1] = fig

# ── Multi-panel figure ────────────────────────────────────────────────────────
saved_steps = sorted(captured.keys())
n = len(saved_steps)
if n > 0:
    fig_panel, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
    if n == 1:
        axes = [axes]
    fig_panel.patch.set_facecolor('white')

    for i, (s, ax_p) in enumerate(zip(saved_steps, axes)):
        # Re-draw into panel axes — re-run draw from saved figure
        src_ax = captured[s].axes[0]
        # Copy by re-running draw_frame; we need the env state at that step
        # Since we can't rewind, we redraw from the single-frame figures
        # by copying their content
        ax_p.set_aspect('equal')
        ax_p.set_title(f't = {s * config.env.time_step:.1f} s',
                       fontsize=10, fontweight='bold')

    # The panel uses individual saved figures - just note the paths
    panel_path = os.path.join(args.out_dir, 'panel.png')

    # Build panel from individual frame PNGs using imread
    import matplotlib.image as mpimg
    frame_paths = [os.path.join(args.out_dir, f'frame_step{s:04d}.png') for s in saved_steps]
    if all(os.path.exists(p) for p in frame_paths):
        plt.close(fig_panel)
        fig_panel, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.8))
        if n == 1: axes = [axes]
        fig_panel.patch.set_facecolor('white')
        for ax_p, fp, s in zip(axes, frame_paths, saved_steps):
            img = mpimg.imread(fp)
            ax_p.imshow(img)
            ax_p.axis('off')
            ax_p.set_title(f't = {s * config.env.time_step:.1f} s',
                           fontsize=10, fontweight='bold')
        fig_panel.tight_layout(pad=0.5)
        fig_panel.savefig(panel_path, dpi=args.dpi, bbox_inches='tight')
        print(f"\nSaved panel: {panel_path}")
    else:
        print("Panel skipped (run without --panel-only to generate individual frames first)")

print(f"\nDone. Frames in: {args.out_dir}/")
