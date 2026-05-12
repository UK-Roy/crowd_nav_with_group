"""
visualize_grace.py — Rich GRACE visualization for paper and presentation.

Three-panel layout per frame:
  LEFT   : Ground-truth environment
             · Each group = distinct vivid color; members share color + group hull
             · Individual humans = grey
             · Robot = blue ▲, Goal = ★, Arena = dashed circle
             · Full legend
  MIDDLE : Robot's learned perception (slot attention)
             · Humans colored by dominant slot assignment (learned grouping)
             · Group-cohesion cost overlay as background heatmap
             · Undetected / low-confidence humans = grey
             · Legend
  RIGHT  : 3×3 cost-channel grid (all 9 channels labeled)

Usage:
    python visualize_grace.py \\
        --model_dir trained_models/gram_map/stageC \\
        --test_model 41665.pt \\
        --seed 3 --out grace_vis.mp4 --max-steps 400
"""

import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

from crowd_sim import *
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',  type=str, required=True)
parser.add_argument('--test_model', type=str, default='41665.pt')
parser.add_argument('--seed',       type=int, default=0)
parser.add_argument('--out',        type=str, default='grace_vis.mp4')
parser.add_argument('--fps',        type=int, default=8)
parser.add_argument('--dpi',        type=int, default=130)
parser.add_argument('--max-steps',  type=int, default=400)
parser.add_argument('--device',     type=str, default='auto',
                    help='Device to run on: "auto" (default), "cuda", "cpu", or "cuda:0"')
parser.add_argument('--show-hulls',    dest='show_hulls', action='store_true',  default=True,
                    help='Show group convex hull polygons (default: on).')
parser.add_argument('--no-show-hulls', dest='show_hulls', action='store_false',
                    help='Hide group convex hull polygons.')
args = parser.parse_args()

# ── Load config ───────────────────────────────────────────────────────────────
from importlib import import_module
model_dir = args.model_dir.rstrip('/')
try:
    Config   = getattr(import_module(model_dir.replace('/', '.') + '.configs.config'), 'Config')
    get_args = getattr(import_module(model_dir.replace('/', '.') + '.arguments'), 'get_args')
except Exception:
    from crowd_nav.configs.config import Config
    from arguments import get_args

algo_args = get_args()
algo_args.num_processes  = 1
algo_args.num_mini_batch = 1

config = env_config = Config()
config.sim.render      = False
config.env.use_wrapper = False

# Force training env settings — model dir config may have realistic benchmark settings
config.sim.human_num     = 15
config.sim.circle_radius = 6.0
config.sim.arena_size    = 6.0
config.group.num_groups  = 2
config.group.num_on_path = 1
config.group.types       = ['static_f', 'dynamic_lf']
_r = getattr(config, 'realistic', None)
if _r is not None:
    _r.enabled = False

if config.robot.policy == 'grace':
    algo_args._grace_cfg         = config.grace
    algo_args.grace_use_aux_loss = False

torch.manual_seed(args.seed)
if args.device == 'auto':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device(args.device)
print(f"[GRACE] Running on device: {device}"
      + (" (GPU)" if device.type == 'cuda' else " (CPU)"))

env_name_file = os.path.join(model_dir, 'env_name.txt')
env_name = open(env_name_file).read().strip() if os.path.exists(env_name_file) else 'CrowdSimVarNum-v0'
GRID_RANGE = getattr(config.grace, 'grid_range', 6.0)

# ── Build env and policy ──────────────────────────────────────────────────────
envs = make_vec_envs(env_name, args.seed, 1, algo_args.gamma, None, device,
                     False, config=env_config, ax=None, pretext_wrapper=False)

actor_critic = Policy(
    envs.observation_space.spaces,
    envs.action_space,
    base_kwargs=algo_args,
    base=config.robot.policy,
)
load_path = os.path.join(model_dir, 'checkpoints', args.test_model)
actor_critic.load_state_dict(torch.load(load_path, map_location=device), strict=False)
actor_critic.to(device).eval()

# Freeze backbone — weights already embedded in checkpoint
if config.robot.policy == 'grace':
    for p in actor_critic.base.detector.parameters():
        p.requires_grad_(False)
    for p in actor_critic.base.slot_attn.parameters():
        p.requires_grad_(False)
    actor_critic.base.detector.eval()
    actor_critic.base.slot_attn.eval()

# ── rnn_hxs — use network's own size attributes (not algo_args defaults) ─────
rnn_hxs = {
    'human_node_rnn':       torch.zeros(1, 1,
                                        actor_critic.base.human_node_rnn_size,       device=device),
    'human_human_edge_rnn': torch.zeros(1, actor_critic.base.human_num + 1,
                                        actor_critic.base.human_human_edge_rnn_size, device=device),
}
masks = torch.zeros(1, 1, device=device)

# ── Access inner env for ground-truth group info ──────────────────────────────
def _get_inner_env(vecenv):
    e = vecenv
    while hasattr(e, 'venv'):
        e = e.venv
    if hasattr(e, 'envs'):
        e = e.envs[0]
    while hasattr(e, 'env'):
        e = e.env
    return e

inner_env = _get_inner_env(envs)

# ── Color palettes ────────────────────────────────────────────────────────────
# Ground-truth group colors — identical to record_episode.py
GROUP_COLORS    = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
TYPE_HATCH      = {'static_f': '///', 'dynamic_lf': '', 'dynamic_free': '...'}
TYPE_LABEL      = {'static_f': 'Static (F-form)', 'dynamic_lf': 'Dynamic LF',
                   'dynamic_free': 'Dynamic Free'}

# Slot colors for robot perception panel (K=3 slots)
SLOT_COLORS     = ['#e74c3c', '#2980b9', '#27ae60']   # red, blue, green
SLOT_UNDETECTED = '#bdc3c7'                             # light grey

ARENA_LIM = GRID_RANGE + 2.5   # axis limits for env panel (arena=6 → ±8.5)

# Cost channel metadata
CH_NAMES = ['Individual\n(now)', 'Traj 0.3s', 'Traj 0.7s',
            'Traj 1.0s', 'Traj 1.5s', 'Group\nCohesion',
            'Group\nRepulsion', 'Goal\nAttractor', 'Boundary']
CH_CMAPS = ['Reds', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd',
            'Purples', 'RdPu', 'Greens', 'Blues']

# ── Figure layout ─────────────────────────────────────────────────────────────
# Columns: [env(2), perception(2), cost×3(1,1,1)]  rows: 3
fig = plt.figure(figsize=(20, 8))
gs  = fig.add_gridspec(3, 5, width_ratios=[2, 2, 1, 1, 1],
                        wspace=0.40, hspace=0.45,
                        left=0.04, right=0.98, top=0.93, bottom=0.20)

ax_env     = fig.add_subplot(gs[:, 0])
ax_percept = fig.add_subplot(gs[:, 1])
ax_cost    = [fig.add_subplot(gs[r, c + 2]) for r in range(3) for c in range(3)]

# ── Ground-truth environment panel setup (called once after reset) ────────────
drawn_env = []   # dynamic artists cleared each frame

def _setup_env_ax():
    """Permanent axis setup: limits, labels, grid, legend. Called once after reset."""
    ax_env.set_xlim(-ARENA_LIM, ARENA_LIM)
    ax_env.set_ylim(-ARENA_LIM, ARENA_LIM)
    ax_env.set_aspect('equal')
    ax_env.set_facecolor('#f8f8f8')
    ax_env.set_xlabel('x (m)', fontsize=10)
    ax_env.set_ylabel('y (m)', fontsize=10)
    ax_env.tick_params(labelsize=8)
    ax_env.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

    legend_handles = [
        mpatches.Patch(facecolor='gold', edgecolor='black', label='Robot'),
        Line2D([], [], marker='*', color='red', linestyle='None', markersize=10, label='Goal'),
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Individual'),
    ]
    if hasattr(inner_env, 'grp'):
        for g in inner_env.grp:
            if g.members:
                col = GROUP_COLORS[g.id % len(GROUP_COLORS)]
                lbl = f"Group {g.id} ({TYPE_LABEL.get(g.group_type, g.group_type)})"
                legend_handles.append(
                    mpatches.Patch(facecolor=col, alpha=0.75,
                                   hatch=TYPE_HATCH.get(g.group_type, ''),
                                   edgecolor='black', label=lbl))
    ax_env.legend(handles=legend_handles, loc='upper center',
                  bbox_to_anchor=(0.5, -0.08), ncol=3,
                  fontsize=6.5, framealpha=0.90, edgecolor='gray')

def _clear_env():
    for a in drawn_env:
        try:
            a.remove()
        except Exception:
            pass
    drawn_env.clear()
    for t in list(ax_env.texts):
        t.remove()

# ── Draw ground-truth environment (absolute world coordinates) ─────────────────
def _draw_env(step, reward, done):
    _clear_env()

    robot  = inner_env.robot
    humans = getattr(inner_env, 'humans', [])

    # Goal — absolute position
    star, = ax_env.plot(robot.gx, robot.gy, marker='*', color='red',
                        markersize=13, linestyle='None', zorder=10)
    drawn_env.append(star)

    # Robot circle + heading arrow
    rc = plt.Circle((robot.px, robot.py), robot.radius,
                    facecolor='gold', edgecolor='black', linewidth=1.2, zorder=8)
    ax_env.add_patch(rc)
    drawn_env.append(rc)
    spd = np.hypot(robot.vx, robot.vy)
    if spd > 0.05:
        theta = np.arctan2(robot.vy, robot.vx)
        arr = ax_env.annotate(
            '', xy=(robot.px + 0.45*np.cos(theta), robot.py + 0.45*np.sin(theta)),
            xytext=(robot.px, robot.py),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=9)
        drawn_env.append(arr)

    # Humans: group members (hatched, colored) + individuals (lightgray)
    for h in humans:
        gid = getattr(h, 'group_id', None)
        if gid is not None:
            col   = GROUP_COLORS[gid % len(GROUP_COLORS)]
            grp   = inner_env.grp[gid] if hasattr(inner_env, 'grp') and gid < len(inner_env.grp) else None
            hatch = TYPE_HATCH.get(getattr(grp, 'group_type', ''), '') if grp else ''
            c = plt.Circle((h.px, h.py), h.radius,
                           facecolor=col, alpha=0.75,
                           hatch=hatch, edgecolor='black', linewidth=0.8, zorder=5)
        else:
            c = plt.Circle((h.px, h.py), h.radius,
                           facecolor='lightgray', edgecolor='black',
                           linewidth=0.8, zorder=5)
        ax_env.add_patch(c)
        drawn_env.append(c)

        # Velocity arrow
        spd_h = np.hypot(h.vx, h.vy)
        if spd_h > 0.05 and not getattr(h, 'isObstacle', False):
            theta_h = np.arctan2(h.vy, h.vx)
            arr = ax_env.annotate(
                '', xy=(h.px + 0.35*np.cos(theta_h), h.py + 0.35*np.sin(theta_h)),
                xytext=(h.px, h.py),
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.9), zorder=6)
            drawn_env.append(arr)

    # Group hull outlines (polygon or circle fallback)
    if args.show_hulls and hasattr(inner_env, 'group_hulls') and inner_env.group_hulls:
        for gid, hull in inner_env.group_hulls.items():
            col = GROUP_COLORS[gid % len(GROUP_COLORS)]
            if getattr(hull, '_kind', '') == 'polygon':
                poly = plt.Polygon(hull._verts, fill=False, edgecolor=col,
                                   linestyle='--', linewidth=1.8, zorder=7)
                ax_env.add_patch(poly)
                drawn_env.append(poly)
            else:
                hc = plt.Circle(hull.centroid, hull.bounding_radius,
                                fill=False, edgecolor=col,
                                linestyle='--', linewidth=1.8, zorder=7)
                ax_env.add_patch(hc)
                drawn_env.append(hc)

    # Status text
    status = 'DONE' if done else f'r={reward:+.3f}'
    ts = getattr(inner_env, 'time_step', 0.25)
    t = ax_env.text(0.02, 0.98,
                    f't={step * ts:.1f}s  step={step}  {status}',
                    transform=ax_env.transAxes, fontsize=9, va='top',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    drawn_env.append(t)
    ax_env.set_title(f'GRACE (SR=89%) | Seed: {args.seed}',
                     fontsize=9, fontweight='bold')

# ── Draw robot perception (slot assignments) ──────────────────────────────────
def _draw_perception(ax, obs_dict, alpha_kn, cost_map, step):
    """
    alpha_kn : (K, N) numpy — slot-to-human assignment weights
    cost_map  : (C, H, W) numpy — cost channels
    """
    ax.cla()
    ax.set_xlim(-GRID_RANGE, GRID_RANGE)
    ax.set_ylim(-GRID_RANGE, GRID_RANGE)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f0f0')
    ax.set_xticks([]); ax.set_yticks([])

    # Group-cohesion channel (ch5, index 5) as background heatmap
    if cost_map is not None and cost_map.shape[0] > 5:
        grp_layer = cost_map[5]   # group cohesion
        ax.imshow(grp_layer, origin='lower',
                  extent=[-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
                  cmap='Purples', vmin=0, vmax=1, alpha=0.35,
                  aspect='equal', interpolation='bilinear', zorder=1)

    sp = obs_dict['spatial_edges'][0].cpu().numpy()
    vm = obs_dict['visible_masks'][0].cpu().numpy().astype(bool)
    N  = sp.shape[0]

    # Arena boundary
    ax.add_patch(mpatches.Circle((0, 0), GRID_RANGE, fill=False,
                                  edgecolor='#7f8c8d', lw=1.2, linestyle='--', zorder=2))

    # For each visible human: color by dominant slot, size by confidence
    for idx in range(N):
        if not vm[idx] or np.linalg.norm(sp[idx]) >= GRID_RANGE * 1.4:
            continue
        pos = sp[idx]
        if alpha_kn is not None and idx < alpha_kn.shape[1]:
            weights = alpha_kn[:, idx]          # (K,) softmax weights
            dom_slot = int(np.argmax(weights))
            conf     = float(weights[dom_slot])  # 0..1
            if conf > 0.45:
                col   = SLOT_COLORS[dom_slot % len(SLOT_COLORS)]
                alpha = 0.5 + 0.5 * conf
                label = f'S{dom_slot}'
            else:
                col   = SLOT_UNDETECTED
                alpha = 0.6
                label = '?'
        else:
            col, alpha, label = SLOT_UNDETECTED, 0.6, '?'

        c = mpatches.Circle(pos, 0.3, color=col, alpha=alpha, zorder=3)
        ax.add_patch(c)
        ax.text(pos[0], pos[1], label, fontsize=5,
                ha='center', va='center', color='white', fontweight='bold', zorder=4)

    # Individual channel (ch0) as individual cost dots
    if cost_map is not None:
        ind_layer = cost_map[0]
        # Just use the background; skip per-human cost labels to keep it clean

    # Robot (gold circle, same as env panel)
    rc = plt.Circle((0, 0), 0.25, facecolor='gold', edgecolor='black', linewidth=1.0, zorder=6)
    ax.add_patch(rc)

    # Goal arrow (robot-centric — robot always at origin in this panel)
    rn = obs_dict['robot_node'][0, 0].cpu().numpy()
    goal_rel = rn[3:5] - rn[0:2]
    dist = np.linalg.norm(goal_rel)
    if dist > 0.05:
        arrow_end = goal_rel / dist * min(dist, GRID_RANGE * 0.85)
        ax.annotate('', xy=arrow_end, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.8))
    ax.plot(goal_rel[0], goal_rel[1], '*', color='red',
            markersize=13, zorder=7)

    # Legend
    K = len(SLOT_COLORS)
    legend_elems = [
        mpatches.Patch(facecolor='gold', edgecolor='black', label='Robot'),
        Line2D([0],[0], marker='*', color='red', linestyle='None', markersize=8, label='Goal'),
        mpatches.Patch(facecolor=SLOT_UNDETECTED, alpha=0.8, label='Low-conf / individual'),
    ]
    for k in range(K):
        legend_elems.append(mpatches.Patch(facecolor=SLOT_COLORS[k], alpha=0.9,
                                            label=f'Group slot {k}'))
    legend_elems.append(mpatches.Patch(facecolor='#b39ddb', alpha=0.35,
                                        label='Group-cohesion cost\n(background)'))
    ax.legend(handles=legend_elems, loc='upper center',
              bbox_to_anchor=(0.5, -0.08), ncol=3,
              fontsize=6, framealpha=0.90, borderpad=0.4)

    ax.set_title('Robot Perception (Slot Attention)\nS0/S1/S2 = learned group slots',
                 fontsize=8, fontweight='bold', pad=4)

# ── Draw 3×3 cost channel grid ────────────────────────────────────────────────
def _draw_cost_channels(cost_map):
    C = cost_map.shape[0] if cost_map is not None else 0
    for idx, ax in enumerate(ax_cost):
        ax.cla()
        if cost_map is None or idx >= C:
            ax.axis('off')
            continue
        layer = cost_map[idx]
        im = ax.imshow(layer, origin='lower',
                       extent=[-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
                       cmap=CH_CMAPS[idx], vmin=0, vmax=1, aspect='equal',
                       interpolation='nearest')
        ax.plot(0, 0, '^', color='navy', markersize=4, zorder=5)
        name = CH_NAMES[idx] if idx < len(CH_NAMES) else f'Ch{idx}'
        peak = float(layer.max())
        ax.set_title(f'{name}\npeak={peak:.2f}', fontsize=6, pad=2)
        ax.set_xticks([]); ax.set_yticks([])

# ── Run episode ───────────────────────────────────────────────────────────────
obs   = envs.reset()
masks = torch.zeros(1, 1, device=device)

# Permanent axis setup (needs group info from reset)
_setup_env_ax()

writer = FFMpegWriter(fps=args.fps,
                      metadata={'title': 'GRACE visualization', 'artist': 'GRACE'})

print(f"Recording to {args.out} ...")
step     = 0
reward   = 0.0
done_flag = False

with writer.saving(fig, args.out, dpi=args.dpi):
    for step in range(args.max_steps):
        obs_current = obs

        with torch.no_grad():
            _, action, _, rnn_hxs = actor_critic.act(
                obs_current, rnn_hxs, masks, deterministic=True)

        # Extract cost map and slot assignments (set by forward() during act)
        cost_map = None
        if hasattr(actor_critic.base, '_last_cost_stack') and \
                actor_critic.base._last_cost_stack is not None:
            cost_map = actor_critic.base._last_cost_stack[0].numpy()  # (C, H, W)

        alpha_kn = None
        if hasattr(actor_critic.base, '_last_alpha') and \
                actor_critic.base._last_alpha is not None:
            alpha_kn = actor_critic.base._last_alpha[0].numpy()   # (K, N)

        obs, rew, done, infos = envs.step(action)
        reward    = float(rew[0]) if rew is not None else 0.0
        done_flag = bool(done[0])
        masks     = torch.FloatTensor([[0.0] if done_flag else [1.0]]).to(device)

        # Draw all panels
        _draw_env(step, reward, done_flag)
        _draw_perception(ax_percept, obs_current, alpha_kn, cost_map, step)
        _draw_cost_channels(cost_map)

        fig.suptitle('GRACE: Ground-Truth vs Learned Perception vs Cost Stack',
                     fontsize=9, fontweight='bold', y=0.99)

        writer.grab_frame()

        if done_flag:
            info = infos[0].get('info', '') if infos else ''
            print(f"Episode ended at step {step + 1} — {info}")
            break

print(f"Done. Saved {args.out}")
envs.close()
