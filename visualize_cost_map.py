"""
visualize_cost_map.py — Render GRACE cost layers alongside the live environment.

Produces a side-by-side video:
  Left  : bird's-eye environment (robot, humans, group hulls)
  Right : 3×3 grid of the 9 cost-map channels

Usage:
    python visualize_cost_map.py \
        --model_dir trained_models/gram_map/stageC \
        --test_model 41600.pt \
        --seed 3 --out cost_map.mp4

Output: MP4 (H.264) at --fps frames per second.
"""

import sys, os, argparse
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FFMpegWriter

from crowd_sim import *
from rl.networks.envs import make_vec_envs
from rl.networks.model import Policy

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model_dir',  type=str, required=True)
parser.add_argument('--test_model', type=str, default='best.pt')
parser.add_argument('--seed',       type=int, default=0)
parser.add_argument('--out',        type=str, default='cost_map.mp4')
parser.add_argument('--fps',        type=int, default=8)
parser.add_argument('--dpi',        type=int, default=130)
parser.add_argument('--max-steps',  type=int, default=200)
parser.add_argument('--device',     type=str, default='auto',
                    help='Device to run on: "auto" (default), "cuda", "cpu", or "cuda:0"')
args = parser.parse_args()

# ── Load config from model directory ─────────────────────────────────────────
from importlib import import_module
model_dir = args.model_dir.rstrip('/')

try:
    Config   = getattr(import_module(model_dir.replace('/', '.') + '.configs.config'), 'Config')
    get_args = getattr(import_module(model_dir.replace('/', '.') + '.arguments'), 'get_args')
except Exception:
    from crowd_nav.configs.config import Config
    from arguments import get_args

algo_args              = get_args()
algo_args.num_processes  = 1
algo_args.num_mini_batch = 1

config = env_config = Config()
config.sim.render      = False
config.env.use_wrapper = False

# Force training env settings for visualization.
# The model dir config may have been updated for realistic benchmark evaluation
# (human_num=20, realistic.enabled=True) — revert to what the model was trained on.
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

# Backbone weights (GroupDetector + SlotAttention) are embedded in the main
# checkpoint via load_state_dict above — no separate backbone load needed.
# Calling load_frozen_backbones here would overwrite fine-tuned Stage C weights
# with the original Phase 2/3 checkpoint, which is wrong for evaluation.
if config.robot.policy == 'grace':
    for p in actor_critic.base.detector.parameters():
        p.requires_grad_(False)
    for p in actor_critic.base.slot_attn.parameters():
        p.requires_grad_(False)
    actor_critic.base.detector.eval()
    actor_critic.base.slot_attn.eval()

# ── Initial hidden states ────────────────────────────────────────────────────
# Use the network's own size attributes, not algo_args — the CLI defaults for
# human_node_rnn_size (128) and human_human_edge_rnn_size (256) differ from
# the training values (256 and 14). The network knows the right sizes.
rnn_hxs = {
    'human_node_rnn':       torch.zeros(1, 1,
                                        actor_critic.base.human_node_rnn_size, device=device),
    'human_human_edge_rnn': torch.zeros(1, actor_critic.base.human_num + 1,
                                        actor_critic.base.human_human_edge_rnn_size, device=device),
}
masks = torch.zeros(1, 1, device=device)

# ── Channel metadata ──────────────────────────────────────────────────────────
CH_NAMES = ['Individual\n(current)', 'Traj 0.3 s', 'Traj 0.7 s',
            'Traj 1.0 s', 'Traj 1.5 s', 'Group\nCohesion',
            'Group\nRepulsion', 'Goal\nAttractor', 'Boundary']
CH_CMAPS = ['Reds', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd',
            'Purples', 'RdPu', 'Greens', 'Blues']

# ── Figure layout ─────────────────────────────────────────────────────────────
# Left col (width 5): environment   |   Right 3×3 (width 9): cost channels
fig = plt.figure(figsize=(14, 6))
gs  = fig.add_gridspec(3, 4, wspace=0.35, hspace=0.45)

ax_env  = fig.add_subplot(gs[:, 0])       # full left column
ax_cost = [fig.add_subplot(gs[r, c + 1])  # 3×3 grid on the right
           for r in range(3) for c in range(3)]

def _setup_env_ax(ax, r):
    ax.set_xlim(-r, r); ax.set_ylim(-r, r)
    ax.set_aspect('equal'); ax.set_facecolor('#f0f0f0')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('Environment (robot-centric)', fontsize=8)

def _draw_env(ax, obs_dict, r):
    ax.cla()
    _setup_env_ax(ax, r)

    # human positions from spatial_edges (robot-centric)
    sp = obs_dict['spatial_edges'][0].cpu().numpy()        # (N, 2)
    vm = obs_dict['visible_masks'][0].cpu().numpy().astype(bool)  # (N,)
    for i, (pos, vis) in enumerate(zip(sp, vm)):
        if vis and np.linalg.norm(pos) < r * 1.5:
            c = mpatches.Circle(pos, 0.3, color='#888888', alpha=0.7, zorder=3)
            ax.add_patch(c)

    # robot at origin
    robot_patch = mpatches.Circle((0, 0), 0.3, color='royalblue', zorder=5)
    ax.add_patch(robot_patch)

    # goal direction from robot_node [px,py,radius,gx,gy,v_pref,theta]
    rn = obs_dict['robot_node'][0, 0].cpu().numpy()  # (7,)
    goal_rel = rn[3:5] - rn[0:2]
    dist = np.linalg.norm(goal_rel)
    if dist > 0.1:
        arrow_end = goal_rel / dist * min(dist, r * 0.8)
        ax.annotate('', xy=arrow_end, xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='green', lw=1.5))

    # arena boundary
    boundary = mpatches.Circle((0, 0), r, fill=False, edgecolor='black', lw=1, linestyle='--')
    ax.add_patch(boundary)

    ax.plot(goal_rel[0], goal_rel[1], '*', color='green', markersize=10, zorder=6)
    ax.text(0.02, 0.98, f'Step {step}', transform=ax.transAxes,
            fontsize=7, va='top', ha='left')

def _draw_cost_channels(cost_map):
    C = cost_map.shape[0]
    for idx, ax in enumerate(ax_cost):
        ax.cla()
        if idx >= C:
            ax.axis('off')
            continue
        layer = cost_map[idx]
        ax.imshow(layer, origin='lower',
                  extent=[-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE],
                  cmap=CH_CMAPS[idx], vmin=0, vmax=1, aspect='equal',
                  interpolation='nearest')
        ax.plot(0, 0, 'b^', markersize=4, zorder=5)   # robot
        ax.set_title(CH_NAMES[idx] if idx < len(CH_NAMES) else f'Ch{idx}',
                     fontsize=7, pad=2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-GRID_RANGE, GRID_RANGE)
        ax.set_ylim(-GRID_RANGE, GRID_RANGE)

# ── Run episode ───────────────────────────────────────────────────────────────
obs   = envs.reset()
masks = torch.zeros(1, 1, device=device)

writer = FFMpegWriter(fps=args.fps,
                      metadata={'title': 'GRACE cost map', 'artist': 'GRACE'})

print(f"Recording to {args.out} ...")
step = 0
with writer.saving(fig, args.out, dpi=args.dpi):
    for step in range(args.max_steps):
        obs_current = obs   # save current obs — cost map and env must be in sync

        with torch.no_grad():
            _, action, _, rnn_hxs = actor_critic.act(
                obs_current, rnn_hxs, masks, deterministic=True)

        cost_map = None
        if hasattr(actor_critic.base, '_last_cost_stack') and \
                actor_critic.base._last_cost_stack is not None:
            cost_map = actor_critic.base._last_cost_stack[0].numpy()  # (C, H, W)

        obs, _, done, _ = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done[0] else [1.0]]).to(device)

        if cost_map is not None:
            _draw_env(ax_env, obs_current, GRID_RANGE)   # draw same frame as cost map
            _draw_cost_channels(cost_map)
            fig.suptitle('GRACE: Environment vs Cost Stack', fontsize=9, y=1.01)
            writer.grab_frame()

        if done[0]:
            print(f"Episode ended at step {step + 1}")
            break

print(f"Done. Saved {args.out}")
envs.close()
