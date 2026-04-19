"""
Record and compare navigation policies on identical scenarios.

Supports:
  - Classical policies (ORCA, Social Force, Zone-based, F-formation)
  - Neural network policies loaded from checkpoints (GARN, DS-RNN/GRAM, etc.)
  - Each classical/neural policy optionally wrapped with TAGA (+taga variant)
  - GARN is never wrapped with TAGA (it has its own group-awareness)

Policy config is defined in POLICY_REGISTRY at the bottom of this file.
Each entry specifies: display name, policy key, optional model_dir,
and whether to also produce a TAGA variant.

Usage:
    # All registered policies (defined below)
    python record_comparison.py --seeds 0,1,2

    # Subset by name
    python record_comparison.py --policies orca,orca+taga,garn --seeds 0,1,2

    # High quality, more seeds
    python record_comparison.py --seeds 0,1,2,3,4 --dpi 200 --fps 15

    # Metrics only (fast, no video rendering)
    python record_comparison.py --seeds 0,1,2,3,4 --no-video

Output:
    videos/<policy_label>_seed<N>.mp4   — paper-quality video
    results/metrics.csv                  — SR, CR, GCR, reward per run
    (summary table printed to terminal)
"""
import sys, argparse, os, csv, copy
sys.path.insert(0, '.')
sys.path.insert(0, 'crowd_nav')

import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.animation import FFMpegWriter

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--policies',  default=None,
                    help='Comma-separated policy labels to run (default: all in registry)')
parser.add_argument('--seeds',     default='0,1,2',
                    help='Comma-separated integer seeds')
parser.add_argument('--max-steps', type=int, default=300)
parser.add_argument('--fps',       type=int, default=10)
parser.add_argument('--dpi',       type=int, default=150)
parser.add_argument('--no-video',  action='store_true')
parser.add_argument('--group-types', default='static_f,dynamic_lf,dynamic_free')
args = parser.parse_args()

SEEDS = [int(s.strip()) for s in args.seeds.split(',')]
os.makedirs('videos',  exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Policy registry ───────────────────────────────────────────────────────────
# Each entry:
#   label      : short name used in filenames and table headers
#   policy_key : key in config.robot.policy (also in policy_factory)
#   model_dir  : path to trained checkpoint dir (None for classical)
#   with_taga  : if True, also produce a "<label>+taga" variant
#
# ADD NEW POLICIES HERE — no other code changes needed.
POLICY_REGISTRY = [
    dict(label='orca',           policy_key='orca',                    model_dir=None,
         with_taga=True),
    dict(label='social_force',   policy_key='social_force',            model_dir=None,
         with_taga=True),
    dict(label='srnn',           policy_key='srnn',
         model_dir='trained_models/srnn',                              with_taga=True),
    dict(label='intention_rl',   policy_key='selfAttn_merge_srnn',
         model_dir='trained_models/GST_predictor_rand',                with_taga=True),
    dict(label='gram',           policy_key='selfAttn_merge_srnn_grpAttn',
         model_dir='trained_models/my_model',                          with_taga=True),
    dict(label='garn',           policy_key='garn',
         model_dir='trained_models/garn',                              with_taga=False),
]

# ── Filter registry by --policies flag ───────────────────────────────────────
def expand_registry(registry):
    """Expand each entry into base + optional TAGA variant."""
    entries = []
    for r in registry:
        entries.append(dict(label=r['label'], policy_key=r['policy_key'],
                            model_dir=r['model_dir'], use_taga=False))
        if r['with_taga']:
            entries.append(dict(label=r['label'] + '+taga', policy_key=r['policy_key'],
                                model_dir=r['model_dir'], use_taga=True))
    return entries

ALL_ENTRIES = expand_registry(POLICY_REGISTRY)

if args.policies:
    wanted = {p.strip() for p in args.policies.split(',')}
    ALL_ENTRIES = [e for e in ALL_ENTRIES if e['label'] in wanted]
    missing = wanted - {e['label'] for e in ALL_ENTRIES}
    if missing:
        print(f"Warning: unknown policies requested: {missing}")
        print(f"Available: {[e['label'] for e in ALL_ENTRIES]}")

print(f"Running {len(ALL_ENTRIES)} policy variants × {len(SEEDS)} seeds")

# ── Imports ───────────────────────────────────────────────────────────────────
from crowd_nav.configs.config import Config
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.taga_safety import TAGASafetyController
from rl.networks.model import Policy as NetPolicy

# ── Neural network loader ─────────────────────────────────────────────────────
def load_neural_policy(model_dir, config, device):
    """Load a trained actor-critic checkpoint from model_dir."""
    import importlib, glob

    # Load the training-time arguments
    arg_module = importlib.import_module(model_dir.replace('/', '.') + '.arguments')
    algo_args  = arg_module.get_args()

    # Find latest checkpoint if not specified
    ckpt_dir = os.path.join(model_dir, 'checkpoints')
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, '*.pt')))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    load_path = ckpts[-1]
    print(f"    Loading checkpoint: {os.path.basename(load_path)}")

    actor_critic = NetPolicy(
        config.robot.policy,
        config,
        algo_args,
    )
    actor_critic.load_state_dict(torch.load(load_path, map_location=device))
    actor_critic.base.nenv = 1
    actor_critic = nn.DataParallel(actor_critic).to(device)
    actor_critic.eval()
    return actor_critic, algo_args

# ── Classical policy loader ───────────────────────────────────────────────────
def load_classical_policy(policy_key, config):
    cls = policy_factory.get(policy_key)
    if cls is None:
        raise ValueError(f"Unknown policy '{policy_key}'")
    p = cls(config)
    if hasattr(p, 'time_step'):
        p.time_step = config.env.time_step
    return p

# ── TAGA helpers ─────────────────────────────────────────────────────────────
def _sigmoid_alpha(d_group, d_switch, band):
    """Sigmoid blend: 1 when close, 0 when far. Smooth S-curve, no step jump."""
    return 1.0 / (1.0 + math.exp((d_group - d_switch) / max(band / 3.0, 1e-6)))


def _obstacle_cost_np(tang_unit, rx, ry, env, skip_centroid_xy, taga_cfg):
    """Obstacle density in a forward cone along tang_unit."""
    cos_thresh = math.cos(math.radians(taga_cfg.cone_half_angle))
    look_ahead = taga_cfg.look_ahead
    cost = 0.0
    for h in env.humans:
        dx, dy = h.px - rx, h.py - ry
        d = math.hypot(dx, dy)
        if d < 1e-3 or d > look_ahead:
            continue
        rel_unit = np.array([dx, dy]) / d
        if np.dot(tang_unit, rel_unit) > cos_thresh:
            cost += 1.0 / (d + 0.1)
    for (cx, cy), r in zip(env.group_centroids or [], env.group_radii or []):
        if abs(cx - skip_centroid_xy[0]) < 1e-3 and abs(cy - skip_centroid_xy[1]) < 1e-3:
            continue
        dx, dy = cx - rx, cy - ry
        d = math.hypot(dx, dy)
        if d > look_ahead + r:
            continue
        rel_unit = np.array([dx, dy]) / (d + 1e-9)
        if np.dot(tang_unit, rel_unit) > cos_thresh:
            cost += 2.0 / (d + 0.1)
    return cost


# ── TAGA action computation ───────────────────────────────────────────────────
def apply_taga(obs, base_action_np, robot, env, safety_ctrl, config, device):
    """Blend base_action with TAGA tangent when robot is near a group."""
    from crowd_sim.envs.utils.action import ActionXY

    taga_cfg = config.taga
    v_pref   = robot.v_pref

    centroids = env.group_centroids or []
    radii     = env.group_radii    or []

    if not centroids:
        return ActionXY(*base_action_np)

    rx, ry = robot.px, robot.py
    gx, gy = robot.gx, robot.gy
    d_robot_goal = math.hypot(gx - rx, gy - ry)
    goal_dir = np.array([gx - rx, gy - ry])
    goal_dir /= (np.linalg.norm(goal_dir) + 1e-9)

    # Outer goal priority check — if any group passes this, skip TAGA
    goal_priority_fire = any(
        d_robot_goal < taga_cfg.goal_threshold and
        d_robot_goal < math.hypot((c[0] if hasattr(c,'__len__') else c) - gx,
                                   (c[1] if hasattr(c,'__len__') else 0) - gy)
        for c, _ in zip(centroids, radii)
    )

    best_action = base_action_np.copy()
    if not goal_priority_fire:
        # P2: collect ALL blocking groups
        blocking = []
        for centroid, radius in zip(centroids, radii):
            cx, cy = (centroid[0], centroid[1]) if hasattr(centroid, '__len__') else centroid
            d_centroid      = math.hypot(cx - rx, cy - ry)
            d_centroid_goal = math.hypot(cx - gx, cy - gy)

            robot_to_group = np.array([cx - rx, cy - ry])
            if np.dot(goal_dir, robot_to_group) <= 0:
                continue
            if d_robot_goal < d_centroid_goal:
                continue

            # P0: sigmoid alpha
            d_switch = radius + taga_cfg.safe_margin
            if taga_cfg.smooth_switching:
                alpha = _sigmoid_alpha(d_centroid, d_switch, taga_cfg.switch_band)
            else:
                alpha = 1.0 if d_centroid < d_switch else 0.0
            if alpha < 1e-3:
                continue

            dx, dy   = cx - rx, cy - ry
            norm_dxy = math.hypot(dx, dy) + 1e-9
            cw_tang  = np.array([ dy, -dx]) / norm_dxy
            ccw_tang = np.array([-dy,  dx]) / norm_dxy

            # P1: cost-aware side selection
            if getattr(taga_cfg, 'cost_aware_side', False):
                cw_cost  = (taga_cfg.w_goal * (1 - np.dot(cw_tang,  goal_dir)) / 2
                            + taga_cfg.w_obstacle * _obstacle_cost_np(cw_tang,  rx, ry, env, (cx, cy), taga_cfg))
                ccw_cost = (taga_cfg.w_goal * (1 - np.dot(ccw_tang, goal_dir)) / 2
                            + taga_cfg.w_obstacle * _obstacle_cost_np(ccw_tang, rx, ry, env, (cx, cy), taga_cfg))
                tangent = cw_tang if cw_cost <= ccw_cost else ccw_tang
            else:
                cw_angle  = np.arccos(np.clip(np.dot(cw_tang,  goal_dir), -1, 1))
                ccw_angle = np.arccos(np.clip(np.dot(ccw_tang, goal_dir), -1, 1))
                tangent   = cw_tang if cw_angle < ccw_angle else ccw_tang

            blocking.append((alpha, tangent, d_centroid))

        if blocking:
            if getattr(taga_cfg, 'multi_group', False):
                # P2: weighted-average tangent, k-nearest groups
                blocking.sort(key=lambda x: x[2])
                blocking = blocking[:getattr(taga_cfg, 'max_groups', 3)]
                agg = np.zeros(2)
                total_w = 0.0
                for alpha, tangent, d_g in blocking:
                    w = alpha / (d_g + 1e-9)
                    agg     += w * tangent
                    total_w += w
                norm_agg      = np.linalg.norm(agg)
                tangent_final = agg / (norm_agg + 1e-9)
                alpha_final   = min(sum(a for a, _, _ in blocking), 1.0)
            else:
                blocking.sort(key=lambda x: x[2])
                alpha_final, tangent_final, _ = blocking[0]

            taga_action = tangent_final * v_pref
            desired     = alpha_final * taga_action + (1.0 - alpha_final) * base_action_np

            if obs is not None and safety_ctrl is not None:
                obs_tensor  = torch.FloatTensor(obs).unsqueeze(0).to(device)
                best_action = safety_ctrl.get_safe_taga_action(obs_tensor, desired, device)
            else:
                best_action = desired

    return ActionXY(*best_action) if not isinstance(best_action, type(ActionXY(0, 0))) else best_action

# ── Render helpers ────────────────────────────────────────────────────────────
GROUP_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
TYPE_HATCH   = {'static_f': '///', 'dynamic_lf': '', 'dynamic_free': '...'}
TYPE_LABEL   = {'static_f': 'Static F-form', 'dynamic_lf': 'Dynamic LF',
                'dynamic_free': 'Dynamic Free'}

def make_figure():
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')
    ax.set_xlim(-9, 9); ax.set_ylim(-9, 9)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)', fontsize=11)
    ax.set_ylabel('y (m)', fontsize=11)
    ax.tick_params(labelsize=9)
    ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)
    return fig, ax

drawn = []

def clear_frame(ax):
    for a in drawn:
        try: a.remove()
        except Exception: pass
    drawn.clear()
    for t in list(ax.texts): t.remove()

def draw_state(ax, env, robot, config, step_num, reward, done, label):
    clear_frame(ax)
    s, = ax.plot(robot.gx, robot.gy, marker='*', color='red',
                 markersize=14, linestyle='None', zorder=10)
    drawn.append(s)
    rc = plt.Circle((robot.px, robot.py), robot.radius,
                    facecolor='gold', edgecolor='black', linewidth=1.2, zorder=8)
    ax.add_patch(rc); drawn.append(rc)
    spd = np.hypot(robot.vx, robot.vy)
    if spd > 0.05:
        theta = np.arctan2(robot.vy, robot.vx)
        arr = ax.annotate('', xy=(robot.px + 0.45*np.cos(theta),
                                   robot.py + 0.45*np.sin(theta)),
                          xytext=(robot.px, robot.py),
                          arrowprops=dict(arrowstyle='->', color='black', lw=1.5), zorder=9)
        drawn.append(arr)
    for h in env.humans:
        gid = getattr(h, 'group_id', None)
        if gid is not None:
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            grp   = env.grp[gid] if gid < len(env.grp) else None
            hatch = TYPE_HATCH.get(getattr(grp, 'group_type', ''), '') if grp else ''
            c = plt.Circle((h.px, h.py), h.radius, facecolor=color, alpha=0.75,
                            hatch=hatch, edgecolor='black', linewidth=0.8, zorder=5)
        else:
            c = plt.Circle((h.px, h.py), h.radius, facecolor='lightgray',
                            edgecolor='black', linewidth=0.8, zorder=5)
        ax.add_patch(c); drawn.append(c)
        if np.hypot(h.vx, h.vy) > 0.05 and not h.isObstacle:
            th = np.arctan2(h.vy, h.vx)
            a  = ax.annotate('', xy=(h.px+0.35*np.cos(th), h.py+0.35*np.sin(th)),
                             xytext=(h.px, h.py),
                             arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.9), zorder=6)
            drawn.append(a)
    for gid, hull in (env.group_hulls or {}).items():
        color = GROUP_COLORS[gid % len(GROUP_COLORS)]
        if hull._kind == 'polygon':
            poly = plt.Polygon(hull._verts, fill=False, edgecolor=color,
                               linestyle='--', linewidth=1.8, zorder=7)
            ax.add_patch(poly); drawn.append(poly)
        else:
            hc = plt.Circle(hull.centroid, hull.bounding_radius, fill=False,
                            edgecolor=color, linestyle='--', linewidth=1.8, zorder=7)
            ax.add_patch(hc); drawn.append(hc)
    status = 'DONE' if done else f'r={reward:+.3f}'
    t = ax.text(0.02, 0.98,
                f't={step_num*config.env.time_step:.1f}s  step={step_num}  {status}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    drawn.append(t)
    taga_flag = ' [+TAGA]' if '+taga' in label else ''
    ax.set_title(f'Policy: {label.upper()}{taga_flag}', fontsize=12, fontweight='bold')

def build_legend(ax, env):
    handles = [
        mpatches.Patch(facecolor='gold',      edgecolor='black', label='Robot'),
        mlines.Line2D([], [], marker='*', color='red', linestyle='None',
                      markersize=12, label='Goal'),
        mpatches.Patch(facecolor='lightgray', edgecolor='black', label='Individual'),
    ]
    for g in env.grp:
        if g.members:
            color = GROUP_COLORS[g.id % len(GROUP_COLORS)]
            lbl   = f"Grp {g.id}: {TYPE_LABEL.get(g.group_type, g.group_type)}"
            handles.append(mpatches.Patch(facecolor=color, alpha=0.7,
                hatch=TYPE_HATCH.get(g.group_type,''), edgecolor='black', label=lbl))
    ax.legend(handles=handles, loc='upper right', fontsize=7.5,
              framealpha=0.85, edgecolor='gray')

# ── GCR ───────────────────────────────────────────────────────────────────────
def compute_gcr(robot, env):
    rp = np.array([robot.px, robot.py])
    return sum(1 for h in (env.group_hulls or {}).values() if h.contains(rp))

# ── Hidden state helpers for neural policies ──────────────────────────────────
def init_hidden(actor_critic, device):
    base  = actor_critic.module.base
    node_num = 1
    edge_num = base.human_num + 1
    return {
        'human_node_rnn':       torch.zeros(1, node_num, base.human_node_rnn_size, device=device),
        'human_human_edge_rnn': torch.zeros(1, edge_num, base.human_human_edge_rnn_size, device=device),
    }

# ── Main loop ─────────────────────────────────────────────────────────────────
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
all_metrics = []

for entry in ALL_ENTRIES:
    label      = entry['label']
    policy_key = entry['policy_key']
    model_dir  = entry['model_dir']
    use_taga   = entry['use_taga']
    is_neural  = model_dir is not None

    print(f"\n{'='*55}")
    print(f"Policy: {label.upper()}")
    print(f"{'='*55}")

    # Load neural model once (reused across seeds)
    actor_critic = None
    algo_args    = None
    if is_neural:
        if not os.path.isdir(model_dir):
            print(f"  SKIP — model_dir not found: {model_dir}")
            continue
        try:
            # Load config from model dir
            import importlib
            cfg_mod    = importlib.import_module(model_dir.replace('/', '.') + '.configs.config')
            run_config = cfg_mod.Config()
            actor_critic, algo_args = load_neural_policy(model_dir, run_config, device)
        except Exception as e:
            print(f"  SKIP — failed to load: {e}")
            continue

    for seed in SEEDS:
        # Fresh config and env per seed so RNG is identical across policies
        config = Config()
        config.sim.predict_method = 'none'
        config.env.use_wrapper    = False
        config.group.types        = args.group_types.split(',')
        config.robot.policy       = policy_key

        env    = CrowdSimVarNum()
        env.configure(config)
        robot  = Robot(config, 'robot')
        env.set_robot(robot)
        env.thisSeed = seed
        env.nenv     = 1
        env.phase    = 'test'

        # Policy
        if is_neural:
            robot.policy      = actor_critic  # stored for reference; actions computed below
            robot.kinematics  = config.action_space.kinematics
            hidden            = None          # initialised after reset
        else:
            pol               = load_classical_policy(policy_key, config)
            robot.policy      = pol
            robot.kinematics  = config.action_space.kinematics

        # TAGA controller
        safety_ctrl = TAGASafetyController(config) if use_taga else None

        obs = env.reset()
        print(f"  Seed {seed} | groups: {[(g.id, g.group_type) for g in env.grp if g.members]}")

        if is_neural:
            hidden = init_hidden(actor_critic, device)

        # Metrics
        total_reward = 0.0
        gcr_steps    = 0
        collisions   = 0
        success      = False
        timeout      = False
        n_steps      = 0

        # Video
        vid_path = f"videos/{label}_seed{seed}.mp4"
        if not args.no_video:
            fig, ax = make_figure()
            build_legend(ax, env)
            writer     = FFMpegWriter(fps=args.fps, codec='libx264', bitrate=4000,
                                      extra_args=['-pix_fmt','yuv420p','-crf','18','-preset','slow'])
            writer_ctx = writer.saving(fig, vid_path, dpi=args.dpi)
            writer_ctx.__enter__()
            draw_state(ax, env, robot, config, 0, 0.0, False, label)
            writer.grab_frame()

        for step in range(1, args.max_steps + 1):
            # ── Compute base action ──────────────────────────────────────────
            if is_neural:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                masks       = torch.zeros(1, 1, device=device)
                with torch.no_grad():
                    _, action_tensor, _, hidden = actor_critic(
                        obs_tensor, hidden, masks)
                base_action_np = action_tensor.squeeze().cpu().numpy()
            else:
                state          = JointState(robot.get_full_state(),
                    [env.humans[i].get_observable_state() for i in range(env.human_num)])
                base_act       = pol.predict(state)
                base_action_np = np.array([base_act.vx, base_act.vy])

            # ── Apply TAGA if enabled ────────────────────────────────────────
            if use_taga and safety_ctrl is not None:
                obs_for_taga = obs if is_neural else None
                final_action = apply_taga(obs_for_taga, base_action_np,
                                          robot, env, safety_ctrl, config, device)
            else:
                from crowd_sim.envs.utils.action import ActionXY
                final_action = ActionXY(*base_action_np)

            ob, reward, done, info = env.step(final_action)
            obs = ob

            total_reward += reward
            n_steps      += 1
            gcr_steps    += compute_gcr(robot, env)

            cls = info['info'].__class__.__name__
            if cls == 'Collision':   collisions += 1
            elif cls == 'ReachGoal': success     = True
            elif cls == 'Timeout':   timeout     = True

            if not args.no_video:
                draw_state(ax, env, robot, config, step, reward, done, label)
                writer.grab_frame()

            if done:
                if not args.no_video:
                    for _ in range(args.fps): writer.grab_frame()
                break

        gcr = gcr_steps / max(n_steps, 1)
        result = dict(policy=label, seed=seed, success=int(success),
                      collision=int(collisions > 0), timeout=int(timeout),
                      n_steps=n_steps, total_reward=round(total_reward, 3),
                      gcr=round(gcr, 4))
        all_metrics.append(result)
        print(f"    → success={success}  collisions={collisions}  "
              f"steps={n_steps}  GCR={gcr:.3f}  reward={total_reward:.2f}")

        if not args.no_video:
            writer_ctx.__exit__(None, None, None)
            plt.close(fig)
            print(f"    Saved: {vid_path}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
if all_metrics:
    csv_path = 'results/metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        w.writeheader()
        w.writerows(all_metrics)
    print(f"\nMetrics saved → {csv_path}")

    print(f"\n{'Policy':<25} {'SR':>5} {'CR':>5} {'TR':>5} {'Avg Steps':>10} {'Avg GCR':>9} {'Avg Reward':>11}")
    print('-' * 76)
    seen = []
    for entry in ALL_ENTRIES:
        lbl  = entry['label']
        if lbl in seen: continue
        seen.append(lbl)
        rows = [r for r in all_metrics if r['policy'] == lbl]
        if not rows: continue
        print(f"{lbl:<25} "
              f"{np.mean([r['success']      for r in rows]):>5.2f} "
              f"{np.mean([r['collision']    for r in rows]):>5.2f} "
              f"{np.mean([r['timeout']      for r in rows]):>5.2f} "
              f"{np.mean([r['n_steps']      for r in rows]):>10.1f} "
              f"{np.mean([r['gcr']          for r in rows]):>9.4f} "
              f"{np.mean([r['total_reward'] for r in rows]):>11.2f}")
