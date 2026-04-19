"""
Record and compare multiple navigation policies on identical scenarios.

Each policy is run on the same seeds so group types, positions, and crowd
behaviour are bit-for-bit identical — the only variable is the robot policy.

Usage:
    # Record ORCA vs Social Force on 3 seeds
    python record_comparison.py --policies orca,social_force --seeds 0,1,2

    # Single policy, single seed (same as record_episode.py but with metrics)
    python record_comparison.py --policies orca --seeds 3

    # High-quality + all baseline policies
    python record_comparison.py --policies orca,social_force,zone_based,f_formation --seeds 0,1 --dpi 200

Output per policy×seed:
    videos/<policy>_seed<N>.mp4   — video
    results/metrics.csv           — aggregated metrics table
"""
import sys, argparse, os, csv, copy
sys.path.insert(0, '.')
sys.path.insert(0, 'crowd_nav')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.animation import FFMpegWriter

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--policies',   default='orca,social_force',
                    help='Comma-separated policy names from policy_factory')
parser.add_argument('--seeds',      default='0,1,2',
                    help='Comma-separated seeds (same seed = identical scenario)')
parser.add_argument('--max-steps',  type=int, default=300)
parser.add_argument('--fps',        type=int, default=10)
parser.add_argument('--dpi',        type=int, default=150)
parser.add_argument('--group-types', default='static_f,dynamic_lf,dynamic_free')
parser.add_argument('--no-video',   action='store_true',
                    help='Skip video recording, only collect metrics')
args = parser.parse_args()

POLICIES = [p.strip() for p in args.policies.split(',')]
SEEDS    = [int(s.strip()) for s in args.seeds.split(',')]

os.makedirs('videos',  exist_ok=True)
os.makedirs('results', exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
from crowd_nav.configs.config import Config
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_nav.policy.policy_factory import policy_factory

GROUP_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
TYPE_HATCH   = {'static_f': '///', 'dynamic_lf': '', 'dynamic_free': '...'}
TYPE_LABEL   = {'static_f': 'Static F-form', 'dynamic_lf': 'Dynamic LF',
                'dynamic_free': 'Dynamic Free'}

# ── Build policy ──────────────────────────────────────────────────────────────
def make_policy(name, config):
    cls = policy_factory.get(name)
    if cls is None:
        raise ValueError(f"Unknown policy '{name}'. Available: {list(policy_factory.keys())}")
    p = cls(config)
    if hasattr(p, 'time_step'):
        p.time_step = config.env.time_step
    return p

# ── Render helpers ────────────────────────────────────────────────────────────
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

def draw_state(ax, env, robot, config, step_num, reward, done, policy_name):
    clear_frame(ax)
    # goal
    s, = ax.plot(robot.gx, robot.gy, marker='*', color='red',
                 markersize=14, linestyle='None', zorder=10)
    drawn.append(s)
    # robot
    rc = plt.Circle((robot.px, robot.py), robot.radius,
                    facecolor='gold', edgecolor='black', linewidth=1.2, zorder=8)
    ax.add_patch(rc); drawn.append(rc)
    spd = np.hypot(robot.vx, robot.vy)
    if spd > 0.05:
        theta = np.arctan2(robot.vy, robot.vx)
        arr = ax.annotate('', xy=(robot.px + 0.45*np.cos(theta),
                                   robot.py + 0.45*np.sin(theta)),
                          xytext=(robot.px, robot.py),
                          arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                          zorder=9)
        drawn.append(arr)
    # humans
    for h in env.humans:
        gid = getattr(h, 'group_id', None)
        if gid is not None:
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            grp = env.grp[gid] if gid < len(env.grp) else None
            hatch = TYPE_HATCH.get(getattr(grp, 'group_type', ''), '') if grp else ''
            c = plt.Circle((h.px, h.py), h.radius, facecolor=color, alpha=0.75,
                            hatch=hatch, edgecolor='black', linewidth=0.8, zorder=5)
        else:
            c = plt.Circle((h.px, h.py), h.radius, facecolor='lightgray',
                            edgecolor='black', linewidth=0.8, zorder=5)
        ax.add_patch(c); drawn.append(c)
        if np.hypot(h.vx, h.vy) > 0.05 and not h.isObstacle:
            th = np.arctan2(h.vy, h.vx)
            a = ax.annotate('', xy=(h.px+0.35*np.cos(th), h.py+0.35*np.sin(th)),
                            xytext=(h.px, h.py),
                            arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.9),
                            zorder=6)
            drawn.append(a)
    # group hulls
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
    # info overlay
    status = 'DONE' if done else f'r={reward:+.3f}'
    t = ax.text(0.02, 0.98,
                f't={step_num*config.env.time_step:.1f}s  step={step_num}  {status}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    drawn.append(t)
    ax.set_title(f'Policy: {policy_name.upper()}', fontsize=12, fontweight='bold')

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
            lbl = f"Grp {g.id}: {TYPE_LABEL.get(g.group_type, g.group_type)}"
            handles.append(mpatches.Patch(
                facecolor=color, alpha=0.7,
                hatch=TYPE_HATCH.get(g.group_type, ''),
                edgecolor='black', label=lbl))
    ax.legend(handles=handles, loc='upper right', fontsize=7.5,
              framealpha=0.85, edgecolor='gray')

# ── Metrics tracking ──────────────────────────────────────────────────────────
def compute_gcr(robot, env):
    """Fraction of group hulls the robot is currently inside."""
    rp = np.array([robot.px, robot.py])
    if not env.group_hulls:
        return 0
    return sum(1 for h in env.group_hulls.values() if h.contains(rp))

# ── Main loop ─────────────────────────────────────────────────────────────────
all_metrics = []

for policy_name in POLICIES:
    print(f"\n{'='*50}")
    print(f"Policy: {policy_name.upper()}")
    print(f"{'='*50}")

    for seed in SEEDS:
        config = Config()
        config.sim.predict_method = 'none'
        config.env.use_wrapper = False
        config.group.types = args.group_types.split(',')

        env = CrowdSimVarNum()
        env.configure(config)
        robot = Robot(config, 'robot')

        try:
            pol = make_policy(policy_name, config)
        except Exception as e:
            print(f"  Skipping {policy_name}: {e}")
            continue

        robot.policy = pol
        robot.kinematics = config.action_space.kinematics
        env.set_robot(robot)
        env.thisSeed = seed
        env.nenv = 1
        env.phase = 'test'

        obs = env.reset()
        print(f"  Seed {seed} | groups: {[(g.id, g.group_type) for g in env.grp if g.members]}")

        # metric accumulators
        total_reward = 0.0
        gcr_steps    = 0      # steps robot is inside any group hull
        collisions   = 0
        success      = False
        timeout      = False
        n_steps      = 0

        # video setup
        vid_path = f"videos/{policy_name}_seed{seed}.mp4"
        if not args.no_video:
            fig, ax = make_figure()
            build_legend(ax, env)
            writer = FFMpegWriter(fps=args.fps, codec='libx264', bitrate=4000,
                                  extra_args=['-pix_fmt', 'yuv420p',
                                              '-crf', '18', '-preset', 'slow'])
            writer_ctx = writer.saving(fig, vid_path, dpi=args.dpi)
            writer_ctx.__enter__()
            draw_state(ax, env, robot, config, 0, 0.0, False, policy_name)
            writer.grab_frame()

        for step in range(1, args.max_steps + 1):
            state = JointState(
                robot.get_full_state(),
                [env.humans[i].get_observable_state() for i in range(env.human_num)])
            action = pol.predict(state)
            ob, reward, done, info = env.step(action)

            total_reward += reward
            n_steps      += 1
            gcr_steps    += compute_gcr(robot, env)

            info_obj = info['info']
            if hasattr(info_obj, '__class__'):
                cls = info_obj.__class__.__name__
                if cls == 'Collision':
                    collisions += 1
                elif cls == 'ReachGoal':
                    success = True
                elif cls == 'Timeout':
                    timeout = True

            if not args.no_video:
                draw_state(ax, env, robot, config, step, reward, done, policy_name)
                writer.grab_frame()

            if done:
                if not args.no_video:
                    for _ in range(args.fps):  # hold 1 s
                        writer.grab_frame()
                break

        # GCR = fraction of steps robot was inside a group
        gcr = gcr_steps / max(n_steps, 1)

        result = dict(
            policy=policy_name, seed=seed,
            success=int(success), collision=int(collisions > 0),
            timeout=int(timeout), n_steps=n_steps,
            total_reward=round(total_reward, 3),
            gcr=round(gcr, 4),
        )
        all_metrics.append(result)
        print(f"    → success={success}  collisions={collisions}  "
              f"steps={n_steps}  GCR={gcr:.3f}  reward={total_reward:.2f}")

        if not args.no_video:
            writer_ctx.__exit__(None, None, None)
            plt.close(fig)
            print(f"    Saved: {vid_path}")

# ── Write CSV ─────────────────────────────────────────────────────────────────
csv_path = 'results/metrics.csv'
if all_metrics:
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
        w.writeheader()
        w.writerows(all_metrics)
    print(f"\nMetrics saved → {csv_path}")

    # Print summary table
    print(f"\n{'Policy':<20} {'SR':>5} {'CR':>5} {'Avg Steps':>10} {'Avg GCR':>9} {'Avg Reward':>11}")
    print('-' * 65)
    for pol in POLICIES:
        rows = [r for r in all_metrics if r['policy'] == pol]
        if not rows: continue
        sr  = np.mean([r['success']      for r in rows])
        cr  = np.mean([r['collision']    for r in rows])
        st  = np.mean([r['n_steps']      for r in rows])
        gcr = np.mean([r['gcr']          for r in rows])
        rew = np.mean([r['total_reward'] for r in rows])
        print(f"{pol:<20} {sr:>5.2f} {cr:>5.2f} {st:>10.1f} {gcr:>9.4f} {rew:>11.2f}")
