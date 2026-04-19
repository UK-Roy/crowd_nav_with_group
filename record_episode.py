"""
Record a paper-quality video of one navigation episode.

Usage:
    python record_episode.py                          # default: ORCA, mixed env
    python record_episode.py --policy social_force
    python record_episode.py --seed 5 --out my_video.mp4
    python record_episode.py --groups-only
    python record_episode.py --fps 10 --dpi 200

Output: MP4 (H.264, yuv420p) suitable for IEEE/IROS/RA-L paper submission.
"""
import sys, argparse
sys.path.insert(0, '.')
sys.path.insert(0, 'crowd_nav')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyArrowPatch, ArrowStyle

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--policy',       default='orca',
                    choices=['orca', 'social_force'],
                    help='Navigation policy for the robot')
parser.add_argument('--seed',         type=int, default=0)
parser.add_argument('--out',          default='episode.mp4')
parser.add_argument('--fps',          type=int, default=10,
                    help='Frames per second (10 = real-time at dt=0.1)')
parser.add_argument('--dpi',          type=int, default=150,
                    help='Resolution (150 = good quality, 200 = print quality)')
parser.add_argument('--max-steps',    type=int, default=300)
parser.add_argument('--groups-only',  action='store_true')
parser.add_argument('--individuals-only', action='store_true')
parser.add_argument('--group-types',  default='static_f,dynamic_lf,dynamic_free',
                    help='Comma-separated list of group types to include')
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
from crowd_nav.configs.config import Config
config = Config()
config.sim.predict_method = 'none'
config.env.use_wrapper = False
config.sim.has_individuals = not args.groups_only
config.sim.has_groups      = not args.individuals_only
config.group.types = args.group_types.split(',')

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

# ── Policy ────────────────────────────────────────────────────────────────────
from crowd_sim.envs.utils.state import JointState
if args.policy == 'orca':
    from crowd_nav.policy.orca import ORCA
    policy = ORCA(config)
else:
    from crowd_nav.policy.social_force import SocialForce
    policy = SocialForce(config)
policy.time_step = config.env.time_step
robot.policy = policy
robot.kinematics = config.action_space.kinematics

# ── Color palette ─────────────────────────────────────────────────────────────
GROUP_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
TYPE_HATCH   = {'static_f': '///', 'dynamic_lf': '', 'dynamic_free': '...'}
TYPE_LABEL   = {'static_f': 'Static (F-form)', 'dynamic_lf': 'Dynamic LF', 'dynamic_free': 'Dynamic Free'}

# ── Figure setup ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('white')
ax.set_facecolor('#f8f8f8')
ax.set_xlim(-9, 9)
ax.set_ylim(-9, 9)
ax.set_aspect('equal')
ax.set_xlabel('x (m)', fontsize=11)
ax.set_ylabel('y (m)', fontsize=11)
ax.tick_params(labelsize=9)
ax.grid(True, linestyle='--', alpha=0.4, linewidth=0.5)

# ── Reset ─────────────────────────────────────────────────────────────────────
obs = env.reset()
print(f"Seed {args.seed} | {env.human_num} humans | groups:")
for g in env.grp:
    if g.members:
        print(f"  Group {g.id}: {g.group_type}, {len(g.members)} members")

# ── Legend items ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor='gold', edgecolor='black', label='Robot'),
    mlines.Line2D([], [], marker='*', color='red',   linestyle='None', markersize=12, label='Goal'),
    mpatches.Patch(facecolor='none', edgecolor='black', label='Individual'),
]
active_types = set()
for g in env.grp:
    if g.members and g.group_type:
        active_types.add(g.group_type)
for i, g in enumerate(env.grp):
    if g.members:
        color = GROUP_COLORS[g.id % len(GROUP_COLORS)]
        lbl = f"Group {g.id} ({TYPE_LABEL.get(g.group_type, g.group_type)})"
        legend_handles.append(
            mpatches.Patch(facecolor=color, alpha=0.7,
                           hatch=TYPE_HATCH.get(g.group_type, ''),
                           edgecolor='black', label=lbl))

ax.legend(handles=legend_handles, loc='upper right', fontsize=7.5,
          framealpha=0.85, edgecolor='gray')

# ── Animation writer ─────────────────────────────────────────────────────────
writer = FFMpegWriter(
    fps=args.fps,
    codec='libx264',
    bitrate=4000,
    extra_args=['-pix_fmt', 'yuv420p',   # broadest player compatibility
                '-crf', '18',             # near-lossless quality (0=lossless, 23=default)
                '-preset', 'slow'],       # better compression
)

drawn = []

def clear_frame():
    for a in drawn:
        try:
            a.remove()
        except Exception:
            pass
    drawn.clear()
    for t in list(ax.texts):
        t.remove()

def draw_state(step_num, reward, done):
    clear_frame()

    # --- goal ---
    star, = ax.plot(robot.gx, robot.gy, marker='*', color='red',
                    markersize=14, linestyle='None', zorder=10)
    drawn.append(star)

    # --- robot ---
    rc = plt.Circle((robot.px, robot.py), robot.radius,
                    facecolor='gold', edgecolor='black', linewidth=1.2, zorder=8)
    ax.add_patch(rc)
    drawn.append(rc)

    # robot heading arrow
    spd = np.hypot(robot.vx, robot.vy)
    if spd > 0.05:
        theta = np.arctan2(robot.vy, robot.vx)
        ax.annotate('', xy=(robot.px + 0.45*np.cos(theta),
                             robot.py + 0.45*np.sin(theta)),
                    xytext=(robot.px, robot.py),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                    zorder=9)

    # --- humans ---
    for h in env.humans:
        gid = getattr(h, 'group_id', None)
        if gid is not None:
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            grp = env.grp[gid] if gid < len(env.grp) else None
            hatch = TYPE_HATCH.get(getattr(grp, 'group_type', ''), '') if grp else ''
            c = plt.Circle((h.px, h.py), h.radius,
                            facecolor=color, alpha=0.75,
                            hatch=hatch, edgecolor='black', linewidth=0.8, zorder=5)
        else:
            c = plt.Circle((h.px, h.py), h.radius,
                            facecolor='lightgray', edgecolor='black',
                            linewidth=0.8, zorder=5)
        ax.add_patch(c)
        drawn.append(c)

        # velocity arrow
        spd_h = np.hypot(h.vx, h.vy)
        if spd_h > 0.05 and not h.isObstacle:
            theta_h = np.arctan2(h.vy, h.vx)
            arr = ax.annotate('',
                xy=(h.px + 0.35*np.cos(theta_h), h.py + 0.35*np.sin(theta_h)),
                xytext=(h.px, h.py),
                arrowprops=dict(arrowstyle='->', color='dimgray', lw=0.9),
                zorder=6)
            drawn.append(arr)

    # --- convex hull outlines ---
    if env.group_hulls:
        for gid, hull in env.group_hulls.items():
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            if hull._kind == 'polygon':
                poly = plt.Polygon(hull._verts, fill=False, edgecolor=color,
                                   linestyle='--', linewidth=1.8, zorder=7)
                ax.add_patch(poly)
                drawn.append(poly)
            else:
                hc = plt.Circle(hull.centroid, hull.bounding_radius,
                                fill=False, edgecolor=color,
                                linestyle='--', linewidth=1.8, zorder=7)
                ax.add_patch(hc)
                drawn.append(hc)

    # --- info text ---
    status = 'DONE' if done else f'r={reward:+.3f}'
    t = ax.text(0.02, 0.98, f't={step_num*config.env.time_step:.1f}s  step={step_num}  {status}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    drawn.append(t)

    ax.set_title(f'Policy: {args.policy.upper()} | Seed: {args.seed}', fontsize=11)

# ── Record ────────────────────────────────────────────────────────────────────
print(f"Recording → {args.out}  (fps={args.fps}, dpi={args.dpi}, max_steps={args.max_steps})")

with writer.saving(fig, args.out, dpi=args.dpi):
    draw_state(0, 0.0, False)
    writer.grab_frame()

    for step in range(1, args.max_steps + 1):
        state = JointState(
            robot.get_full_state(),
            [env.humans[i].get_observable_state() for i in range(env.human_num)])
        action = robot.policy.predict(state)
        ob, reward, done, info = env.step(action)

        draw_state(step, reward, done)
        writer.grab_frame()

        if done:
            # hold last frame for 1 second
            for _ in range(args.fps):
                writer.grab_frame()
            print(f"  Episode ended at step {step}: {info['info']}")
            break

print(f"Saved: {args.out}")
plt.close(fig)
