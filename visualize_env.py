"""
Quick visual check of the realistic crowd environment.
Groups are shown as colored filled circles (one color per group).
Individuals are shown as open black circles.
Robot is yellow. Goal is a red star.

Usage:
    python visualize_env.py               # both individuals + groups (default)
    python visualize_env.py --groups-only
    python visualize_env.py --individuals-only
"""
import sys, argparse
sys.path.insert(0, '.')
sys.path.insert(0, 'crowd_nav')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' if TkAgg is not available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

parser = argparse.ArgumentParser()
parser.add_argument('--groups-only',      action='store_true')
parser.add_argument('--individuals-only', action='store_true')
parser.add_argument('--steps',            type=int, default=200)
parser.add_argument('--no-realistic',     action='store_true',
                    help='Disable realistic modeling phases (A-E)')
args = parser.parse_args()

# ── Config ──────────────────────────────────────────────────────────────────
from crowd_nav.configs.config import Config
config = Config()
config.sim.predict_method = 'none'
config.env.use_wrapper = False

config.sim.has_individuals = not args.groups_only
config.sim.has_groups      = not args.individuals_only

if not args.no_realistic:
    config.realistic.enabled              = True
    config.realistic.use_speed_variation  = True
    config.realistic.use_group_speed_factor = True
    config.realistic.use_f_formations     = True
    config.realistic.use_convex_hull      = True

# ── Environment ──────────────────────────────────────────────────────────────
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.robot import Robot

env = CrowdSimVarNum()
env.configure(config)

robot = Robot(config, 'robot')
env.set_robot(robot)
env.thisSeed = 0
env.nenv = 1
env.phase = 'test'

# ── ORCA policy ──────────────────────────────────────────────────────────────
from crowd_nav.policy.orca import ORCA
from crowd_sim.envs.utils.state import JointState
policy = ORCA(config)
policy.time_step = config.env.time_step
robot.policy = policy
robot.kinematics = config.action_space.kinematics

# ── Color palette for up to 6 groups ─────────────────────────────────────────
GROUP_COLORS = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4']

# ── Plot setup ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.set_title('Crowd: static_f=cross-hatch  dynamic_lf=solid  dynamic_free=striped  individual=open', fontsize=9)
env.render_axis = ax

ob = env.reset()
print(f"Reset OK | humans: {env.human_num} | groups: {len([g for g in env.grp if len(g.members)>0])}")
for gid, g in enumerate(env.grp):
    if g.members:
        ids = [m.id for m in g.members]
        print(f"  Group {gid}: type={g.group_type}, {len(g.members)} members, ids {ids}")

artists = []

def draw_frame():
    for a in artists:
        try: a.remove()
        except: pass
    artists.clear()

    # Goal
    g_star, = ax.plot(robot.gx, robot.gy, marker='*', color='red',
                      markersize=15, linestyle='None')
    artists.append(g_star)

    # Robot
    rc = plt.Circle((robot.px, robot.py), robot.radius, color='yellow', zorder=5)
    ax.add_patch(rc)
    artists.append(rc)

    # Humans
    for h in env.humans:
        gid = getattr(h, 'group_id', None)
        if gid is not None:
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            c = plt.Circle((h.px, h.py), h.radius, color=color, alpha=0.6)
        else:
            c = plt.Circle((h.px, h.py), h.radius, fill=False, color='black')
        ax.add_patch(c)
        artists.append(c)

    # Convex hull outlines
    if config.realistic.enabled and config.realistic.use_convex_hull and env.group_hulls:
        for gid, hull in env.group_hulls.items():
            color = GROUP_COLORS[gid % len(GROUP_COLORS)]
            if hull._kind == 'polygon':
                verts = hull._verts
                poly = plt.Polygon(verts, fill=False, edgecolor=color,
                                   linestyle='--', linewidth=1.5)
                ax.add_patch(poly)
                artists.append(poly)
            else:
                hc = plt.Circle(hull.centroid, hull.bounding_radius,
                                fill=False, edgecolor=color, linestyle='--', linewidth=1.5)
                ax.add_patch(hc)
                artists.append(hc)

    fig.canvas.draw()
    plt.pause(0.05)

draw_frame()
plt.pause(0.5)

from crowd_sim.envs.utils.action import ActionXY

for step in range(args.steps):
    state = JointState(robot.get_full_state(),
                       [env.humans[i].get_observable_state() for i in range(env.human_num)])
    action = robot.policy.predict(state)

    ob, reward, done, info = env.step(action)
    draw_frame()

    if done:
        print(f"Episode ended at step {step+1} | {info['info']}")
        plt.pause(1.0)
        ob = env.reset()
        draw_frame()

plt.show()
