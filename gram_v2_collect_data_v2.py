"""
GRAM-v2 Phase 1 — data collection v2 (coherent-group labels only).

Identical to gram_v2_collect_data.py except build_pair_labels() skips
dynamic_free groups. Members of dynamic_free groups navigate independently
via ORCA — over any temporal window they are perceptually indistinguishable
from unrelated pedestrians walking nearby, making their pair labels noise.

Only static_f and dynamic_lf groups are labeled as positive pairs:
  static_f   — stationary F-formation, members cluster together (zero velocity)
  dynamic_lf — leader-follower, members co-move coherently

This removes the irreducible ~0.25 F1 noise floor from Phase 1 training,
expected to push Phase 1 F1 from ~0.75 to 0.85+ without any model changes.

Usage:
  python gram_v2_collect_data_v2.py               # default: 3000/300/300 episodes
  python gram_v2_collect_data_v2.py --train 500   # quick smoke-test

Output: gram_v2_data_v2/{train, val, test}.npz  (same shape as v1)
  feats  : float32 (T, 20, 21)
  masks  : bool    (T, 20)
  labels : float32 (T, 20, 20)  1.0 for static_f/dynamic_lf pairs only
"""

import os, sys, argparse
import numpy as np
from tqdm import tqdm
from collections import deque

sys.path.insert(0, os.path.dirname(__file__))

from crowd_nav.configs.config import Config
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.policy.orca import ORCA
from crowd_nav.gram_v2.models import FEAT_DIM, T_WINDOW

MAX_HUMANS   = 20
SAMPLE_EVERY = 3

# Group types treated as coherent (detectable from observations)
COHERENT_GROUP_TYPES = {'static_f', 'dynamic_lf'}


# ── Feature extraction ────────────────────────────────────────────────────────

def build_features(ob: dict, visible: np.ndarray) -> np.ndarray:
    spatial  = ob['spatial_edges']
    velocity = ob['velocity_edges']
    feats = np.zeros((MAX_HUMANS, FEAT_DIM), dtype=np.float32)
    for i in range(MAX_HUMANS):
        if visible[i]:
            p = spatial[i]
            v = velocity[i]
            v_norm  = float(np.linalg.norm(v))
            sin_t   = v[1] / (v_norm + 1e-6)
            cos_t   = v[0] / (v_norm + 1e-6)
            feats[i] = [p[0], p[1], v[0], v[1], v_norm, sin_t, cos_t]
    return feats


def build_pair_labels(env) -> np.ndarray:
    """
    Build (MAX_HUMANS, MAX_HUMANS) symmetric GT groupness matrix.
    Only static_f and dynamic_lf group pairs are labeled 1.0.
    dynamic_free pairs are labeled 0.0 — members navigate independently
    and are perceptually indistinguishable from unrelated pedestrians.
    """
    # Map group_id → group_type from env.groups
    gid_to_type = {}
    if hasattr(env, 'groups') and env.groups:
        for group in env.groups:
            gid_to_type[group.id] = group.group_type

    labels = np.zeros((MAX_HUMANS, MAX_HUMANS), dtype=np.float32)
    n = min(env.human_num, MAX_HUMANS)
    for i in range(n):
        gi = env.humans[i].group_id
        if gi is None:
            continue
        if gid_to_type.get(gi) not in COHERENT_GROUP_TYPES:
            continue   # skip dynamic_free — not coherently detectable
        for j in range(n):
            if i == j:
                continue
            gj = env.humans[j].group_id
            if gj is not None and gi == gj:
                labels[i, j] = 1.0
    return labels


# ── Environment setup ─────────────────────────────────────────────────────────

def make_env(seed: int) -> CrowdSimVarNum:
    config = Config()
    env = CrowdSimVarNum()
    env.configure(config)
    env.thisSeed = seed
    env.nenv = 1
    env.phase = 'train'
    env.case_counter  = {'train': 0, 'val': 0, 'test': 0}
    env.case_capacity = {'train': 100000, 'val': 1000, 'test': 1000}
    env.case_size     = {'train': 100000, 'val': 1000, 'test': 1000}

    robot = Robot(config, 'robot')
    robot.set(0, 0, 0, -1.5, 0, 0, np.pi / 2)
    policy = ORCA(config)
    policy.time_step = config.env.time_step
    robot.policy = policy
    robot.kinematics = config.action_space.kinematics
    env.set_robot(robot)
    return env


# ── Collection loop ───────────────────────────────────────────────────────────

def collect(env, n_episodes: int, phase: str, desc: str):
    env.phase = phase
    all_feats, all_masks, all_labels = [], [], []

    for _ in tqdm(range(n_episodes), desc=desc):
        ob = env.reset(phase=phase)
        done = False
        step = 0
        frame_buf = deque(maxlen=T_WINDOW)

        while not done:
            ob, _, done, _ = env.step(np.zeros(2))
            step += 1

            visible = ob['visible_masks'].astype(bool)
            feats   = build_features(ob, visible)
            frame_buf.append(feats)

            if step % SAMPLE_EVERY != 0:
                continue
            if len(frame_buf) < T_WINDOW:
                continue

            labels = build_pair_labels(env)

            if labels.sum() == 0 and visible.sum() < 3:
                continue

            stacked = np.concatenate(list(frame_buf), axis=-1)
            all_feats.append(stacked)
            all_masks.append(visible)
            all_labels.append(labels)

    return (np.array(all_feats,  dtype=np.float32),
            np.array(all_masks,  dtype=bool),
            np.array(all_labels, dtype=np.float32))


# ── Save + report ─────────────────────────────────────────────────────────────

def save_and_report(path: str, feats, masks, labels):
    np.savez_compressed(path, feats=feats, masks=masks, labels=labels)
    T, _, input_dim = feats.shape
    print(f"  Feature shape: (T={T:,}, N=20, input_dim={input_dim})  "
          f"[{T_WINDOW} frames × {FEAT_DIM} dims]")
    pos = labels.sum()
    vis_pairs = 0.0
    for i in range(MAX_HUMANS):
        for j in range(i + 1, MAX_HUMANS):
            vis_pairs += (masks[:, i] & masks[:, j]).sum()
    rate = pos / (2.0 * vis_pairs + 1e-6)
    print(f"  Saved {T:,} samples → {path}")
    print(f"  Visible pair positive rate: {rate:.3f}  "
          f"(recommended pos_weight ≈ {1.0/rate - 1:.1f})")
    print(f"  (dynamic_free pairs excluded from positive labels)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--val',   type=int, default=300)
    parser.add_argument('--test',  type=int, default=300)
    parser.add_argument('--out',   default='gram_v2_data_v2')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Data collection v2 — dynamic_free groups excluded from positive labels.")
    print("Data collection runs on CPU only (pure simulation, no neural network).")

    splits = [
        ('train', args.train, 42),
        ('val',   args.val,   123),
        ('test',  args.test,  999),
    ]

    for split_name, n_ep, seed in splits:
        print(f"\n── {split_name.upper()} ({n_ep} episodes, seed={seed}) ──")
        env = make_env(seed)
        feats, masks, labels = collect(env, n_ep, 'train',
                                       desc=f'collecting {split_name}')
        save_and_report(os.path.join(args.out, f'{split_name}.npz'),
                        feats, masks, labels)

    print("\nDone. Train with:")
    print("  python gram_v2_train_phase1.py --variant B --data gram_v2_data_v2")
    print("  python gram_v2_train_phase2.py --data gram_v2_data_v2 --phase1 trained_models/gram_v2/phase1/B/best.pt")


if __name__ == '__main__':
    main()
