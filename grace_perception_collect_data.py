"""
GRAM-v2 Phase 1 — data collection.

Runs ORCA-piloted rollouts and saves 3-frame temporal windows + ground-truth
pair-groupness labels to gram_v2_data/{train, val, test}.npz.

GT labels use env-internal human.group_id — they are ONLY used here for
supervision targets, never fed into the network at inference.

Each sample is a T_WINDOW=3 consecutive-frame stack (oldest → newest):
  feats[t] = [frame_{t-2}, frame_{t-1}, frame_t] concatenated → (20, 21)
This gives the model velocity trend information: group members consistently
co-move across frames, while strangers drift apart.

Usage:
  python grace_perception_collect_data.py               # default: 3000/300/300 episodes
  python grace_perception_collect_data.py --train 500   # quick smoke-test

Output files (per split):
  feats  : float32 (T, 20, 21)  3-frame stacked features (zeros for invisible)
  masks  : bool    (T, 20)       True = human visible at current (newest) frame
  labels : float32 (T, 20, 20)  1.0 if both humans in the same group, else 0.0
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
from crowd_nav.grace_perception.models import FEAT_DIM, T_WINDOW

MAX_HUMANS   = 20
SAMPLE_EVERY = 3   # collect 1 sample every N steps to reduce temporal correlation


# ── Feature extraction ────────────────────────────────────────────────────────

def build_features(ob: dict, visible: np.ndarray) -> np.ndarray:
    """
    Convert obs dict to (MAX_HUMANS, 7) feature matrix.
    Invisible humans get all-zero features.
    Feature layout: [p_rel_x, p_rel_y, v_rel_x, v_rel_y, v_norm, sin_t, cos_t]
    """
    spatial  = ob['spatial_edges']    # (20, 2) — 15.0 sentinel for invisible
    velocity = ob['velocity_edges']   # (20, 2) — 15.0 sentinel for invisible
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
    labels[i,j] = 1.0 iff humans i and j are in the same group (not None).
    Uses env.humans[i].group_id — TRAINING SUPERVISION ONLY.
    """
    labels = np.zeros((MAX_HUMANS, MAX_HUMANS), dtype=np.float32)
    n = min(env.human_num, MAX_HUMANS)
    for i in range(n):
        gi = env.humans[i].group_id
        if gi is None:
            continue
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
    """Run n_episodes and return (feats, masks, labels) numpy arrays.

    Each sample stacks T_WINDOW consecutive frames (oldest first) so the
    model can observe velocity trends across time.
    """
    env.phase = phase

    all_feats, all_masks, all_labels = [], [], []

    for _ in tqdm(range(n_episodes), desc=desc):
        ob = env.reset(phase=phase)
        done = False
        step = 0
        # Rolling buffer of the last T_WINDOW single-frame feature arrays
        frame_buf = deque(maxlen=T_WINDOW)

        while not done:
            ob, _, done, _ = env.step(np.zeros(2))   # ORCA acts internally
            step += 1

            visible = ob['visible_masks'].astype(bool)  # (20,)
            feats   = build_features(ob, visible)       # (20, 7)
            frame_buf.append(feats)

            if step % SAMPLE_EVERY != 0:
                continue
            if len(frame_buf) < T_WINDOW:
                continue   # not enough history yet

            labels = build_pair_labels(env)

            # Skip steps where no groups are visible at all (not informative)
            if labels.sum() == 0 and visible.sum() < 3:
                continue

            # Stack frames oldest→newest: (20, 7*T_WINDOW) = (20, 21)
            stacked = np.concatenate(list(frame_buf), axis=-1)

            all_feats.append(stacked)
            all_masks.append(visible)   # mask from current (newest) frame
            all_labels.append(labels)

    return (np.array(all_feats,  dtype=np.float32),   # (T, 20, 21)
            np.array(all_masks,  dtype=bool),          # (T, 20)
            np.array(all_labels, dtype=np.float32))    # (T, 20, 20)


# ── Save + report ─────────────────────────────────────────────────────────────

def save_and_report(path: str, feats, masks, labels):
    np.savez_compressed(path, feats=feats, masks=masks, labels=labels)
    T, _, input_dim = feats.shape
    print(f"  Feature shape: (T={T:,}, N=20, input_dim={input_dim})  "
          f"[{T_WINDOW} frames × {FEAT_DIM} dims]")
    # positive pair rate (upper triangle of each sample, masked to visible pairs)
    pos = labels.sum()
    vis_pairs = 0.0
    for i in range(MAX_HUMANS):
        for j in range(i + 1, MAX_HUMANS):
            vis_pairs += (masks[:, i] & masks[:, j]).sum()
    rate = pos / (2.0 * vis_pairs + 1e-6)   # both directions counted in labels
    print(f"  Saved {T:,} samples → {path}")
    print(f"  Visible pair positive rate: {rate:.3f}  "
          f"(recommended pos_weight ≈ {1.0/rate - 1:.1f})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=int, default=3000)
    parser.add_argument('--val',   type=int, default=300)
    parser.add_argument('--test',  type=int, default=300)
    parser.add_argument('--out',   default='gram_v2_data')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Data collection runs on CPU only (pure simulation, no neural network).")

    # Always collect in 'train' phase so calc_reward stays in circle mode
    # (phase='val'/'test' triggers future-trajectory reward that needs GST predictor).
    # Split separation is achieved via different seeds, not env phase.
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

    print("\nDone. Run grace_perception_train_phase1.py to train.")


if __name__ == '__main__':
    main()
