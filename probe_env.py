"""One-shot env probe — prints 5 step observations. Run with crowd_sim python."""
import numpy as np
import sys
sys.path.insert(0, '/home/lenovo/crowd_nav_with_group')

from crowd_nav.configs.config import Config
from crowd_sim.envs.crowd_sim_var_num import CrowdSimVarNum
from crowd_sim.envs.utils.robot import Robot

config = Config()
env = CrowdSimVarNum()
env.configure(config)
env.thisSeed = 0
env.nenv = 1
env.phase = 'test'
env.case_counter = {'train': 0, 'val': 0, 'test': 0}
env.case_capacity = {'train': 100000, 'val': 1000, 'test': 1000}
env.case_size = {'train': 100000, 'val': 1000, 'test': 1000}

robot = Robot(config, 'robot')
robot.set(0, 0, 0, -1.5, 0, 0, np.pi / 2)
from crowd_nav.policy.orca import ORCA
policy = ORCA(config)
policy.time_step = config.env.time_step
robot.policy = policy
robot.kinematics = config.action_space.kinematics
env.set_robot(robot)

ob = env.reset(phase='test')
print("=== RESET OBS ===")
for k, v in ob.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: shape={v.shape} dtype={v.dtype}  val={v}")
    else:
        print(f"  {k}: {v}")

print(f"\naction_space: {env.action_space}")
print(f"observation_space: {env.observation_space}")
print(f"robot: px={env.robot.px:.2f} py={env.robot.py:.2f} gx={env.robot.gx:.2f} gy={env.robot.gy:.2f}")
print(f"human_num={env.human_num}, max_human_num={env.max_human_num}, num_groups={env.num_groups}")

print("\n=== 5 STEPS ===")
for t in range(5):
    action = env.action_space.sample()
    ob, reward, done, info = env.step(action)
    print(f"step {t+1}: action={action.round(4)} reward={reward:.4f} done={done} info={info}")
    for k, v in ob.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    if done:
        break
