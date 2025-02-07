import numpy as np
import gymnasium as gym
from loco_mujoco import LocoEnv

env = gym.make(
    "LocoMujoco",
    env_name="UnitreeA1.simple.perfect",
    render_mode="human",
    random_start=False,
    init_step_no=0,
)

expert_dataset = env.create_dataset()
expert_actions = expert_dataset["actions"]

env.reset()
env.render()

i = 0
while i < 1000:
    action = expert_actions[i, :]
    nstate, reward, terminated, truncated, info = env.step(action)

    env.render()
    i += 1

