import numpy as np
import gymnasium as gym

import loco_mujoco
from loco_mujoco import LocoEnv

np.random.seed(0)

# env = gym.make(
#     "LocoMujoco", 
#     env_name="UnitreeA1.simple.perfect", 
#     render_mode="rgb_array", 
#     random_start=False, 
#     init_step_no=0
# )
perfect_env = LocoEnv.make(
    env_name="UnitreeA1.simple.perfect",
    random_start=False, init_step_no=0, # should be
)
test_env = LocoEnv.make(
    env_name="UnitreeA1.simple",
    random_start=False, init_step_no=0, # should be
    use_foot_forces=True,
)

expert_dataset = perfect_env.create_dataset()
expert_actions = expert_dataset["actions"]

test_env.reset()
terminated = False
i = 0
j = 0

while j < 1000:
    if i == 1000 or terminated:
        test_env.reset()
        i = 0
    action = expert_actions[i, :]
    nstate, reward, terminated, info = test_env.step(action)

    feet_indices = [39, 42, 45, 48]  # Indices for foot contact z-axis forces
    contact = nstate[feet_indices] < -0.005
    # contact = state[37:]
    print(f"contact: {contact}")

    test_env.render()


# nstate: [
# -0.19148665 -0.06715507 -0.05010611  0.02456746  0.19387722  
#  1.0193188 -2.30184897 -0.04651593  0.82791845 -1.91700003 
#  -0.00693765  0.8360486 -1.89015754  0.21086673  1.06365968 
# -2.28611392  0.10284664  0.12294539  0.00754852 -0.2373313   
#  0.013838    0.0730157  -1.49492816 -3.30980746 -3.36642124 
#  -0.33319809  0.51768361 -0.33530835 -0.1995185   0.32513335
#   0.32961651 -0.13991655 -2.03339286 -3.57649319  1.          
#   0.          0.22803894]
# 37개