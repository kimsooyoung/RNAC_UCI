import gymnasium as gym
from go1_mujoco_env import Go1MujocoEnv

# create an environment with a Muscle Humanoid running with motion capture data (real dataset type)
env = Go1MujocoEnv(
    ctrl_type="torque",
    render_mode="human",
)

# Reset the environment
obs, info = env.reset()

for _ in range(1000):  # Run for 1000 steps
    action = env.action_space.sample()  # Take random actions
    # print(f"action: {action} / type: {type(action)}")
    obs, reward, terminated, truncated, info = env.step(action)

    env.render()

    if terminated or truncated:
        obs, info = env.reset()  # Reset environment if episode ends

env.close()