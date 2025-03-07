import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym


# state: [-2.02338202e-01 -1.66293944e-01 -4.30141854e-02 -1.06729561e-02
#   1.78151211e-01  1.06543480e+00 -1.89053155e+00  2.46991746e-01
#   1.35588951e+00 -2.16504495e+00 -1.80170827e-01  7.31700626e-01
#  -1.81934680e+00  8.85233814e-01  7.49501567e-01 -2.52614014e+00
#   1.54939126e-01  8.60650392e-02 -6.68341116e-01  8.11738021e+00
#   7.28012412e-01  5.41277312e-01 -1.43380141e+01 -2.36766944e+00
#   8.36026893e+00 -5.09921624e-01 -4.88007541e+00 -3.71089738e+00
#  -2.10297832e+01 -9.46518217e+00  1.03049143e+00 -1.42685263e+01
#  -1.15428943e+01 -7.33749440e+00  1.00000000e+00  0.00000000e+00
#   2.36557250e-01]
# action: [
#  -0.13810948 -0.76405394 -1.70220186 -0.30373292 -0.80036508 
# -1.02428249
#  -0.41239959  1.75672855 -0.53530012  0.67946995 -0.16635693 
# -0.20215387]

def my_reward_function(state, action, next_state):
    # Extract necessary state variables using correct indices
    vel_x, vel_y = next_state[16], next_state[17]  # trunk_tx and trunk_ty (XY velocity)
    cmd_vel = state[36]  # Desired velocity
    cmd_yaw_sin, cmd_yaw_cos = state[34], state[35]  # Desired velocity angle as sine-cosine
    vel_yaw = next_state[21]  # trunk_rotation velocity (yaw velocity)
    vel_z = next_state[18]  # trunk_tz velocity (Z-axis movement)
    
    pos_z, h_target = next_state[2], state[2]  # trunk_tilt as base height (approximate)
    
    prev_action = state[-12:]  # Last 12 elements as previous action
    theta = state[4:16]  # Joint angles
    theta_default = np.zeros_like(theta)  # Assuming default is zero

    # Convert desired yaw to angle
    cmd_yaw = np.arctan2(cmd_yaw_sin, cmd_yaw_cos)

    tracking_sigma = 0.25

    # Tracking Linear Velocity (XY)
    track_lin_reward = np.exp(-((cmd_vel - np.sqrt(vel_x**2 + vel_y**2))**2) / tracking_sigma)

    # Tracking Angular Velocity (Yaw)
    track_ang_reward = np.exp(-((cmd_yaw - vel_yaw) ** 2) / tracking_sigma)

    # Penalizing Z-Axis Linear Velocity
    penalize_z = vel_z ** 2

    # Penalizing Action Rate (Smooth actions)
    action_rate_penalty = np.sum((action - prev_action) ** 2)

    # # Penalizing Deviation from Default Pose
    # pose_deviation_penalty = -np.sum((theta - theta_default) ** 2)

    # Penalizing Base Height Deviation
    height_deviation_penalty = (pos_z - h_target) ** 2

    # Total reward
    total_reward = (
        track_lin_reward +
        track_ang_reward +
        penalize_z +
        action_rate_penalty +
        # pose_deviation_penalty +
        height_deviation_penalty
    )

    return total_reward

env = gym.make(
    "LocoMujoco", 
    env_name="UnitreeA1.simple", 
    reward_type="custom",
    reward_params=dict(reward_callback=my_reward_function)
)

action_dim = env.action_space.shape[0]

env.reset()
env.render()
terminated = False
i = 0

while True:
    if i == 1000 or terminated:
        env.reset()
        i = 0
    action = np.random.randn(action_dim)
    nstate, reward, terminated, truncated, info = env.step(action)

    print(f"reward: {reward}")
    # reward: 1204.3052036815045
    # reward: 1350.8591943195315
    # reward: 1611.0939164642703
    # reward: 18.09480879774595
    # reward: 588.9647728614419
    # reward: 1122.9844677935757

    # HERE is your favorite RL algorithm

    env.render()
    i += 1
