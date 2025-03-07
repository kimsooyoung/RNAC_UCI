import numpy as np
from loco_mujoco import LocoEnv
import gymnasium as gym

# global prev_action
prev_action = None

def my_reward_function(state, action, next_state):
    global prev_action

    # Extract necessary state variables using correct indices
    vel_x, vel_y = state[16], state[17]  # trunk_tx and trunk_ty (XY velocity)
    cos_sine = state[[34, 35]]
    
    cmd_vel = state[36]  # Desired velocity
    cmd_yaw_sin, cmd_yaw_cos = state[34], state[35]  # Desired velocity angle as sine-cosine
    vel_yaw = state[21]  # trunk_rotation velocity (yaw velocity)
    
    # Convert desired yaw to angle
    cmd_yaw = np.arctan2(cmd_yaw_sin, cmd_yaw_cos)
    tracking_sigma = 0.25
    
    # Tracking Linear Velocity (XY)
    des_vel = state[36] * cos_sine
    curr_velocity_xy = np.array([vel_x, vel_y])
    lin_vel_error = np.sum((des_vel - curr_velocity_xy)**2)
    track_lin_reward = np.exp(-lin_vel_error / tracking_sigma)

    # Tracking Angular Velocity (Yaw)
    # track_ang_reward = np.exp(-((cmd_yaw - vel_yaw) ** 2) / tracking_sigma)

    # Penalizing Z-Axis Linear Velocity
    vel_z = state[18]  # trunk_tz velocity (Z-axis movement)
    penalize_z = vel_z ** 2

    # Penalizing Action Rate (Smooth actions)
    if prev_action is None:
        action_rate_penalty = 0.0
    else:
        action_rate_penalty = np.sum((action - prev_action) ** 2)
        prev_action = action

    # Penalizing Deviation from Default Pose
    theta = state[4:16]  # Joint angles
    theta_default = np.zeros_like(theta)  # Assuming default is zero
    pose_deviation_penalty = np.sum((theta - theta_default) ** 2)

    # Penalizing Base Height Deviation
    h_target = -0.17
    pos_z = state[0]  # trunk_tilt as base height (approximate)
    height_deviation_penalty = (pos_z - h_target) ** 2

    # Total reward
    total_reward = (
        1.0 * track_lin_reward +
        # track_ang_reward +
        -1.0 * penalize_z +
        -0.005 * action_rate_penalty +
        -0.1 * pose_deviation_penalty +
        -50.0 * height_deviation_penalty
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
