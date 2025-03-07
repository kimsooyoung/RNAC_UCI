import os
import argparse

import numpy as np
import loco_mujoco  # needed to register the environments
import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from stable_baselines3 import SAC, TD3, A2C, PPO

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


def train(env, sb3_algo):
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case 'PPO':
            model = PPO('MlpPolicy', env, verbose=1, device='cuda', tensorboard_log=log_dir)
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 75_000
    # iters = 0
    # while True:
    #     iters += 1

    #     model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    #     model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False,
        progress_bar=True
    )
    model.save(f"{model_dir}/{sb3_algo}")

def test(env, sb3_algo, path_to_model):

    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        env.render()

        if done:
            extra_steps -= 1
            if extra_steps < 0:
                break

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

if __name__ == '__main__':

    # create an environment with a Muscle Humanoid running with motion capture data (real dataset type)
    
    # gymenv = gym.make(
    #     "LocoMujoco", 
    #     env_name="UnitreeA1.simple",
    #     # render_mode=None,
    #     # render_mode="human",
    #     render_mode="rgb_array",
    # )

    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e. PandaReach-v2')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. A2C, DDPG, DQN, PPO, SAC, TD3')
    parser.add_argument('normalize', choices=['None', 'VecNormalize'], help='Normalization method: "VecNormalize" or "None"')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    sb3_algo = "PPO"

    # try:
    #     sb3_class = get_model_class(args.sb3_algo)
    # except ValueError as e:
    #     print(e)
    #     exit(1)

    if args.train:
        gymenv = gym.make(
            "LocoMujoco", 
            env_name=args.gymenv,
            render_mode="rgb_array",
            # Custom reward applied
            reward_type="custom",
            reward_params=dict(reward_callback=my_reward_function)
        )
        train(gymenv, sb3_algo)

    if args.test:
        if os.path.isfile(args.test):
            eval_env = gym.make(
                "LocoMujoco", 
                env_name=args.gymenv,
                render_mode="human",
                # Custom reward applied
                reward_type="custom",
                reward_params=dict(reward_callback=my_reward_function)
            )
            test(eval_env, sb3_algo, args.test)
        else:
            print(f'{args.test} not found.')

    gymenv.close()

# python sb3_quad.py UnitreeA1.simple PPO None -t
# python sb3_quad.py UnitreeA1.simple PPO None -s ./models/PPO.zip


# tensorboard --logdir logs
# ./models/PPO_7050000.zip