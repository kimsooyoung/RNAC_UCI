import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import loco_mujoco  # needed to register the environments
import gymnasium as gym

import argparse
import pickle
import math
import random
import copy
import mujoco_py
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from ppo_continuous import PPO_continuous
from torch.distributions import Uniform


def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        # TODO
        # s, _ = env.reset(state=None, x_pos=None)
        s, _ = env.reset()
        
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        
        done = False
        episode_reward = 0
        episode_number = 0
        
        while not done:
            
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a

            s_, r, done, _, _ = env.step(action)

            episode_number += 1
            if episode_number > args.max_episode_steps:
                done = True

            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            
            episode_reward += r
            s = s_

        evaluate_reward += episode_reward

    return evaluate_reward / times


def save_agent(agent, save_path, state_norm, reward_scaling):
    agent.actor.save(f'{save_path}_actor')
    agent.critic.save(f'{save_path}_critic')
    with open(f'{save_path}_state_norm', 'wb') as file1:
        pickle.dump(state_norm, file1)
    with open(f'{save_path}_reward_scaling', 'wb') as file2:
        pickle.dump(reward_scaling, file2)


# global prev_action
prev_action = None

def my_reward_function(state, action, next_state):
    global prev_action

    # Extract necessary state variables using correct indices
    vel_x, vel_y = state[16], state[17]  # trunk_tx and trunk_ty (XY velocity)
    vel_yaw = state[21]  # trunk_rotation velocity (yaw velocity)
    cos_sine = state[[34, 35]]
    
    cmd_vel = state[36]  # Desired velocity
    cmd_yaw_sin, cmd_yaw_cos = state[34], state[35]  # Desired velocity angle as sine-cosine
    
    # Convert desired yaw to angle
    cmd_yaw = np.arctan2(cmd_yaw_sin, cmd_yaw_cos)
    tracking_sigma = 0.25
    
    # Tracking Linear Velocity (XY)
    des_vel = state[36]
    curr_velocity_xy = np.sqrt([vel_x**2 + vel_y**2])[0]
    lin_vel_error = (des_vel - curr_velocity_xy)**2
    track_lin_reward = np.exp(-lin_vel_error / tracking_sigma)

    # Tracking Angular Velocity (Yaw)
    track_ang_reward = np.exp(-((vel_yaw) ** 2) / tracking_sigma)

    # Penalizing Z-Axis Linear Velocity
    vel_z = state[18]  # trunk_tz velocity (Z-axis movement)
    penalize_z = vel_z ** 2

    # Penalizing Action Rate (Smooth actions)
    if prev_action is None:
        action_rate_penalty = 0.0
    else:
        action_rate_penalty = np.sum((action - prev_action) ** 2)
    prev_action = action.copy()

    # Penalizing Deviation from Default Pose
    theta = state[4:16]  # Joint angles
    theta_default = np.zeros_like(theta)  # Assuming default is zero
    pose_deviation_penalty = np.sum(abs(theta - theta_default))

    # Penalizing Base Height Deviation
    h_target = -0.17
    pos_z = state[0]  # trunk_tilt as base height (approximate)
    height_deviation_penalty = (pos_z - h_target) ** 2

    # TODO: feet_air_time
    feet_indices = [39, 42, 45, 48]  # Indices for foot contact z-axis forces
    contact = state[feet_indices] < -0.01
    # contact = state[37:]
    # print(f"contact: {contact}")

    # Penalize high contact forces
    max_contact_force = 0.1
    rew_contact_forces = np.sum(
        (np.linalg.norm(state[37:]) - max_contact_force
    ).clip(min=0.))

    # print(f"""
    #     track_lin_reward: {track_lin_reward}
    #     track_ang_reward: {track_ang_reward}
    #     penalize_z: {penalize_z}
    #     action_rate_penalty: {action_rate_penalty}
    #     pose_deviation_penalty: {pose_deviation_penalty}
    #     height_deviation_penalty: {height_deviation_penalty}
    #     rew_contact_forces: {rew_contact_forces}
    # """)

    # Total reward
    total_reward = (
        1.0 * track_lin_reward +
        0.2 * track_ang_reward +
        -1.0 * penalize_z +
        -0.005 * action_rate_penalty +
        -0.01 * pose_deviation_penalty +
        -50.0 * height_deviation_penalty +
        -0.0 * rew_contact_forces
    )

    return total_reward


def main(args, number):
    seed, GAMMA = args.seed, args.GAMMA
    
    env = gym.make(
        "LocoMujoco", 
        env_name=args.env,
        reward_type="custom",
        reward_params=dict(reward_callback=my_reward_function),
        use_foot_forces=True,
    )
    env_evaluate = gym.make(
        "LocoMujoco", 
        env_name=args.env,
        reward_type="custom",
        reward_params=dict(reward_callback=my_reward_function),
        use_foot_forces=True,
    )
    env_reset = gym.make(
        "LocoMujoco", 
        env_name=args.env,
        reward_type="custom",
        reward_params=dict(reward_callback=my_reward_function),
        use_foot_forces=True,
    )

    env.reset(seed=args.seed)
    env_evaluate.reset(seed=args.seed)
    env_reset.reset(seed=args.seed)
    
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args.state_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    args.max_action = float(env.action_space.high[0])
    # temporarly 1000
    args.max_episode_steps = 1000 
    # env._max_episode_steps  # Maximum number of steps per episode
    
    print("env={}".format(args.env))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training
    max_value = -np.inf
    save_path = f"./models/RNAC_{args.env}_{GAMMA}"

    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)

    # Build a tensorboard
    writer = SummaryWriter(log_dir='runs/RNAC/env_{}_{}_number_{}_seed_{}_GAMMA_{}'.format(args.env, args.policy_dist, number, seed, GAMMA))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    # print(f"args.max_train_steps: {args.max_train_steps}")

    while total_steps < args.max_train_steps:

        # Gymnasium returns a tuple (obs, info)
        # TODO: adding state / x_pos
        # s, _ = env.reset(state=None, x_pos=None)
        s, _ = env.reset()
    
        # TODO: check and get x_pos
        s_org, x_pos = copy.deepcopy(s), np.array([env.unwrapped._data.qpos[0]])

        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset()
        
        episode_steps = 0
        done = False

        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)

            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a

            # Example action format
            # action: [ 0.2136665 -0.6555134  0.5038482]
            # print(f"type: {type(action)}")
            # print(f"action: {action}")

            if args.uncer_set == "DS":
                # Multi-run
                v_min, index = torch.tensor(float('inf')), 0
                noise_list, nexts_list, r_list = [], [], []

                for i in range(args.next_steps):
                    # TODO: adding state / x_pos
                    # obs, _ = env_reset.reset(state=s_org, x_pos=x_pos) 
                    obs, _ = env_reset.reset() 

                    s_, r, done, truncated, info = env_reset.step(action)
                    r_list.append(r)
                    print(f"info: {info}")
                    noise_list.append(info['noise'])

                    if args.use_state_norm:
                        s_ = state_norm(s_, update=False)

                    nexts_list.append(s_)

                    with torch.no_grad():   
                        if agent.critic(torch.tensor(s_, dtype=torch.float)) < v_min:
                            v_min = agent.critic(torch.tensor(s_, dtype=torch.float))
                            index = i
            
                # pick next state for robust critic update
                ridx = random.randint(0, args.next_steps)
                
                if ridx == args.next_steps:
                    ridx = index
                
                s_, r, done, truncated, info = env.step(np.concatenate((action, noise_list[ridx])))
            else:
                s_, r, done, truncated, info = env.step(action)

            # x_pos = np.array([info['x_position']])
            if args.use_state_norm:
                #nexts = state_norm(nexts, update=False)
                s_ = state_norm(s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # Take the 'action'，but store the original 'a'（especially for Beta）
            replay_buffer.store(s, a, a_logprob, r, s_, dw, done)
            s = copy.deepcopy(s_)
            s_org = copy.deepcopy(state_norm.denormal(s_, update=False))
            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                # print("total_steps:{} \t evaluate_freq:{} \t".format(total_steps, args.evaluate_freq))
                print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))

                # Record on Tensorboard 
                writer.add_scalar('step_rewards_{}'.format(args.env), evaluate_rewards[-1], global_step=total_steps)
                
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data_train/RNAC_{}_env_{}_number_{}_seed_{}_GAMMA_{}.npy'.format(args.policy_dist, args.env, number, seed, GAMMA), np.array(evaluate_rewards))

                # save actor, critic for evaluation in perturbed environment
                if evaluate_reward > max_value:
                    save_agent(agent, save_path, state_norm, reward_scaling)
                    max_value = evaluate_reward
                    print(f"max value found from total_steps : {total_steps}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for RNAC")
    parser.add_argument("--env", type=str, default='UnitreeA1.simple', help="HalfCheetah-v4/Hopper-v3/Walker2d-v3")
    parser.add_argument("--uncer_set", type=str, default='IPM', help="DS/IPM")
    parser.add_argument("--next_steps", type=int, default=2, help="Number of next states")
    parser.add_argument("--random_steps", type=int, default=int(25e3), help="Uniformlly sample action within random steps")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=5e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor 0.99")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter 0.95")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=True, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    parser.add_argument("--adaptive_alpha", type=float, default=False, help="Trick 11: adaptive entropy regularization")
    # parser.add_argument("--weight_reg", type=float, default=0, help="Regularization for weight of critic")
    parser.add_argument("--weight_reg", type=float, default=1e-5, help="Regularization for weight of critic")
    parser.add_argument("--seed", type=int, default=2, help="seed")
    parser.add_argument("--GAMMA", type=str, default='0', help="file name")

    args = parser.parse_args()
    
    # make folders to dump results
    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./data_train"):
        os.makedirs("./data_train")

    main(args, number=1)
