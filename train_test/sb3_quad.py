import os
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

    TIMESTEPS = 25000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS*iters}")

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

if __name__ == '__main__':

    # create an environment with a Muscle Humanoid running with motion capture data (real dataset type)
    gymenv = gym.make(
        "LocoMujoco", 
        env_name="UnitreeA1.simple",
        # render_mode=None,
        # render_mode="human",
        render_mode="rgb_array",
    )
    gymenv = RecordVideo(
        gymenv, 
        video_folder="videos/", 
        episode_trigger=lambda e: True
    )


    sb3_algo = "PPO"

    # train(gymenv, sb3_algo)

    path_to_model = "/home/kimsooyoung/Documents/Research/RNAC_UCI/train_test/models/PPO_4475000.zip"
    test(gymenv, sb3_algo, path_to_model=path_to_model)

    gymenv.close()
    print("Video saved in 'videos/' directory")


# tensorboard --logdir logs
# ./models/PPO_7050000.zip