# RNAC_UCI

## **Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation - UCI Version**

Original Paper from [Natural Actor-Critic for Robust Reinforcement Learning with Function Approximation (NeurIPS 2023)](https://arxiv.org/abs/2307.08875). 

> This version uses outdated gym / numpy / mujoco versions therefore I updated and tested on more new version.

## Prerequisites
```
numpy==1.23.5
gymnasium==0.29.1
mujoco==2.3.7
torch==2.6.0
```
## Usage

This project uses mujoco_env from gymnasium. And also requires a bit of modification. For instance, like the code below, by adding more params to the `reset` function, you can reset your environment with specific status.

```python
    # Customized
    def reset_model(self, state=None, x_pos=None) -> NDArray[np.float64]:
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError
```

Therefore, Copy below python codes and paste into your local gymnasium package location 

* find gymnasium's local location

```bash
# example
$ pip show gymnasium
...
Location: /home/kimsooyoung/anaconda3/envs/gym-mujoco/lib/python3.10/site-packages/gymnasium/envs/mujoco
```

* Copy & paste/overwrite below files under `gymnasium/envs/mujoco` (highly recommend to backup original files)
1. hopper_v4_perturbed.py
2. mujoco_env.py
3. hopper_v4.py
4. other files ongoing ... 

* Now, register your new environments by editting `gymnasium/envs/toy_text/__init__.py`

```python
# example
register(
    id="HopperPerturbed-v4",
    entry_point="gymnasium.envs.mujoco.hopper_v4_perturbed:HopperEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
```

* Test newly registered environment

```python
import gymnasium as gym

# Create customed HopperPerturbed-v4 environment
env = gym.make("HopperPerturbed-v4", render_mode="human")

# Reset the environment
obs, info = env.reset()

for _ in range(1000):  # Run for 1000 steps
    action = env.action_space.sample()  # Take random actions
    print(f"action: {action} / type: {type(action)}")

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()  # Reset environment if episode ends

env.close()
```

## Train Policy

To train an RNAC policy on `Hopper-v4` with DS uncertainty set, please run 

```bash
python train_rnac.py --env='Hopper-v4' --uncer_set='DS' --weight_reg=0.0

# Check learning status
tensorboard --logdir runs/RNAC/XXXX
```

## Evaluate Policy

To evaluate an RNAC policy on `HopperPerturbed-v4`, please run

```bash
python eval_rnac.py --env='Hopper-v4'
```

> Don't type `HopperPerturbed-v4` python code will automatically spawn perturbed env

```python
parser.add_argument("--env", type=str, default='Hopper-v4', help="HalfCheetah-v4/Hopper-v4/Walker2d-v4")
...
save_path = f"./perturbed_results/RNAC_{args.env}_{GAMMA}"
```

## Train Quadruped robot

We use [loco_mujoco](https://github.com/robfiras/loco-mujoco) as our environment. Checkout their documentations for details.

## Train Policy

To train an RNAC policy on `UnitreeA1.simple` with DS uncertainty set, please run 

```bash
python quad_train_rnac.py --env='UnitreeA1.simple' --uncer_set='IPM' --weight_reg=0.0

python quad_train_rnac.py --env='UnitreeA1.simple' --uncer_set='DS' --weight_reg=0.0

# Check learning status
tensorboard --logdir runs/RNAC/XXXX
```

## Evaluate Policy

To evaluate an RNAC policy on `UnitreeA1.simple`, please run

```bash
python quad_eval_rnac.py --env='UnitreeA1.simple'
```