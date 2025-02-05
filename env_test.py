import gymnasium as gym

# Create the Hopper-v4 environment
env = gym.make("Hopper-v4", render_mode="human")
# env = gym.make("HopperPerturbed-v4", render_mode="human")

# Reset the environment
obs, info = env.reset()

for _ in range(1000):  # Run for 1000 steps
    action = env.action_space.sample()  # Take random actions
    print(f"action: {action} / type: {type(action)}")

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()  # Reset environment if episode ends

env.close()