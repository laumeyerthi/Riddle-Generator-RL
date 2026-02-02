import gymnasium as gym
from gymnasium_env.envs.lab_env import LabEnv
import time
import random

def main():
    # Initialize environment with render_mode="human"
    env = LabEnv(render_mode="human", number_of_rooms=4)
    obs, info = env.reset()

    print("Environment initialized with Pygame rendering.")
    print("Running random actions...")

    for i in range(50):
        # Pick a random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render is called inside step() when render_mode="human"
        
        # Slow down to see the rendering
        time.sleep(0.1)
        
        if terminated or truncated:
            print("Episode finished")
            obs, info = env.reset()

    print("Closing environment...")
    env.close()

if __name__ == "__main__":
    main()
