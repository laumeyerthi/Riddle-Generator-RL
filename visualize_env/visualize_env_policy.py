import gymnasium as gym
from gymnasium_env.envs.lab_env import LabEnv
from llm_interface.ppo_interface import PPOInterface
import time
import random

def main():
    # Initialize environment with render_mode="human"
    env = LabEnv(render_mode="human", number_of_rooms=4)
    obs, info = env.reset()
    interface = PPOInterface()
    print("Environment initialized with Pygame rendering.")
    print("Running selected actions...")

    for i in range(10):
        action = interface.get_action(obs)
        match action:
            case 0:
                print("Right")
            case 1:
                print("Up")
            case 2:
                print("Left")
            case 3:
                print("Down")
            case 4:
                print("Backtrack")
            case _:
                print("Button " + str(action - 5))
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
