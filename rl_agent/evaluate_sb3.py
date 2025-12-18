import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

def evaluate():
    env = LabEnv(number_of_rooms=4)
    env = FlattenObservation(env)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', "dqn_lab_env")
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip. Please run rl_agent/dqn_sb3_agent.py first.")
        return

    print(f"Loading model from {model_path}...")
    model = DQN.load(model_path, env=env)
    
    print("Evaluating policy (10 episodes)...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    evaluate()
