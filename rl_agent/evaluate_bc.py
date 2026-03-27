import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import sys
import os
import torch

original_torch_load = torch.load
def safe_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_torch_load(*args, **kwargs)
torch.load = safe_torch_load

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

def evaluate():
    number_of_rooms = 4 
    env = LabEnv(number_of_rooms=number_of_rooms)
    env = TimeLimit(env, max_episode_steps=100)
    env = FlattenObservation(env)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', "bc_lab_env")
    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run rl_agent/bc_agent.py first.")
        return

    print(f"Loading BC agent policy from {model_path}...")
    
    
    try:
        policy = ActorCriticPolicy.load(model_path)
    except Exception as e:
        print(f"Could not load using ActorCriticPolicy: {e}")
        return
        
    class BCPolicyModel:
        def __init__(self, policy):
            self.policy = policy
            
        def predict(self, observation, state=None, episode_start=None, deterministic=False):
            return self.policy.predict(observation, state, episode_start, deterministic)
            
    model = BCPolicyModel(policy)
    
    print("Evaluating policy (10 episodes)...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

if __name__ == "__main__":
    evaluate()
