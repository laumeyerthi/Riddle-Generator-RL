import torch
import gymnasium as gym
import sys
import os
import glob
from skrl.agents.torch.q_learning import Q_LEARNING, Q_LEARNING_DEFAULT_CONFIG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rl_agent.q_learning import DiscreteObservationWrapper, EpsilonGreedyPolicy
from gymnasium_env.envs.lab_env import LabEnv
from skrl.envs.wrappers.torch import wrap_env
from skrl.utils import set_seed

def evaluate():
    set_seed(42)
    # Try to load specific model file first
    model_path = os.path.join(os.path.dirname(__file__), '..', 'q_learning_agent.pt')
    if os.path.exists(model_path):
        print(f"Loading from: {model_path}")
        latest_checkpoint = model_path
    else:
        # Find latest checkpoint in runs
        run_dir = os.path.join(os.path.dirname(__file__), '..', 'runs', 'torch', 'LabEnv_QLearning')
        runs = glob.glob(os.path.join(run_dir, '*'))
        if not runs:
            print("No Q-Learning runs found and no q_learning_agent.pt found.")
            return
        latest_run = max(runs, key=os.path.getmtime)
        print(f"Loading from run: {latest_run}")
        
        checkpoints = glob.glob(os.path.join(latest_run, 'checkpoints', '*.pt'))
        if not checkpoints:
             print("No checkpoints found in latest run.")
             return
        latest_checkpoint = max(checkpoints, key=os.path.getmtime)
        print(f"Checkpoint: {latest_checkpoint}")

    # Setup Env
    base_env = LabEnv(number_of_rooms=4)
    env = DiscreteObservationWrapper(base_env, num_states=20000)
    env = wrap_env(env)
    device = env.device

    # Setup Agent
    models = {}
    # Use Epsilon 0 for evaluation (Greedy)
    models["policy"] = EpsilonGreedyPolicy(env.observation_space, env.action_space, device, num_envs=env.num_envs, epsilon=0.0)

    cfg = Q_LEARNING_DEFAULT_CONFIG.copy()
    cfg["experiment"]["write_interval"] = 0 # Disable writing
    
    agent = Q_LEARNING(models=models,
                       memory=None,
                       cfg=cfg,
                       observation_space=env.observation_space,
                       action_space=env.action_space,
                       device=device)

    print("Loading agent...")
    agent.load(latest_checkpoint)
    
    # Run
    agent.set_running_mode("eval")
    print("Starting Evaluation (5 Episodes)...")
    
    for i in range(5): # 5 Episodes
        obs, _ = env.reset()
        terminated = False
        truncated = False
        metrics = {"reward": 0}
        step = 0
        
        while not (terminated or truncated) and step < 100:
            with torch.no_grad():
                actions = agent.act(obs, timestep=0, timesteps=0)[0]
            
            obs, rewards, terminated, truncated, infos = env.step(actions)
            metrics["reward"] += rewards.item()
            step += 1
            
        print(f"Episode {i+1} Reward: {metrics['reward']}")

if __name__ == "__main__":
    evaluate()
