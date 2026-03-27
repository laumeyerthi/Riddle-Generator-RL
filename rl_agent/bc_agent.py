import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, TimeLimit
import numpy as np
import collections
import heapq
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gymnasium_env.envs.lab_env import LabEnv

from imitation.data.types import TrajectoryWithRew
from imitation.algorithms import bc

def a_star_solve(unwrapped_env):
    lab = unwrapped_env.lab
     
    # 1. Precalculate shortest physical paths (Heuristic)
    h_dist = np.full(lab.number_of_rooms, np.inf)
    h_dist[lab.goal_room] = 0
    q = collections.deque([lab.goal_room])
    while q:
        curr = q.popleft()
        for neighbor in range(lab.number_of_rooms):
            if lab.room_trans_matrix[curr, neighbor] == 1 and h_dist[neighbor] == np.inf:
                h_dist[neighbor] = h_dist[curr] + 1
                q.append(neighbor)
    
    # Start state
    start_room = lab.start_room
    start_state = (start_room, 0, -1)
    
    # Priority queue: (f, idx, g, state, path_of_actions)
    pq = [(h_dist[start_room], 0, 0, start_state, [])]
    visited = {start_state: 0} 
    idx = 1
    
    while pq:
        f, _, g, state, path = heapq.heappop(pq)
        curr_room, curr_mask, last_room = state
        
        if curr_room == lab.goal_room:
            return path
            
        curr_r, curr_c = lab.index_to_coord(curr_room)
        
        # Calculate current door states
        curr_doors_row = lab.door_state_matrix[curr_room].copy()
        for btn_idx in range(lab.number_of_buttons):
            if (curr_mask >> btn_idx) & 1:
                curr_doors_row = np.bitwise_xor(curr_doors_row, lab.button2door_behavior_matrix[btn_idx][curr_room])
                
        # Move actions
        deltas = [(0, 1), (-1, 0), (0, -1), (1, 0)] # Right, Up, Left, Down
        for action, (dr, dc) in enumerate(deltas):
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < lab.grid_size and 0 <= nc < lab.grid_size:
                neighbor = lab.coord_to_index(nr, nc)
                if lab.room_trans_matrix[curr_room, neighbor] == 1 and curr_doors_row[neighbor] == 1:
                    next_state = (neighbor, curr_mask, curr_room)
                    next_g = g + 1
                    if next_state not in visited or next_g < visited[next_state]:
                        visited[next_state] = next_g
                        heapq.heappush(pq, (next_g + h_dist[neighbor], idx, next_g, next_state, path + [action]))
                        idx += 1
                        
        # Backtrack action (Action 4)
        if last_room != -1:
            next_state = (last_room, curr_mask, curr_room)
            next_g = g + 1
            if next_state not in visited or next_g < visited[next_state]:
                visited[next_state] = next_g
                heapq.heappush(pq, (next_g + h_dist[last_room], idx, next_g, next_state, path + [4]))
                idx += 1
                
        # Button actions (Action 5 + btn_idx)
        available_buttons = np.where(lab.button_location_matrix[curr_room] == 1)[0]
        for btn_idx in available_buttons:
            next_mask = curr_mask ^ (1 << btn_idx)
            next_state = (curr_room, next_mask, last_room)
            next_g = g + 1
            if next_state not in visited or next_g < visited[next_state]:
                visited[next_state] = next_g
                heapq.heappush(pq, (next_g + h_dist[curr_room], idx, next_g, next_state, path + [5 + btn_idx]))
                idx += 1
                
    return []

def generate_expert_demonstrations(env, num_episodes=50):
    trajectories = []
    
    for i in range(num_episodes):
        obs, _ = env.reset()
        unwrapped_env = env.unwrapped
        
        
        opt_actions = a_star_solve(unwrapped_env)
        
        if not opt_actions:
            print(f"Warning: Episode {i} is not solvable. Skipping...")
            continue
            
        episode_obs = [obs]
        episode_acts = []
        episode_rews = []
        episode_infos = []
        
        for action in opt_actions:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_obs.append(obs)
            episode_acts.append(action)
            episode_rews.append(reward)
            episode_infos.append(info)
            if terminated or truncated:
                break
                
        trajectories.append(
            TrajectoryWithRew(
                obs=np.array(episode_obs),
                acts=np.array(episode_acts),
                infos=np.array(episode_infos),
                rews=np.array(episode_rews),
                terminal=True,
            )
        )
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_episodes} trajectories...")
            
    return trajectories

def train():
    number_of_rooms = 4
    print(f"Initializing Environment for {number_of_rooms} rooms...")
    env = LabEnv(number_of_rooms=number_of_rooms)
    env = TimeLimit(env, max_episode_steps=100)
    env = FlattenObservation(env)
    
    print("Generating expert demonstrations utilizing A*...")
    trajectories = generate_expert_demonstrations(env, num_episodes=1000)
    
    print(f"Collected {len(trajectories)} trajectories.")
    
    print("Initializing Behavioral Cloning (BC) Agent...")
    rng = np.random.default_rng()
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=trajectories,
        rng=rng,
        custom_logger=None,
    )
    
    print("Training BC Agent...")
    bc_trainer.train(n_epochs=1)
    
    print("Saving Model...")
    # Using stable-baselines3 policy saving functionality
    bc_trainer.policy.save("bc_lab_env")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()
