import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lab_env import LabEnv

class LabEnvCNN(LabEnv):
    def __init__(self, render_mode=None, number_of_rooms=4, valid_seeds=None, max_rooms=None):
        super().__init__(render_mode, number_of_rooms, valid_seeds, max_rooms)
        
        # Override observation space
        # Grid size of the spatial map: 2 * grid_size + 1
        # Example: 3x3 rooms -> 7x7 spatial grid
        self.spatial_size = 2 * self.grid_size + 1
        
        # Channels:
        # 0: Walls
        # 1: Agent
        # 2: Goal
        # 3: Closed Doors
        # 4 to 4+num_buttons-1: Button Locations
        # 4+num_buttons to 4+2*num_buttons-1: Button Behaviors (Doors affected)
        self.num_channels = 4 + 2 * self.lab.number_of_buttons
        
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=(self.num_channels, self.spatial_size, self.spatial_size), 
            dtype=np.float32
        )

    def _get_obs(self):
        obs = np.zeros((self.num_channels, self.spatial_size, self.spatial_size), dtype=np.float32)
        
        # Channel 0: Walls
        obs[0, :, :] = 1.0
        
        # Clear out the room centers
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                obs[0, 2*r + 1, 2*c + 1] = 0.0
                
        # Clear out paths where there is a room transition (i.e. not a solid wall)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr_idx = self.lab.coord_to_index(r, c)
                curr_sr, curr_sc = 2*r + 1, 2*c + 1
                
                # Right
                if c + 1 < self.grid_size:
                    right_idx = self.lab.coord_to_index(r, c + 1)
                    if self.lab.room_trans_matrix[curr_idx, right_idx] == 1:
                        # Clear wall
                        obs[0, curr_sr, curr_sc + 1] = 0.0
                # Down
                if r + 1 < self.grid_size:
                    down_idx = self.lab.coord_to_index(r + 1, c)
                    if self.lab.room_trans_matrix[curr_idx, down_idx] == 1:
                        # Clear wall
                        obs[0, curr_sr + 1, curr_sc] = 0.0

        # Channel 1: Agent Location
        agent_r, agent_c = self.agent_location
        obs[1, 2*agent_r + 1, 2*agent_c + 1] = 1.0
        
        # Channel 2: Goal Location
        goal_r, goal_c = self.lab.index_to_coord(self.lab.goal_room)
        obs[2, 2*goal_r + 1, 2*goal_c + 1] = 1.0
        
        # Channel 3: Closed Doors
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr_idx = self.lab.coord_to_index(r, c)
                curr_sr, curr_sc = 2*r + 1, 2*c + 1
                
                # Right
                if c + 1 < self.grid_size:
                    right_idx = self.lab.coord_to_index(r, c + 1)
                    if self.lab.room_trans_matrix[curr_idx, right_idx] == 1:
                        if self.lab.door_state_matrix[curr_idx, right_idx] == 0: # 0 is closed
                            obs[3, curr_sr, curr_sc + 1] = 1.0
                            
                # Down
                if r + 1 < self.grid_size:
                    down_idx = self.lab.coord_to_index(r + 1, c)
                    if self.lab.room_trans_matrix[curr_idx, down_idx] == 1:
                        if self.lab.door_state_matrix[curr_idx, down_idx] == 0:
                            obs[3, curr_sr + 1, curr_sc] = 1.0

        # Buttons and Behaviors
        base_btn_idx = 4
        base_beh_idx = 4 + self.lab.number_of_buttons
        
        for b in range(self.lab.number_of_buttons):
            # Button Location
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    curr_idx = self.lab.coord_to_index(r, c)
                    if self.lab.button_location_matrix[curr_idx, b] == 1:
                        obs[base_btn_idx + b, 2*r + 1, 2*c + 1] = 1.0
            
            # Button Behavior (which doors it toggles)
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    curr_idx = self.lab.coord_to_index(r, c)
                    curr_sr, curr_sc = 2*r + 1, 2*c + 1
                    
                    # Right
                    if c + 1 < self.grid_size:
                        right_idx = self.lab.coord_to_index(r, c + 1)
                        if self.lab.button2door_behavior_matrix[b, curr_idx, right_idx] == 1:
                            obs[base_beh_idx + b, curr_sr, curr_sc + 1] = 1.0
                            
                    # Down
                    if r + 1 < self.grid_size:
                        down_idx = self.lab.coord_to_index(r + 1, c)
                        if self.lab.button2door_behavior_matrix[b, curr_idx, down_idx] == 1:
                            obs[base_beh_idx + b, curr_sr + 1, curr_sc] = 1.0

        return obs
