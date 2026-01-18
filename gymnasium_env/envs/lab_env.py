import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .lab_generator import LabGenerator

class LabEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, render_mode=None, number_of_rooms=4):
        """
        Args:
            number_of_rooms: Total number of rooms. Must be perfect square (4, 9, 16...)
        """
        self.num_rooms = number_of_rooms 
        self.lab = LabGenerator(number_of_rooms=self.num_rooms)
        self.grid_size = self.lab.grid_size
        
        # Actions: 0:Right, 1:Up, 2:Left, 3:Down, 4:Backtrack, 5..:Buttons
        # Button count is fixed to number_of_rooms as per LabGenerator logic
        self.action_space = spaces.Discrete(5 + self.lab.number_of_buttons)
        
        # Observations
        self.observation_space = spaces.Dict({
            "agent_location": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            "goal_location": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int),
            # Door states: 1 = Open, 0 = Closed/Wall
            "door_states": spaces.Box(0, 1, shape=(self.num_rooms, self.num_rooms), dtype=int),
            "button_locations": spaces.Box(0, 1, shape=(self.num_rooms, self.lab.number_of_buttons), dtype=int),
            "last_pos": spaces.Box(0, self.grid_size - 1, shape=(2,), dtype=int)
        })
        
        self.render_mode = render_mode
        agent_r, agent_c = self.lab.index_to_coord(self.lab.start_room)
        self.agent_location = np.array([agent_r, agent_c])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Note: LabGenerator uses numpy.random global state.
        
        self.lab.generate_lab()
        #TODO also set new goal and start room
        
        # Set agent to start room
        start_idx = self.lab.start_room
        agent_r, agent_c = self.lab.index_to_coord(start_idx)
        self.agent_location = np.array([agent_r, agent_c])
        self.last_pos = np.array([-1, -1])
        
        return self._get_obs(), {}

    def step(self, action):
        reward = -1
        terminated = False
        truncated = False
        
        current_r, current_c = self.agent_location
        current_idx = self.lab.coord_to_index(current_r, current_c)
        
        if action < 5: # Move
            # Calc new pos
            new_r, new_c = current_r, current_c
            if action == 0: new_c += 1 # Right
            elif action == 1: new_r -= 1 # Up
            elif action == 2: new_c -= 1 # Left
            elif action == 3: new_r += 1 # Down
            elif action == 4: # Backtrack
                if self.last_pos[0] != -1: # Only valid if last_pos is set
                    new_r, new_c = self.last_pos
            
            # Check bounds
            if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                target_idx = self.lab.coord_to_index(new_r, new_c)
                
                # Check if door connects and is open (unless backtracking)
                is_open = self.lab.door_state_matrix[current_idx, target_idx] == 1
                is_backtrack = (action == 4)
                
                if is_open or is_backtrack:
                    # Move successful
                    self.last_pos = self.agent_location.copy()
                    self.agent_location = np.array([new_r, new_c])
                    
                    # Check Goal
                    if target_idx == self.lab.goal_room:
                        terminated = True
                        reward = 10.0
                else:
                    # Blocked (Wall or Closed Door)
                    pass
        else: # Button
            btn_idx = action - 5
            # Check if button exists in current room
            if btn_idx < self.lab.number_of_buttons:
                if self.lab.button_location_matrix[current_idx, btn_idx] == 1:
                    # Toggle doors
                    behavior = self.lab.button2door_behavior_matrix[btn_idx]
                    
                    # XOR current states with behavior
                    # Behavior 1 means "Toggle this edge"
                    current_states = self.lab.door_state_matrix
                    new_states = np.logical_xor(current_states, behavior).astype(int)
                    
                    # Enforce Walls stay Walls (TransMatrix == 0 -> State = 0)
                    new_states = new_states * self.lab.room_trans_matrix
                    
                    self.lab.door_state_matrix = new_states
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        goal_r, goal_c = self.lab.index_to_coord(self.lab.goal_room)
        return {
            "agent_location": self.agent_location,
            "goal_location": np.array([goal_r, goal_c], dtype=int),
            "door_states": self.lab.door_state_matrix.copy().astype(int),
            "button_locations": self.lab.button_location_matrix.copy().astype(int),
            "last_pos": self.last_pos
        }
