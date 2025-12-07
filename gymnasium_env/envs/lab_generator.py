import numpy as np
from collections import deque


class LabGenerator:
    def __init__(self, number_of_rooms=4):
        self.number_of_rooms = number_of_rooms
        self.grid_size = int(np.sqrt(self.number_of_rooms))
        assert self.grid_size ** 2 == self.number_of_rooms, "Number of rooms must be a perfect square for Grid World"

        self.room_trans_matrix = None
        # pick a random start room
        self.start_room = np.random.randint(self.number_of_rooms)
        # pick a random goal room that is not the start room
        self.goal_room = np.random.choice([x for x in range(self.number_of_rooms) if x != self.start_room])
        self.door_state_matrix = None
        self.button_location_matrix = None
        self.button2door_behavior_matrix = None
        self.valid_layout = False
        # User requested potentially num_rooms buttons
        self.number_of_buttons = self.number_of_rooms
        self.generate_lab()

    def get_grid_adjacency(self):
        # Returns a mask of all VALID connections in a grid
        adj = np.zeros((self.number_of_rooms, self.number_of_rooms), dtype=int)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                curr = r * self.grid_size + c
                # Down
                if r + 1 < self.grid_size:
                    next_node = (r + 1) * self.grid_size + c
                    adj[curr, next_node] = adj[next_node, curr] = 1
                # Right
                if c + 1 < self.grid_size:
                    next_node = r * self.grid_size + (c + 1)
                    adj[curr, next_node] = adj[next_node, curr] = 1
        # Self loop
        np.fill_diagonal(adj, 1)
        return adj

    def generate_rooms(self):
        # Generate random matrix
        rooms = np.random.randint(0, 2, size=(self.number_of_rooms, self.number_of_rooms))
        rooms = np.triu(rooms, 1) # Upper triangle
        rooms = rooms + rooms.T # Symmetric
        np.fill_diagonal(rooms, 1) # Self connected
        
        # Apply Grid Mask
        grid_adj = self.get_grid_adjacency()
        rooms = rooms * grid_adj
        return rooms

    def sanity_check(self):
        """
        Check if there is a path from start room to goal room in the labyrinth.
        """
        queue = deque([self.start_room])
        visited = set()
        
        while queue:
            current = queue.popleft()
            if current == self.goal_room:
                return True
            if current in visited:
                continue
            visited.add(current)
            for neighbor in range(self.number_of_rooms):
                if self.room_trans_matrix[current, neighbor] == 1 and neighbor not in visited:
                    queue.append(neighbor)
        return False

    def generate_door_states(self):
        # Random initial states
        self.door_state_matrix = self.generate_rooms()
        # close doors where transitions are not possible (walls)
        zero_mask = self.room_trans_matrix == 0
        self.door_state_matrix[zero_mask] = 0

    def generate_button_locations(self):
        # Randomly place buttons.
        # Shape: (num_rooms, num_buttons)
        # Assuming we want random distribution.
        self.button_location_matrix = np.random.randint(0, 2, size=(self.number_of_rooms, self.number_of_buttons))

    def generate_button2door_behavior(self):
        self.button2door_behavior_matrix = np.array([self.generate_single_button_matrix() for _ in range(self.number_of_buttons)])

    def generate_single_button_matrix(self):
        # create room x room matrix that shows which doors to toogle for one button
        single_button_matrix = np.random.randint(0, 2, size=(self.number_of_rooms, self.number_of_rooms))
        single_button_matrix = np.triu(single_button_matrix, 1)
        single_button_matrix += single_button_matrix.T
        
        # Apply Grid Mask so buttons only affect physical doors
        grid_adj = self.get_grid_adjacency()
        single_button_matrix = single_button_matrix * grid_adj
        
        # set diagonal to zero as there are no doors that lead to the same room
        np.fill_diagonal(single_button_matrix, 0)
        return single_button_matrix

    def generate_lab(self):
        while not self.valid_layout:
            self.room_trans_matrix = self.generate_rooms()
            self.valid_layout = self.sanity_check()
        self.generate_door_states()
        self.generate_button_locations()
        self.generate_button2door_behavior()

    def coord_to_index(self, r, c):
        return r * self.grid_size + c

    def index_to_coord(self, i):
        return np.array([i // self.grid_size, i % self.grid_size])
