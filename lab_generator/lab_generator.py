import numpy as np
from collections import deque


class LabGenerator:
    def __init__(self, number_of_rooms=4):
        self.number_of_rooms = number_of_rooms
        self.room_trans_matrix = None
        # pick a random start room
        self.start_room = np.random.randint(self.number_of_rooms)
        # pick a random goal room that is not the start room
        self.goal_room = np.random.choice([x for x in range(self.number_of_rooms) if x != self.start_room])
        self.door_state_matrix = None
        self.button_location_matrix = None
        self.button2door_behavior_matrix = None
        self.valid_layout = False
        self.number_of_buttons = np.random.randint(self.number_of_rooms)

    def generate_rooms(self):
        # Generate matrix with random 0 and 1 with size num_rooms
        self.room_trans_matrix = np.random.randint(0, 2, size=(self.number_of_rooms, self.number_of_rooms))
        # make sure each room is connected with itself
        for i in range(self.number_of_rooms):
            self.room_trans_matrix[i][i] = 1
        # make it symentric to make sure transitions go both ways
        np.triu(self.room_trans_matrix)
        self.room_trans_matrix += self.room_trans_matrix.T - np.diag(self.room_trans_matrix.diagonal())
        self.room_trans_matrix[self.room_trans_matrix > 1] = 1

    def sanity_check(self):
        """
        Check if there is a path from start room to goal room in the labyrinth.
        Matrix: Transition matrix (numpy array)
        start: Index of the starting room (0-based)
        goal: Index of the goal room (0-based)
        """
        # Create a queue for BFS and a set to track visited rooms
        queue = deque([self.start_room])
        visited = set()
        
        while queue:
            current = queue.popleft()
            
            # If we reached the goal room
            if current == self.goal_room:
                return True
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            
            # Check all connected rooms
            for neighbor in range(self.number_of_rooms):
                if self.room_trans_matrix[current, neighbor] == 1 and neighbor not in visited:
                    queue.append(neighbor)
        
        # If BFS finishes without finding the goal room
        return False

    def generate_door_states(self):
        self.door_state_matrix = self.generate_rooms()
        # close doors where no transtions between rooms are not possible
        zero_mask = self.room_trans_matrix == 0
        self.door_state_matrix[zero_mask] = 0

    def generate_button_locations(self):
        self.button_location_matrix = np.random.randint(0, 2, size=(self.number_of_rooms, self.number_of_buttons))

    def generate_button2door_behavior(self):
        # create a button behavior matrix for each button in a stacked matrix
        self.button2door_behavior_matrix = np.array([self.generate_single_button_matrix() for _ in range(self.number_of_buttons)])

    def generate_single_button_matrix(self):
        # create room x room matrix that shows which doors to toogle for one button
        single_button_matrix = np.random.randint(0, 2, size=(self.number_of_rooms, self.number_of_rooms))
        # cut in half for door transition in both ways
        single_button_matrix = np.triu(single_button_matrix, 1)
        # set diagonal to zero as there are no doors that lead to the same room they are in
        single_button_matrix += single_button_matrix.T

    def generate_lab(self):
        while self.valid_layout:
            self.generate_rooms()
            self.valid_layout = self.sanity_check()
        self.generate_door_states()
        self.generate_button_locations()
        self.generate_button2door_behavior()
