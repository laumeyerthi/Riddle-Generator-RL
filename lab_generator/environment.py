import numpy as np
from lab_generator import LabGenerator
from lab_generator import agent


class Environment():
    def __init__(self, lab=LabGenerator(), agent=agent()):
        self.lab = lab
        self.agent = agent
        self.dynamic_action_space = None
        self.history = []
        self.current_room = self.lab.start_room
        self.done = False
        self.last_room = None

    def step(self):
        action, subaction = self.agent.select_action(self.dynamic_action_space)

        # press button
        if action == 0:
            self.press_button(subaction)

        # move
        if action == 1:
            self.move(subaction)

        # backtrack
        if action == 2:
            self.backtrack()

        if self.check_victory_condition():
            self.done = True


    def generate_action_space(self):
        # delete past action space
        self.dynamic_action_space = []
        # get the indices of the buttons in the current room
        self.dynamic_action_space.append([np.where(self.lab.button_location_matrix[self.current_room] == 1)[0].tolist()])
        #get the indices of the possible room transtions in the current room
        room_transitions = np.where(self.lab.door_state_matrix[self.current_room] == 1)[0]
        # remove the current room
        room_transitions = room_transitions[room_transitions != self.current_room]
        # append to possible move throu door actions to the action space
        self.dynamic_action_space.append([room_transitions.tolist()])
        # append the backtrack action
        if self.last_room is not None:
            self.dynamic_action_space.append([self.backtrack])


    def check_victory_condition(self):
        if self.current_room == self.lab.goal_room:
            return True
        else:
            return False
        
    def press_button(self, button):
        button_behavior = self.lab.button2door_behavior_matrix[button]
        # change the doors where the button controls it and a transition is possible due to room_trans_matrix
        self.lab.door_state_matrix[(button_behavior == 1) & (self.lab.room_trans_matrix == 1)] ^= 1
        

    def move(self, door):
        self.last_room = self.current_room
        self.current_room = self.dynamic_action_space[1][door]

    def backtrack(self):
        self.current_room = self.last_room
