import random


class Agent:
    def __init__(self):
        pass

    def select_Action(self, dynamic_action_space):
        # pick action
        action = random.randint(0, range(dynamic_action_space))
        # pick subaction
        if action is not 2:
            subaction = random.randint(0, range(dynamic_action_space[action]))
        else:
            subaction = None
        return action, subaction