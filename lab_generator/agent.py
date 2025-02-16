import random


class Agent:
    def __init__(self):
        pass

    def select_action(self, dynamic_action_space):
        # pick action
        action = random.randint(0, len(dynamic_action_space))
        # pick subaction
        if action != 2:
            subaction = random.randint(0, len(dynamic_action_space[action]))
        else:
            subaction = None
        return action, subaction