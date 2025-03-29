import random

class Agent:
    def __init__(self):
        pass

    def select_action(self, dynamic_action_space):
        # Filter out empty subaction lists
        valid_actions = [(i, actions) for i, actions in enumerate(dynamic_action_space) if actions]

        if not valid_actions:
            return None, None  # No valid actions at all

        # Choose a valid action randomly
        action, actions = random.choice(valid_actions)
        subaction = random.randint(0, len(actions) - 1) if action != 2 else None

        return action, subaction
