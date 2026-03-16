import re

class LocalCommandMatcher:
    def __init__(self):
        self.COMMAND_MAP = {
            "stop": "STOP", "halt": "STOP", "stoppen": "STOP", "anhalten": "STOP",
            "go": "MOVE", "geh": "MOVE", "laufen": "MOVE", "move": "MOVE"
        }
        
    def process_input(self, user_text):
        words = user_text.lower().split()
        for word in words:
            if word in self.COMMAND_MAP:
                return self.COMMAND_MAP[word]
        return "NO_MATCH"