import numpy as np
from lab_generator import LabGenerator
from lab_generator import Environment

# Unit Tests
def test_press_button():
    lab = LabGenerator()
    agent = None
    env = Environment(lab, agent)
    # Create test matrices
    lab.room_trans_matrix = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    lab.button2door_behavior_matrix = np.array([[
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]])
    
    lab.door_state_matrix = np.array([
        [1, 0, 0],
        [0, 1, 1],
        [0, 1, 1]
    ])

    button = 0  # Column index to check
    
    expected_door_state_matrix = np.array([
        [1, 1, 0],
        [1, 1, 0],
        [0, 0, 1]
    ])

    print("Door State before:\n", lab.door_state_matrix)
    env.press_button(button)
    print("Door State after:\n", lab.door_state_matrix)
    # Assertions
    assert np.array_equal(lab.door_state_matrix, expected_door_state_matrix), "Door_state_matrices do not match expected"

