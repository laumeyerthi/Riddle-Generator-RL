import numpy as np
from lab_generator import LabGenerator
from lab_generator import Environment

# Unit Tests
def test_press_button():
    env = Environment()
    # Create test matrices
    env.lab.room_trans_matrix = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [0, 1, 1]
    ])
    
    env.lab.button2door_behavior_matrix = np.array([[
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]])
    
    env.lab.door_state_matrix = np.array([
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

    print("Door State before:\n", env.lab.door_state_matrix)
    env.press_button(button)
    print("Door State after:\n", env.lab.door_state_matrix)
    # Assertions
    assert np.array_equal(env.lab.door_state_matrix, expected_door_state_matrix), "Door_state_matrices do not match expected"

