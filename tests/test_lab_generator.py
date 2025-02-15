from lab_generator import LabGenerator
import numpy as np

def test_generate_rooms():
    lab = LabGenerator()
    lab.generate_rooms()
    print(lab.room_trans_matrix)

def test_sanity_check():
    lab = LabGenerator()
    lab.room_trans_matrix = np.array([[1, 1, 0, 0],
                                      [1, 1, 1, 0],
                                      [0, 1, 1, 1],
                                      [0, 0, 1, 1]])
    lab.start_room = 0
    lab.goal_room = 3

    assert lab.sanity_check() == True, f"Expected True, but got False for matrix: {lab.room_trans_matrix} (start_room={lab.start_room}, goal_room={lab.goal_room})"

    lab.room_trans_matrix = np.array([[1, 0, 0, 0],
                                      [0, 1, 0, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
    
    assert lab.sanity_check() == False, f"Expected False, but got True for matrix: {lab.room_trans_matrix} (start_room={lab.start_room}, goal_room={lab.goal_room})"