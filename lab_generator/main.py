from .environment import Environment


def main():
    my_env = Environment()
    max_iterations = 1000000
    count = 0

    while not my_env.done:
        my_env.step()
        count += 1
        if count == max_iterations or my_env.truncated is True:
            print(f"No solution for \n {my_env.lab.room_trans_matrix} \n found")
            my_env.reset()
            count = 0
    print(f"""Solution for room_trans_matrix: 
    {my_env.lab.room_trans_matrix}

    button_location_matrix: 
    {my_env.lab.button_location_matrix}

    button2door_behavior_matrix: 
    {my_env.lab.button2door_behavior_matrix}

    Found! 

    History [Last room, action, subaction]: 
    {my_env.history}
    """)


        


if __name__ == "__main__":
    main()  # Call the main function
    
