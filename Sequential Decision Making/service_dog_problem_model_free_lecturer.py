
def get_data_textbook():
    """
    Creates the model-free data for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        legal_action: Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """

    # Define legal actions
    legal_actions = {
        "Room 1": ["Go to room 2"],
        "Room 2": ["Go to room 1", "Go to room 3", "Go outside"],
        "Room 3": ["Go to room 2", "Search"],
        "Outside": ["Go inside", "Go outside"],
        "Found item": []
    }

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 2"): -1,
              ("Room 2", "Go to room 1"): -2,
              ("Room 2", "Go to room 3"): -1,
              ("Room 2", "Go outside"): 0,
              ("Room 3", "Go to room 2"): -2,
              ("Room 3", "Search"): 10,
              ("Outside", "Go outside"): -2,    # =-1 in textbook figure, but =-2 in textbook GitHub code (and used for textbook values)
              ("Outside", "Go inside"): 0              
              }

    # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
    transition_prob = {
        ("Room 1", "Go to room 2"): {"Room 2": 1.0},
        ("Room 2", "Go to room 1"): {"Room 1": 1.0},
        ("Room 2", "Go to room 3"): {"Room 3": 1.0},
        ("Room 2", "Go outside"): {"Outside": 1.0},
        ("Room 3", "Go to room 2"): {"Room 2": 1.0},
        ("Room 3", "Search"): {"Found item": 1.0},
        ("Outside", "Go outside"): {"Outside": 1.0},   
        ("Outside", "Go inside"): {"Room 2": 1.0} 
    }

    return legal_actions, reward, transition_prob

