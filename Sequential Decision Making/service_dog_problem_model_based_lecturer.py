
def get_data_task_2():
    """
    Creates the model-based data for the service dog problem based on "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Outside", "Found item"]

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 2"): 1,
              ("Room 2", "Go to room 1"): -2,
              ("Room 2", "Go to room 3"): 3,
              ("Room 2", "Go outside"): -2,
              ("Room 3", "Go to room 2"): -4,
              ("Room 3", "Search"): 12,
              ("Outside", "Go outside"): 1,  
              ("Outside", "Go inside"): 3              
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

    return states, reward, transition_prob


def get_data_task_3():
    """
    Creates the model-based data for the service dog example based on "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Room 4", "Outside", "Found item"]

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 1"): -1,
              ("Room 1", "Go to room 2"): -1,
              ("Room 1", "Go to room 4"): 1,
              #
              ("Room 2", "Go to room 3"): -1,
              ("Room 2", "Go outside"): 1,
              # 
              ("Room 3", "Go outside"): 1,
              ("Room 3", "Search"): 10,
              # 
              ("Room 4", "Search"): 10,
              # 
              ("Outside", "Go inside"): 0   
              }

    # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
    transition_prob = {
        ("Room 1", "Go to room 1"): {"Room 1": 1.0},
        ("Room 1", "Go to room 2"): {"Room 2": 1.0},
        ("Room 1", "Go to room 4"): {"Room 4": 1.0},
        #
        ("Room 2", "Go to room 3"): {"Room 3": 1.0},
        ("Room 2", "Go outside"): {"Outside": 1.0},
        # 
        ("Room 3", "Go outside"): {"Outside": 1.0},
        ("Room 3", "Search"): {"Found item": 1.0},
        # 
        ("Room 4", "Search"): {"Found item": 1.0},
        # 
        ("Outside", "Go inside"): {"Room 2": 1.0}  
        }
    
    return states, reward, transition_prob


def get_data_task_4_and_5():
    """
    Creates the model-based data for the service dog example based on "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        states: The set of all possible configurations or observations of the environment that we can be in.
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    """
    
    # State space [𝒮]: The set of all possible configurations or observations of the environment that we can be in.
    states = ["Room 1", "Room 2", "Room 3", "Room 4", "Outside", "Found item"]

    # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
    reward = {("Room 1", "Go to room 1"): -1,
              ("Room 1", "Go to room 2"): -1,
              ("Room 1", "Go to room 4"): 1,
              #
              ("Room 2", "Go to room 3"): -1,
              ("Room 2", "Go outside"): 1,
              # 
              ("Room 3", "Go outside"): 1,
              ("Room 3", "Search"): 10,
              # 
              ("Room 4", "Search"): 10,
              # 
              ("Outside", "Go inside"): 0   
              }

    # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
    # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
    transition_prob = {
        ("Room 1", "Go to room 1"): {"Room 1": 0.8, "Outside": 0.2},
        ("Room 1", "Go to room 2"): {"Room 2": 0.8, "Outside": 0.2},
        ("Room 1", "Go to room 4"): {"Room 4": 0.8, "Outside": 0.2},
        #
        ("Room 2", "Go to room 3"): {"Room 3": 0.8, "Outside": 0.2},
        ("Room 2", "Go outside"): {"Outside": 1.0},
        # 
        ("Room 3", "Go outside"): {"Outside": 1.0},
        ("Room 3", "Search"): {"Found item": 0.8, "Outside": 0.2},
        # 
        ("Room 4", "Search"): {"Found item": 0.8, "Outside": 0.2},
        # 
        ("Outside", "Go inside"): {"Room 2": 0.8, "Outside": 0.2}  
        }
    
    return states, reward, transition_prob


def get_policy_random_task_3_and_4():
    """
    Creates a random policy for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        policy: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    """

    # Policy [𝜋(a|s)]: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    policy_random = {
        "Room 1": {"Go to room 1": 1/3, "Go to room 2": 1/3, "Go to room 4": 1/3},
        "Room 2": {"Go to room 3": 0.5, "Go outside": 0.5},
        "Room 3": {"Go outside": 0.5, "Search": 0.5},
        "Room 4": {"Search": 1.0},
        "Outside": {"Go inside": 1.0},
        "Found item": {}
    }

    return policy_random


def get_policy_task_5():
    """
    Creates a random policy for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        policy: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    """

    # Define random policy [𝜋(a|s)]: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
    policy = {
        "Room 1": {"Go to room 4": 1.0},
        "Room 2": {"Go to room 3": 0.8, "Go outside": 0.2},
        "Room 3": {"Go outside": 0.5, "Search": 0.5},
        "Room 4": {"Search": 1.0},
        "Outside": {"Go inside": 1.0},
        "Found item": {}
    }

    return policy


def Q_value_iteration(states, reward, transition_prob, discount, delta_threshold=0.00001):
    """
    Given a policy function, reward function, transition probability function, discount factor and a delta threshold,
    find a optimal policy 𝜋* along with optimal state value function V*.

    Args:
        states: State space 𝒮
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′. There may be multiple successor states.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy of the estimation.
    """

    # Initialize counter
    count = 0

    # Initialize & retrieve legal actions from reward function
    legal_actions = {state: [] for state in states}
    for (state, action) in reward.keys():
        legal_actions[state].append(action)

    # Initialize state-action value function to be all zeros for all states.
    Q = {s: {a: 0 for a in legal_actions[s]} for s in states}

    while True:
        delta = 0
        for state in states:
            # For every legal action
            for action in legal_actions[state]:   

                # Old Q (needed to compute convergence)
                old_Q = Q[state][action]
            
                # Immediate reward
                Q[state][action] = reward[(state, action)]

                # Future reward                
                for successor_state, successor_state_prob in transition_prob[(state, action)].items():
                    # Note one state-action might have multiple successor states with different transition probability
                    # Weight by the transition probability

                    if len(legal_actions[successor_state]) > 0:
                        Q[state][action] += discount * successor_state_prob * max(Q[successor_state][successor_action] for successor_action in legal_actions[successor_state])

                # Use the maximum expected returns across all actions as state value.
                delta = max(delta, abs(old_Q - Q[state][action]))

        count += 1
        if delta < delta_threshold:            
            break

    # Step 2: Compute an optimal policy from optimal state value function.
    optimal_policy = {s: dict() for s in states}
    for state in states:
        # Store the expected returns for each action.
        estimated_returns = {}

        # For every legal action
        for action in legal_actions[state]:   
                            
            estimated_returns[action] = Q[state][action]
        
        if len(estimated_returns) > 0:
            # Get the best action a based on the q(s, a) values, notice the action is the key in the dict estimated_returns.
            best_action = max(estimated_returns, key=estimated_returns.get)

            # Set the probability to 1.0 for the best action.
            optimal_policy[state][best_action] = 1.0

    # Round state value function
    for state in Q:
        for action in Q[state]:
            Q[state][action] = round(Q[state][action], 2)

    print(f"State value function after {count} iterations: {Q}")
    print(f"Optimal policy after {count} iterations: {optimal_policy}")


def value_iteration_tasks():
    from service_dog_problem_model_based import value_iteration, get_data_textbook
    
    # Default
    print("Default:")
    states, reward, transition_prob = get_data_textbook()
    value_iteration(states, reward, transition_prob, discount=0.9)

    # Task 1
    print("Task 1:")
    states, reward, transition_prob = get_data_textbook()
    value_iteration(states, reward, transition_prob, discount=0.2)

    # Task 2
    print("Task 2:")
    states, reward, transition_prob = get_data_task_2()
    value_iteration(states, reward, transition_prob, discount=0.9)

    # Task 3
    print("Task 3:")
    states, reward, transition_prob = get_data_task_3()
    value_iteration(states, reward, transition_prob, discount=0.9)

    # Task 4
    print("Task 4:")
    states, reward, transition_prob = get_data_task_4_and_5()
    value_iteration(states, reward, transition_prob, discount=0.9)

    # Task 4
    print("Task 4:")
    states, reward, transition_prob = get_data_task_4_and_5()
    value_iteration(states, reward, transition_prob, discount=0.9)

    # Optimal state-action values (Q) for modified service dog problem
    print("State-action (Q) values for modified service dog problem")
    states, reward, transition_prob = get_data_task_3()
    Q_value_iteration(states, reward, transition_prob, discount=0.4)


def policy_evaluation_tasks():
    from service_dog_problem_model_based import policy_evaluation, get_data_textbook, get_policy_random
    
    # Default
    print("Default:")
    states, reward, transition_prob = get_data_textbook()
    policy = get_policy_random()
    policy_evaluation(states, policy, reward, transition_prob, discount=0.9)

    # Task 1
    print("Task 1:")
    states, reward, transition_prob = get_data_textbook()
    policy = get_policy_random()
    policy_evaluation(states, policy, reward, transition_prob, discount=0.2)

    # Task 2
    print("Task 2:")
    states, reward, transition_prob = get_data_task_2()
    policy = get_policy_random()
    policy_evaluation(states, policy, reward, transition_prob, discount=0.9)

    # Task 3
    print("Task 3:")
    states, reward, transition_prob = get_data_task_3()
    policy = get_policy_random_task_3_and_4()
    policy_evaluation(states, policy, reward, transition_prob, discount=0.9)

    # Task 4
    print("Task 4:")
    states, reward, transition_prob = get_data_task_4_and_5()
    policy = get_policy_random_task_3_and_4()
    policy_evaluation(states, policy, reward, transition_prob, discount=0.9)

    # Task 5
    print("Task 5:")
    states, reward, transition_prob = get_data_task_4_and_5()
    policy = get_policy_task_5()
    policy_evaluation(states, policy, reward, transition_prob, discount=0.9)


if __name__ == "__main__":

    policy_evaluation_tasks()

    value_iteration_tasks()

