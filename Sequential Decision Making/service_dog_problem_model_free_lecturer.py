
# Optimal Q-Values (see Fig. 3.5 in "The Art of Reinforcement Learning" by Michael Hu)
# self.Q = {
#     "Room 1": {"Go to room 2": 6.2},
#     "Room 2": {"Go to room 1": 3.6, "Go to room 3": 8.0, "Go outside": 6.5},
#     "Room 3": {"Go to room 2": 5.2, "Search": 10},
#     "Outside": {"Go outside": 4.8, "Go inside": 7.2}
# }

def get_data_task_2():
    """
    Creates the model-free data for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        legal_actions: Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠 BASED ON REWARD FUNCTION
    """
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

    # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠
    legal_actions = {
        "Room 1": ["Go to room 2"],
        "Room 2": ["Go to room 1", "Go to room 3", "Go outside"],
        "Room 3": ["Go to room 2", "Search"],
        "Outside": ["Go inside", "Go outside"],
        "Found item": []
    }

    return legal_actions, reward, transition_prob


def get_data_task_3():
    """
    Creates the model-free data for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        legal_actions: Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠 BASED ON REWARD FUNCTION
    """
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

    # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠
    legal_actions = {
        "Room 1": ["Go to room 1", "Go to room 2", "Go to room 4"],
        "Room 2": ["Go to room 3", "Go outside"],
        "Room 3": ["Go outside", "Search"],
        "Room 4": ["Search"],
        "Outside": ["Go inside"],
        "Found item": []
    }

    return legal_actions, reward, transition_prob


def get_data_task_4_and_5():
    """
    Creates the model-free data for the service dog example used in "The Art of Reinforcement Learning" by Michael Hu

    Returns:
        reward: Reward of taking action 𝑎 in state 𝑠 
        transition_prob: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        legal_actions: Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠 BASED ON REWARD FUNCTION
    """
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

    # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠
    legal_actions = {
        "Room 1": ["Go to room 1", "Go to room 2", "Go to room 4"],
        "Room 2": ["Go to room 3", "Go outside"],
        "Room 3": ["Go outside", "Search"],
        "Room 4": ["Search"],
        "Outside": ["Go inside"],
        "Found item": []
    }

    return legal_actions, reward, transition_prob


class PolicyTask5():
    
    def __init__(self):       
        pass
    
    def _get_action(self, env):
        """Returns an action based on a policy specified in the lecture script. 

        Args:
            env: The environment action is taken in

        Returns:
            A legal action 
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

        # Get current state from environment 
        state = env._get_obs()

        # Select the state's policy
        state_policy = policy[state]       

        # Select action based on state policy's probabilities
        action = str(env.np_random.choice(a=list(state_policy.keys()), 
                                          p=list(state_policy.values())))

        return action


def policy_evaluation_tasks():
    from service_dog_problem_model_free import get_data_textbook, EnvServiceDog, PolicyRandom, mc_policy_evaluation

    # Task 1
    legal_actions, reward, transition_prob = get_data_textbook()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    policy = PolicyRandom()
    mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)

    # Task 2
    legal_actions, reward, transition_prob = get_data_task_2()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    policy = PolicyRandom()
    mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)

    # Task 3
    legal_actions, reward, transition_prob = get_data_task_3()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    policy = PolicyRandom()
    mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)

    # Task 4
    legal_actions, reward, transition_prob = get_data_task_4_and_5()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    policy = PolicyRandom()
    mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)

    # Task 5
    legal_actions, reward, transition_prob = get_data_task_4_and_5()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    policy = PolicyTask5()
    mc_policy_evaluation(env, policy, discount=0.9, num_episodes=1000, first_visit=True, seed=2506)


def step_size_epsilon_tasks():
    # Step size and epsilon tasks
    from service_dog_problem_model_free import get_data_textbook, EnvServiceDog, PolicySARSA, PolicyQLearning

    legal_actions, reward, transition_prob = get_data_textbook()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    
    print("SARSA")
    num_updates = 100000
    step_size_epsilon_tasks_helper(env=env, policy=PolicySARSA(), num_updates=num_updates)
    
    print("Q-Learning")
    num_updates = 20000
    # step_size_epsilon_tasks_helper(env=env, policy=PolicyQLearning(), num_updates=num_updates)


def step_size_epsilon_tasks_helper(env, policy, num_updates=100000):
    import pandas as pd

    # Task "step size"
    # History
    data = {
        "learning_rate": [],
        "R1_GoR2": [],
        "R2_GoR1": [],
        "R2_GoR3": [],
        "R2_GoOut": [],
        "R3_GoR2": [],
        "R3_Search": [],
        "Out_GoIn": [],
        "Out_GoOut": []
    }

    for learning_rate in [0.1,0.3,0.5,0.7,1.0]:

        policy._learn(env=env, discount=0.9, epsilon=0.4, learning_rate=learning_rate, num_updates=num_updates, seed=2506)

        # Round Q values and print results
        for s in policy.Q:
            for a in policy.Q[s]:
                policy.Q[s][a] = round(policy.Q[s][a], 2)

        data["learning_rate"].append(learning_rate)
        data["R1_GoR2"].append(policy.Q["Room 1"]["Go to room 2"])
        data["R2_GoR1"].append(policy.Q["Room 2"]["Go to room 1"])
        data["R2_GoR3"].append(policy.Q["Room 2"]["Go to room 3"])
        data["R2_GoOut"].append(policy.Q["Room 2"]["Go outside"])
        data["R3_GoR2"].append(policy.Q["Room 3"]["Go to room 2"])
        data["R3_Search"].append(policy.Q["Room 3"]["Search"])
        data["Out_GoIn"].append(policy.Q["Outside"]["Go inside"])
        data["Out_GoOut"].append(policy.Q["Outside"]["Go outside"])
    
    df = pd.DataFrame(data)
    print(df)

    # Task "epsilon" (exploration vs. exploitation)    
    # History
    data = {
        "eps": [],
        "R1_GoR2": [],
        "R2_GoR1": [],
        "R2_GoR3": [],
        "R2_GoOut": [],
        "R3_GoR2": [],
        "R3_Search": [],
        "Out_GoIn": [],
        "Out_GoOut": []
    }

    for epsilon in [0.1,0.3,0.5,0.7,1.0]:
    
        policy._learn(env=env, discount=0.9, epsilon=epsilon, learning_rate=0.01, num_updates=num_updates, seed=2506)

        # Round Q values and print results
        for s in policy.Q:
            for a in policy.Q[s]:
                policy.Q[s][a] = round(policy.Q[s][a], 2)

        data["eps"].append(epsilon)
        data["R1_GoR2"].append(policy.Q["Room 1"]["Go to room 2"])
        data["R2_GoR1"].append(policy.Q["Room 2"]["Go to room 1"])
        data["R2_GoR3"].append(policy.Q["Room 2"]["Go to room 3"])
        data["R2_GoOut"].append(policy.Q["Room 2"]["Go outside"])
        data["R3_GoR2"].append(policy.Q["Room 3"]["Go to room 2"])
        data["R3_Search"].append(policy.Q["Room 3"]["Search"])
        data["Out_GoIn"].append(policy.Q["Outside"]["Go inside"])
        data["Out_GoOut"].append(policy.Q["Outside"]["Go outside"])
    
    df = pd.DataFrame(data)
    print(df)


def modified_service_dog_problem():
    from service_dog_problem_model_free import get_data_textbook, EnvServiceDog, PolicyQLearning
    
    
    print("Q-Learning for modified service dog problem")

    legal_actions, reward, transition_prob = get_data_task_3()
    env = EnvServiceDog(legal_actions, reward, transition_prob)
    policy = PolicyQLearning()
    policy._learn(env=env, discount=0.4, epsilon=1, learning_rate=0.1, num_updates=1000, seed=2506)
    print(policy.Q)


if __name__ == "__main__":

    # Policy evaluation tasks
    # policy_evaluation_tasks()

    # Step size and epsilon tasks
    step_size_epsilon_tasks()

    # Q-Learning for modified service dog problem
    # modified_service_dog_problem()
    
