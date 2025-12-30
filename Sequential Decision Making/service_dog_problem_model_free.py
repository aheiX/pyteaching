from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import statistics
import math
from service_dog_problem_model_free_data import get_data_textbook


class ServiceDogEnvGym(gym.Env):

    def __init__(self):
        super().__init__()

        # Define what the agent can observe (state space)
        self.observation_space = gym.spaces.Discrete(5)

        # Map observation numbers to actual observation names
        # This makes the code more readable than using raw numbers
        self._observation_to_room = {
            0: "Room 1", 
            1: "Room 2",
            2: "Room 3",
            3: "Outside",
            4: "Found item"
        }
        # Create reverse mapping
        self._room_to_observation = {v: k for k, v in self._observation_to_room.items()}

        # Define what actions are available (action space)
        self.action_space = gym.spaces.Discrete(6)

        # Map action numbers to actual movements
        # This makes the code more readable than using raw numbers
        self._action_to_direction = {
            0: "Go to room 1", 
            1: "Go to room 2",
            2: "Go to room 3",
            3: "Go outside",
            4: "Go inside",
            5: "Search"
        }
        # Create reverse mapping
        self._direction_to_action = {v: k for k, v in self._action_to_direction.items()}
    
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            Observation with dog position
        """

        return self._dog_location

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Observation with dog position
        """

        return {"dog location": self._dog_location}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Args:
            seed: Random seed for reproducible episodes
            options: Additional configuration (unused in this example)

        Returns:
            tuple: (observation, info) for the initial state
        """
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the dog in one of the four locations
        self._dog_location = self.np_random.integers(0, 3, dtype=int)
        # self._dog_location = self._room_to_observation["Room 1"]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """The step() method contains the core environment logic. It takes an action, updates the environment state, and returns the results. 
           This is where the physics, game rules, and reward logic live.
           The step() method execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Map the discrete action to a movement direction
        direction = self._action_to_direction[action]
        dog_location = self._observation_to_room[self._dog_location]

        # Update dog's position for legal actions. For illegal actions, the dog remains at its current position
        reward = 0
        if dog_location == "Room 1":
            if direction == "Go to room 2":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Room 2"]
                reward = -1
        elif dog_location == "Room 2":
            if direction == "Go to room 1":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Room 1"]
                reward = -2     
            elif direction == "Go to room 3":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Room 3"]
                reward = -1     
            elif direction == "Go outside":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Outside"]
                reward = 0   
        elif dog_location == "Room 3":
            if direction == "Go to room 2":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Room 2"]
                reward = -2   
            elif direction == "Search":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Found item"]
                reward = 10
        elif dog_location == "Outside":
            if direction == "Go inside":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Room 2"]
                reward = 0     
            elif direction == "Go outside":
                # update dog's location & reward
                self._dog_location = self._room_to_observation["Room 3"]
                reward = -1

        # Check if agent reached the target        
        terminated = True if self._observation_to_room[self._dog_location] == "Found item" else False

        # We don't use truncation in this simple environment
        # (could add a step limit here if desired)
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def check(self):
        # This will catch many common issues
        try:
            check_env(self)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")


class ServiceDogEnv():

    def __init__(self, legal_actions, reward, transition_prob, seed=None):

        # Initialize environment's random generator
        self.rnd_generator = np.random.default_rng(seed=seed)

        # Legal actions [𝒜(𝑠)]: Set of legal actions for a state 𝑠     
        self.legal_actions = legal_actions

        # Reward [R(s,a)]: Reward of taking action 𝑎 in state 𝑠 
        self.reward = reward

        # Transition probability [P(s'|s,a)]: The transition probability from the current state 𝑠 to its successor state 𝑠′ 
        # depends on the current state 𝑠 and the action 𝑎 chosen by the agent. There may be multiple successor states.
        self.transition_prob = transition_prob

    def _get_obs(self):
        """Convert internal envirnment state to observation format.

        Returns:
            Observation with dog position
        """

        return self._dog_location
    
    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Some information
        """

        return self._dog_location
  
    def reset(self):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial period
        """
       
        # Initial location of the dog
        self._dog_location = "Room 1"

        # Initialize step counter
        self._step_counter = 0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """The step() method contains the core environment logic. 

        Args:
            action: The action to take

        Returns:
            tuple: (obs, reward, terminated, truncated, info)
        """

        # Get current observation
        obs = self._get_obs()

        # Compute reward
        reward = self.reward[(obs, action)]

        # Update dog's position
        _transition_prob = self.transition_prob[(obs, action)]

        self._dog_location = self.rnd_generator.choice(a=list(_transition_prob.keys()), p=list(_transition_prob.values()))

        # Update step counter
        self._step_counter += 1

        # Check if agent reached the target        
        terminated = True if self._dog_location == "Found item" else False

        # Check if episode reached step limit
        truncated = True if self._step_counter >= 500 else False

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info
  
    def _get_legal_actions(self):
        """This method contains returns a list of legal actions for the current observation. 

        Args:
            action: The action to take

        Returns:
            List of legal actions
        """
        
        return self.legal_actions[self._get_obs()]

class PolicyRandom():
    
    def __init__(self, seed=None):
        """Initialize a random policy by explicitly defining the state-action probabilities

        Args:
            seed: An integer number to set the policy's random generator for reproducible results. Default: None will randomly set a seed. 

        """        
        # Initialize policy's random generator
        self.rnd_generator = np.random.default_rng(seed=seed)


        # Define random policy [𝜋(a|s)]: The probability to take action 𝑎 in the current state 𝑠 under policy 𝜋
        self.policy = {
            "Room 1": {"Go to room 2": 1.0},
            "Room 2": {"Go to room 1": 1/3, "Go to room 3": 1/3, "Go outside": 1/3},
            "Room 3": {"Go to room 2": 0.5, "Search": 0.5},
            "Outside": {"Go inside": 0.5, "Go outside": 0.5},
            "Found item": {}
        }
    
    def _get_action(self, env):
        """Returns an action based on the policy

        Args:
            env: The environment action is taken in

        """

        # Get current state from environment 
        state = env._get_obs()

        # Select the state's policy
        state_policy = self.policy[state]

        # Select action based on state policy's probabilities
        action = self.rnd_generator.choice(a=list(state_policy.keys()), p=list(state_policy.values()))
        
        # Print state -> action for debugging
        # print(f"{state} -> {action}")

        return action


class PolicyTD():

    def __init__(self):
        self.Q = dict()

        # Optimal Q-Values (see Fig. 3.5 in "The Art of Reinforcement Learning" by Michael Hu)
        self.Q = {
            "Room 1": {"Go to room 2": 6.2},
            "Room 2": {"Go to room 1": 3.6, "Go to room 3": 8.0, "Go outside": 6.5},
            "Room 3": {"Go to room 2": 5.2, "Search": 10},
            "Outside": {"Go outside": 4.8, "Go inside": 7.2}
        }

    def _get_action(self, env, epsilon=0):
        """Returns an action based on the policy

        Args:
            env: The environment action is taken in

        """
        legal_actions = env._get_legal_actions()
        state = env._get_obs()

        # print(f"state: {state}, legal action: {legal_actions}")

        if epsilon > 0 and np.random.rand() < epsilon:
            action = np.random.choice(legal_actions)
        else:
            
            if state not in self.Q:
                self.Q[state] = {}

            for a in legal_actions:
                if a not in self.Q[state]:
                    self.Q[state][a] = 0

            action = max(self.Q[state], key=self.Q[state].get)
        
        return str(action)

    def _learn(self, env, discount, epsilon, learning_rate, num_updates, on_policy=True):
        """Q-learning off-policy algorithm.

        Args:
            env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
            discount: discount factor, must be 0 <= discount <= 1.
            epsilon: exploration rate for the e-greedy policy, must be 0 <= epsilon < 1.
            learning_rate: the learning rate when update step size
            num_updates: number of updates to the value function.

        Returns:
            policy: the optimal policy based on the estimated (possible optimal) after run the search for num_updates.
            Q: the estimated (possible optimal) state-action value function.
        """

        assert 0.0 <= discount <= 1.0
        assert 0.0 <= epsilon <= 1.0
        assert isinstance(num_updates, int)

        # Initialize state-action value function
        print(f"Optimal Q: {self.Q}")
        self.Q = dict()

        i = 0

        state, info = env.reset()
        while i < num_updates:
            # print(self.Q)
            # Sample an action for state when following the e-greedy policy.
            action = self._get_action(env, epsilon)

            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, terminated, truncated, info = env.step(action)

            # Learning        
            delta = reward

            if not terminated and not truncated:
                # TP 1 action: On-policy = SARSA, Off-Policy = Q-Learning
                action_tp1 = self._get_action(env) if on_policy else self._get_action(env, epsilon=0)

                # Learning (second part, downstream reward)
                self.Q[state_tp1] = self.Q.get(state_tp1, {})
                self.Q[state_tp1][action_tp1] = self.Q[state_tp1].get(action_tp1, 0)
                delta += discount * self.Q[state_tp1][action_tp1]

            
            self.Q[state] = self.Q.get(state, {})
            self.Q[state][action] = self.Q[state].get(action, 0) + learning_rate * (delta - self.Q[state].get(action, 0))

            if terminated or truncated:
                state, info = env.reset()

            i += 1

        for s in self.Q:
            for a in self.Q[s]:
                self.Q[s][a] = round(self.Q[s][a], 4)

        print(self.Q)


def compute_returns(rewards, discount):
    """Compute returns for every time step in the episode trajectory.

    Args:
        rewards: a list of rewards from an episode.
        discount: discount factor, must be 0 <= discount <= 1.

    Returns:
        returns: return for every single time step in the episode trajectory.
    """
    assert 0.0 <= discount <= 1.0

    returns = []
    G_t = 0
    # We do it backwards so it's more efficient and easier to implement.
    for t in reversed(range(len(rewards))):
        G_t = rewards[t] + discount * G_t
        returns.append(G_t)
    returns.reverse()

    return returns


def mc_policy_evaluation(env, policy, discount, num_episodes, first_visit=True):
    """Run Monte Carlo policy evaluation for state value function.

    Args:
        env: a reinforcement learning environment, must have get_states(), reset(), and step() methods.
        policy: the policy that we want to evaluate.
        discount: discount factor, must be 0 <= discount <= 1.
        num_episodes: number of episodes to run.
        first_visit: use first-visit MC, default on.

    Returns:
        V: the estimated state value function for the input policy after run evaluation for num_episodes.
    """
    assert 0.0 <= discount <= 1.0
    assert isinstance(num_episodes, int)

    # Initialize
    N = dict()  # counter for visits number
    V = dict()  # state value function
    G = dict()  # total returns

    for _ in range(num_episodes):
        # Sample an episode trajectory using the given policy.
        episode = []
        state, info = env.reset()
        while True:
            # Get action when following the policy.
            action = policy._get_action(env)

            # Take the action in the environment and observe successor state and reward.
            state_tp1, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = state_tp1
            if terminated or truncated:
                # if truncated:
                #     print("truncated")
                break

        # Unpack list of tuples into separate lists.
        # print(episode)
        states, _, rewards = map(list, zip(*episode))

        # Compute returns for every time step in the episode.
        returns = compute_returns(rewards, discount)

        # Loop over all state in the episode.
        for t, state in enumerate(states):
            G_t = returns[t]
            # Check if this is the first time state visited in the episode.
            if first_visit and state in states[:t]:
                continue

            N[state] = N.get(state, 0) + 1
            G[state] = G.get(state, 0) + G_t
            V[state] = G[state] / N[state]

    # Round state value function
    for state in V:
        V[state] = round(V[state], 2)

    print(f"State value function after {num_episodes} iterations: {V}")

if __name__ == "__main__":

    legal_actions, reward, transition_prob = get_data_textbook()

    env = ServiceDogEnv(legal_actions, reward, transition_prob)
    policy = PolicyRandom(seed=2506)
    # policy = PolicyTD()
    # policy._learn(env, discount=0.9, epsilon=0.3, learning_rate=0.01, num_updates=10000, on_policy=False)

    # mc_policy_evaluation(env, policy, discount=0.9, num_episodes=2000, first_visit=True)

    # Reset environment
    env.reset()

    # Print information
    print(f"Observation: {env._get_obs()}, info: {env._get_info()}")
    
    # Select action
    action = policy._get_action(env)
    
    # Transition via dynamic model/function
    env.step(action)

    # Print information
    print(f"Observation: {env._get_obs()}, info: {env._get_info()}")

    # policy.test(env=env, num_episodes=10000)

    # # Create model
    # # Reset environment to start a new episode
    # env = ServiceDogEnvGym()

    # # This will catch many common issues
    # # try:
    # #     check_env(env)
    # #     print("Environment passes all checks!")
    # # except Exception as e:
    # #     print(f"Environment has issues: {e}")
    # #     # Reset environment to start a new episode
        
    # observation, info = env.reset()
    # # observation: what the agent can "see" - cart position, velocity, pole angle, etc.
    # # info: extra debugging information (usually not needed for basic learning)
    
    # dog_sequence = [observation]

    # episode_over = False
    # total_reward = 0

    # while not episode_over:

    #     # Choose an action: 0 = push cart left, 1 = push cart right
    #     action = env.action_space.sample()  # Random action for now - real agents will be smarter!

    #     # Take the action and see what happens
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     dog_sequence.append(observation)

    #     # reward: +1 for each step the pole stays upright
    #     # terminated: True if pole falls too far (agent failed)
    #     # truncated: True if we hit the time limit (500 steps)

    #     total_reward += reward
    #     episode_over = terminated or truncated

    # print(f"Episode finished! Total reward: {total_reward} after {steps} steps.")
    # env.close()