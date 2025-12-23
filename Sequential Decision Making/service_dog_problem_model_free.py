from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env
import random
import statistics
import math


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

    def __init__(self):
        # Define what the agent can observe (state space)
        self.observation_space = ["Room 1", "Room 2", "Room 3", "Outside", "Found item"]

        # Define what actions are available (action space)
        self.action_space = ["Go to room 1", "Go to room 2", "Go to room 3", "Go outside", "Go inside", "Search"]
        
    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            Observation is the dog's position
        """

        return self._dog_location

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Observation with dog position
        """

        return {"dog location": self._dog_location, "step": self._step_counter}

    def _get_legal_actions(self, obs=None):
        
        if obs is None:
            obs = self._get_obs()

        if obs == "Room 1":
            legal_actions = ["Go to room 2"]
        elif obs == "Room 2":
            legal_actions = ["Go to room 1", "Go to room 3", "Go outside"]
        elif obs == "Room 3":
            legal_actions = ["Go to room 2", "Search"]
        elif obs == "Outside":
            legal_actions = ["Go inside", "Go outside"]
        else:
            legal_actions = []

        # legal_actions = ["Go to room 1", "Go to room 2", "Go to room 3", "Go outside", "Go inside", "Search"]
        return legal_actions

    def reset(self):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial state
        """
       
        # Initial location of the dog
        self._dog_location = "Room 1"

        # Initialize step counter
        self._step_counter = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """The step() method contains the core environment logic. 

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Update dog's position for legal actions and reward. For illegal actions, the dog remains at its current position.
        reward = 0
        if self._dog_location == "Room 1":
            if action == "Go to room 2":
                # update dog's location & reward
                self._dog_location = "Room 2"
                reward = -1
        elif self._dog_location == "Room 2":
            if action == "Go to room 1":
                # update dog's location & reward
                self._dog_location = "Room 1"
                reward = -2     
            elif action == "Go to room 3":
                # update dog's location & reward
                self._dog_location = "Room 3"
                reward = -1     
            elif action == "Go outside":
                # update dog's location & reward
                self._dog_location = "Outside"
                reward = 0   
        elif self._dog_location == "Room 3":
            if action == "Go to room 2":
                # update dog's location & reward
                self._dog_location = "Room 2"
                reward = -2   
            elif action == "Search":
                # update dog's location & reward
                self._dog_location = "Found item"
                reward = 10
        elif self._dog_location == "Outside":
            if action == "Go inside":
                # update dog's location & reward
                self._dog_location = "Room 2"
                reward = 0     
            elif action == "Go outside":
                # update dog's location & reward
                self._dog_location = "Room 3"
                reward = -1
      
        # Discount reward
        reward = reward * math.pow(0.9, self._step_counter)

        # Update step counter
        self._step_counter += 1

        # Check if agent reached the target        
        terminated = True if self._dog_location == "Found item" else False

        # Check if episode reached step limit
        truncated = True if self._step_counter >= 500 else False

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
                  

class ServiceDogPolicyRandom():
    
    def __init__(self):
        """Initialize a policy by defining all parameters and variables (e.g., Q-table, learning rate)

        """        
        pass
    
    def _get_action(self, env):
        """Returns an action based on the policy

        Args:
            env: The environment action is taken in

        """
        legal_actions = env._get_legal_actions()
        action = random.choice(legal_actions)
        print(action)
        # print(env._get_obs(), legal_actions)
        
        return action
    

    def test(self, env, num_episodes=1):
        """ A method to test the policy 

        Args:
            env: The environment

        """
        total_reward_history = []

        for episode in range(num_episodes):

            observation, info = env.reset()

            episode_over = False
            total_reward = 0
            obs_sequence = []

            while not episode_over:

                # Choose an action
                action = self._get_action(env)

                # Take the action and see what happens (transition)
                observation, reward, terminated, truncated, info = env.step(action)

                obs_sequence.append(observation)
                total_reward += reward
                episode_over = terminated or truncated

            total_reward_history.append(total_reward)
            # print(f"Episode {episode} finished: Total reward: {total_reward} after {len(obs_sequence)} steps.")
        
        # Print descriptive statistics
        print(f"Rewards ({len(total_reward_history)} episodes): avg={round(statistics.mean(total_reward_history), 2)}, std. dev.={round(statistics.stdev(total_reward_history), 2)}, min={round(min(total_reward_history), 2)}, max={round(max(total_reward_history), 2)}")
        # print(total_reward_history)


def policy_evaluation(discount=0.9, delta_threshold=0.001):
    """
    Given a policy, and state value function, using dynamic programming to
    estimate the state-value function for this policy.

    Args:
        env: a MDP environment.
        policy: policy we want to evaluate.
        V: state value function for the input policy.
        discount: discount factor, must be 0 <= discount <= 1.
        delta_threshold: the threshold determining the accuracy of the estimation.

    Returns:
        estimated state value function for the input policy.

    """

    count = 0
    states = ["Room 1", "Room 2", "Room 3", "Outside", "Found item"]
    legal_actions = {
        "Room 1": ["Go to room 2"],
        "Room 2": ["Go to room 1", "Go to room 3", "Go outside"],
        "Room 3": ["Go to room 2", "Search"],
        "Outside": ["Go inside", "Go outside"],
        "Found item": []
    }

    reward = {("Room 1", "Go to room 2"): -1,
              ("Room 2", "Go to room 1"): -2,
              ("Room 2", "Go to room 3"): -1,
              ("Room 2", "Go outside"): 0,
              ("Room 3", "Go to room 2"): -2,
              ("Room 3", "Search"): 10,
              ("Outside", "Go outside"): -2,    # =-1 in textbook
              ("Outside", "Go inside"): 0              
              }
    
    successor_state = {"Go to room 1": "Room 1",
                       "Go to room 2": "Room 2",
                       "Go to room 3": "Room 3",
                       "Go outside": "Outside",
                       "Go inside": "Room 2",
                       "Search": "Found item"}

    # Initialize state value function to be all zeros for all states.
    V = {s: 0 for s in states}

    while True:
        delta = 0
        for state in states:
            old_v = V[state]
            new_v = 0
            for action in legal_actions[state]:  # For every legal action
                g = 0
                pi_prob = 1/len(legal_actions[state])

                g += reward[(state, action)]

                g += discount * V[successor_state[action]]

                # Weight by the probability of selecting this action when following the policy
                new_v += pi_prob * g
            V[state] = new_v
            delta = max(delta, abs(old_v - new_v))

        count += 1
        if delta < delta_threshold:
            break
    print(V)
    return V

if __name__ == "__main__":

    policy_evaluation()

    # env = ServiceDogEnv()
    # policy = ServiceDogPolicyRandom()
    
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