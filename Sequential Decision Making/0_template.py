import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from typing import Optional

class Environment(gym.Env):

    def __init__(self):
        # Initialize from gymansium environment
        super().__init__()

        # Define what the agent can observe (state space). This variable can be of any type (e.g., list, dict, tuple, etc.)
        self.observation_space = ["Obs 1", "Obs 2"]

        # Define what actions are available (action space). This variable can be of any type (e.g., list, dict, tuple, etc.)
        self.action_space = ["Action 1", "Action 2"]

    def _get_obs(self):
        """Convert internal state to observation format. 
           Obersvation: The information that the agent can observe or perceive.
           State: The is full description of the environment (needed to make a decision).
           In fully observable environment State Space is same as Observation Space. But in partially observable environments, the observation space is a subset of State Space.
           Source: https://medium.com/@walkerastro41/action-space-state-space-observation-space-demystified-6c9c00a355b4

        Returns:
            Observation
        """

        return True

    def _get_info(self):
        """Compute auxiliary information for debugging.

        Returns:
            Whatever needs to be returned
        """

        return {"info": True}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Start a new episode.

        Returns:
            tuple: (observation, info) for the initial period
        """
        # IMPORTANT: Must call this first to seed the random number generator (called via self.np_random)
        super().reset(seed=seed)  

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """The step() method contains the core environment logic. 
           It takes an action, updates the environment state, and returns the results. 
           This is where the physics, game rules, and reward logic live.
           The step() method execute one timestep within the environment.

        Args:
            action: The action to take

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        # Update observation based on the action and the dynamics (e.g., new position and random move of the other player)

        # Update reward
        reward = 0 

        # Check if agent reached the target        
        terminated = False

        # A a step limit here if desired
        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
    
    def check(self):
        """This method catches many common issues with the Gymnasium environment

        """

        try:
            check_env(self)
            print("Environment passes all checks!")
        except Exception as e:
            print(f"Environment has issues: {e}")


class Policy():
    
    def __init__(self):
        """Initialize a policy by defining all parameters and variables (e.g., Q-table, learning rate)

        """
        
        pass
    
    def _get_action(self, env):
        """Returns an action based on the policy

        Args:
            env: The environment action is taken in

        """
        
        return True

    def test(self, env, num_episodes):
        """ A method to test the policy 

        Args:
            env: The environment

        """
        avg_reward_per_episode = []

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

            avg_reward_per_episode.append(total_reward)
            print(f"Episode {episode} finished: Total reward: {total_reward} after {len(obs_sequence)} steps.")
        
        # Print descriptive statistics
