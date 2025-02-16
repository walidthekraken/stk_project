
import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import torch.functional as F
import inspect
from .replay_buffer import SACRolloutBuffer

class ObsTimeExtensionWrapper(gym.Wrapper):
    def __init__(self, env):
        """
        Initializes the wrapper to extend the observation with a memory of the past observation.
        
        :param env: The inner environment to wrap.
        """
        super(ObsTimeExtensionWrapper, self).__init__(env)
        # Initialize memory with a null observation
        self.prev_obs = np.zeros(env.observation_space.shape)
        self.prev_prev_obs = np.zeros(env.observation_space.shape)
        # Double the observation space
        self.observation_space = self._extend_observation_space(self.env.observation_space)

    def _extend_observation_space(self, observation_space):
        """
        Extends the observation space to accommodate the current and previous observations.
        
        :param observation_space: The original observation space.
        :return: The extended observation space.
        """
        shape = (3 * observation_space.shape[0],)
        return gym.spaces.Box(
            low=np.concatenate([observation_space.low, observation_space.low, observation_space.low]),
            high=np.concatenate([observation_space.high, observation_space.high, observation_space.high]),
            shape=shape,
            dtype=observation_space.dtype
        )

    def _extend_observation(self, current_obs):
        """
        Concatenates the current observation with the previous observation.
        
        :param current_obs: The current observation.
        :return: The concatenated observation.
        """
        extended_obs = np.concatenate([self.prev_prev_obs, self.prev_obs, current_obs])
        # Update the previous observation with the current one
        self.prev_prev_obs = self.prev_obs
        self.prev_obs = current_obs
        return extended_obs

    def reset(self, **kwargs):
        """
        Resets the environment and reinitializes the observation memory.
        
        :param kwargs: Additional arguments for the reset method.
        :return: The extended initial observation (null + current observation).
        """
        # Reset the environment and get the current observation
        current_obs, info = self.env.reset(**kwargs)
        # Reset the memory (null observation)
        self.prev_obs = np.zeros_like(current_obs)
        self.prev_prev_obs = np.zeros_like(current_obs)
        return self._extend_observation(current_obs), info

    def step(self, action):
        """
        Takes a step in the environment using the provided action.
        Extends the observation by concatenating the previous and current observations.
        
        :param action: The action to take.
        :return: A tuple (observation, reward, done, info) with the extended observation.
        """
        # Take the step in the environment
        current_obs, reward, terminated, truncated, info, *_ = self.env.step(action)
        # Return the extended observation (previous + current), reward, done, and info
        return self._extend_observation(current_obs), reward, terminated, truncated, info

class StuckStopWrapper(gym.Wrapper):
    def __init__(self, env, n=16):
        """
        :param env: The inner environment to wrap.
        :param n: Stuck timesteps to tolerate.
        """
        super(StuckStopWrapper, self).__init__(env)
        # Initialize memory with a null observation
        self.rewards = [0 for _ in range(n)]

    def step(self, action):
        """
        Takes a step in the environment using the provided action.
        Extends the observation by concatenating the previous and current observations.
        
        :param action: The action to take.
        :return: A tuple (observation, reward, done, info) with the extended observation.
        """
        # Take the step in the environment
        current_obs, reward, terminated, truncated, info, *_ = self.env.step(action)
        self.rewards.pop(0)
        self.rewards.append(reward)
        # print([round(x,1) for x in self.rewards])
        if all([round(x,1)==-0.1 for x in self.rewards]):
            truncated = True
        return current_obs, reward, terminated, truncated, info

class PreprocessObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, ret_dict=True, norm=True, agent_name='normed_a2c_num5_best' ):
        """
        A Gym wrapper to preprocess mixed observation space (continuous + discrete)
        into a flat tensor.
        
        Args:
            env: The Gym environment to wrap.
        """
        super().__init__(env)
        self.observation_space = self._get_flat_observation_space(env.observation_space)
        self.ret_dict = ret_dict
        if norm is True:
            mod_path = Path(inspect.getfile(SACRolloutBuffer)).parent
            self.mean = torch.load(mod_path/f'trained_agents/{agent_name}/buffer_mean', map_location='cpu')
            self.std = torch.load(mod_path/f'trained_agents/{agent_name}/buffer_std', map_location='cpu')
        self.norm = norm

    def _get_flat_observation_space(self, observation_space):
        """
        Create a flat observation space based on the original observation space.
        
        Args:
            observation_space: Original observation space with 'continuous' and 'discrete' components.
        
        Returns:
            A flattened observation space.
        """
        continuous_dim = observation_space['continuous'].shape[0]
        discrete_dims = sum(space.n for space in observation_space['discrete'])
        flat_dim = continuous_dim + discrete_dims
        return gym.spaces.Box(low=-float('inf'), high=float('inf'), shape=(flat_dim,), dtype=float)

    def observation(self, obs):
        """
        Process the observation into a flat numpy array.
        
        Args:
            obs: The raw observation from the environment.
        
        Returns:
            A preprocessed flat numpy array.
        """
        continuous_obs, discrete_obs = obs['continuous'], obs['discrete']
        continuous_array = np.array(continuous_obs, dtype=np.float32)
        
        discrete_arrays = [
            np.eye(num_classes.n, dtype=np.float32)[x]
            for x, num_classes in zip(discrete_obs, self.env.observation_space['discrete'])
        ]
        
        flat_array = np.concatenate([continuous_array] + discrete_arrays)
        flat_array = torch.tensor(flat_array)
        normed_array = flat_array
        if self.norm:
            normed_array = (flat_array - self.mean) / (self.std + 1e-8)

        if self.ret_dict:
            # This is done because of a bug where the evaluation expects a dict observation and not a vector
            return {
                'normed_obs':normed_array.numpy(),
                'obs':flat_array.numpy(),
                'normed':1 if self.norm else 0
            }
        return normed_array.numpy()

class SkipFirstNStepsWrapper(gym.Wrapper):
    def __init__(self, env, n):
        super().__init__(env)
        self.n = n

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        steps_skipped = 0
        
        while steps_skipped < self.n:
            obs, _, done, truncated, info = self.env.step(self.env.action_space.sample())
            steps_skipped += 1
            
            if done or truncated:
                obs, info = self.env.reset(**kwargs)
                steps_skipped = 0  # Restart skipping process if reset happens

        return obs, info
