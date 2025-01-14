import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from functools import partial

import gymnasium as gym
import numpy as np

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

class PreprocessObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        """
        A Gym wrapper to preprocess mixed observation space (continuous + discrete)
        into a flat tensor.
        
        Args:
            env: The Gym environment to wrap.
        """
        super().__init__(env)
        self.observation_space = self._get_flat_observation_space(env.observation_space)

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
        Process the observation into a flat tensor.
        
        Args:
            obs: The raw observation from the environment.
        
        Returns:
            A preprocessed flat tensor.
        """
        continuous_obs, discrete_obs = obs['continuous'], obs['discrete']
        continuous_tensor = torch.FloatTensor(continuous_obs)
        
        discrete_tensors = [
            F.one_hot(torch.tensor(x), num_classes=num_classes.n).float()
            for x, num_classes in zip(discrete_obs, self.env.observation_space['discrete'])
        ]
        
        flat_tensor = torch.cat([continuous_tensor] + discrete_tensors)
        return flat_tensor

# make_stkenv = partial(make_env, "ok", wrappers=[], render_mode=None, autoreset=True, agent=AgentSpec(use_ai=False, name="ok"))

# Parallel environments


class UnifiedSACPolicy(nn.Module):
    def __init__(self, policy_stb,):
        super().__init__()
        
        self.shared = policy_stb
    
    def forward(self, x):
        x = self.shared.features_extractor(x)
        x = self.shared.mlp_extractor.policy_net(x)
        x = self.shared.action_net(x)
        return x
    
    def sample(self, x, deterministic=False):
        logits = self.forward(x)
        
        # Split logits for each action dimension
        split_logits = torch.split(logits, self.action_dims, dim=-1)
        
        actions = []
        log_probs = []
        probs = []
        
        for logit in split_logits:
            distribution = Categorical(logits=logit)
            if deterministic:
                action = torch.argmax(logit, dim=-1)
            else:
                action = distribution.sample()
            
            log_prob = distribution.log_prob(action)
            prob = F.softmax(logit, dim=-1)
            
            actions.append(action)
            log_probs.append(log_prob)
            probs.append(prob)
        
        return (
            torch.stack(actions),
            torch.stack(log_probs),
            probs
        )