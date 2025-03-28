import gymnasium as gym
from bbrl.agents import Agent
import torch
from .agent import UnifiedSACPolicy
from torch.distributions import Categorical
from pathlib import Path
import inspect

class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action

mod_path = Path(inspect.getfile(UnifiedSACPolicy)).parent

class Actor(Agent):
    """Computes probabilities over action"""
    def __init__(
            self, 
            observation_space, 
            action_space,
            *args, 
            net_arch=[1024,1024,1024], 
            activation_fn=torch.nn.Tanh,
            state_dict_path= None,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        action_dims = [space.n for space in action_space]
        policy = UnifiedSACPolicy(
            observation_space=observation_space,
            action_dims=action_dims,
            net_arch = net_arch,
            activation_fn = activation_fn,
        )
        if state_dict_path is not None:
            print('LOADING',state_dict_path)
            policy.load_state_dict(torch.load(state_dict_path, weights_only=True),)
        else:
            print('NOT Loading Dict')

        self.policy = policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dims = action_dims

    def forward(self, t: int):
        # Computes probabilities over actions
        observation = self.get(("env/env_obs/normed_obs", t))
        normed = self.get(("env/env_obs/normed", t))
        logits = self.policy.forward(observation)
        self.set(("logits", t), logits)
        self.set(("action_dims", t), torch.tensor(self.action_dims))

class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        logits = self.get(("logits", t))
        action_dims = self.get(("action_dims", t))
        split_logits = torch.split(logits, action_dims.tolist(), dim=-1)
        actions = []
                
        for logit in split_logits:
            # distribution = Categorical(logits=logit)
            # action = distribution.sample()
            action = torch.argmax(logit, dim=-1)[0]
            actions.append(action)
        self.set(("action", t), torch.stack(actions).unsqueeze(0))


class SamplingActor(Agent):
    """Samples random actions from probability distributions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        logits = self.get(("logits", t))
        action_dims = self.get(("action_dims", t))
        split_logits = torch.split(logits, action_dims.tolist(), dim=-1)
        actions = []
        for logit in split_logits:
            distribution = Categorical(logits=logit)
            action = distribution.sample()
            actions.append(action)
        self.set(("action", t), torch.stack(actions).unsqueeze(0))

