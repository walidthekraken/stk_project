import gymnasium as gym
from bbrl.agents import Agent
import torch
from .agent import UnifiedSACPolicy


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(Agent):
    """Computes probabilities over action"""
    def __init__(
            self, 
            observation_space, 
            action_space,
            *args, 
            net_arch=[1024,1024,1024,1024], 
            activation_fn=torch.nn.SiLU,
            state_dict_path='policy_1024_1024_1024_1024_SiLU_statedict',
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
        policy.load_state_dict(torch.load(state_dict_path))

        self.policy = policy
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dims = action_dims

    def forward(self, t: int):
        # Computes probabilities over actions
        observation = self.get(("env/env_obs", t))
        logits = self.policy.forward(observation)
        split_logits = torch.split(logits, self.action_dims, dim=-1)
        self.set(("split_logits", t), split_logits)


class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        split_logits = self.get(("split_logits", t))
        actions = []
        for logit in split_logits:
            action = torch.argmax(logit, dim=-1)
            actions.append(action)
        self.set(("action", t), torch.stack(actions))


class SamplingActor(Agent):
    """Samples random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
