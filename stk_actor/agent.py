from typing import Dict, List, Tuple, Union, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import gymnasium as gym
from gymnasium import spaces

def get_device(device: Union[torch.device, str] = "auto") -> torch.device:
    if device == "auto":
        device = "cuda"
    device = torch.device(device)
    if device.type == torch.device("cuda").type and not torch.cuda.is_available():
        return torch.device("cpu")
    return device

class BaseFeaturesExtractor(nn.Module):
    def __init__(self, observation_space: gym.Space, features_dim: int = 0) -> None:
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim
    @property
    def features_dim(self) -> int:
        return self._features_dim

def get_flattened_obs_dim(observation_space: spaces.Space) -> int:
    if isinstance(observation_space, spaces.MultiDiscrete):
        return sum(observation_space.nvec)
    else:
        return spaces.utils.flatdim(observation_space)

class FlattenExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space) -> None:
        super().__init__(observation_space, get_flattened_obs_dim(observation_space))
        self.flatten = nn.Flatten()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.flatten(observations)
    
class MlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[torch.device, str] = "auto",
    ) -> None:
        super().__init__()
        # device = torch.get_device(device)
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_pi = feature_dim
        last_layer_dim_vf = feature_dim

        if isinstance(net_arch, dict):
            pi_layers_dims = net_arch.get("pi", []) 
            vf_layers_dims = net_arch.get("vf", []) 
        else:
            pi_layers_dims = vf_layers_dims = net_arch
        for curr_layer_dim in pi_layers_dims:
            policy_net.append(nn.Linear(last_layer_dim_pi, curr_layer_dim))
            policy_net.append(activation_fn())
            last_layer_dim_pi = curr_layer_dim
        for curr_layer_dim in vf_layers_dims:
            value_net.append(nn.Linear(last_layer_dim_vf, curr_layer_dim))
            value_net.append(activation_fn())
            last_layer_dim_vf = curr_layer_dim

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.policy_net = nn.Sequential(*policy_net)#.to(device)
        self.value_net = nn.Sequential(*value_net)#.to(device)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)

    
class Policy(nn.Module):
    def __init__(self, observation_space, action_dims, net_arch, activation_fn,):
        super().__init__()
        self.features_extractor = FlattenExtractor(observation_space)
        self.pi_features_extractor = self.features_extractor
        self.vf_features_extractor = self.features_extractor
        self.mlp_extractor = MlpExtractor(
            self.features_extractor.features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
        )
        self.action_net = nn.Linear(net_arch[-1], sum(action_dims))
        self.value_net = nn.Linear(net_arch[-1], 1)


class UnifiedSACPolicy(nn.Module):
    def __init__(self, observation_space, action_dims, net_arch, activation_fn):
        super().__init__()
        
        self.shared = Policy(
            observation_space,
            action_dims,
            net_arch=net_arch,
            activation_fn=activation_fn
        )
        self.action_dims = action_dims
    
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