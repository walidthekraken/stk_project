import gymnasium as gym
import torch
import torch.nn.functional as F

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


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
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
    
#policy = torch.load('policy_512_512_512_512_SiLU_3_statedict', map_location='cuda')


from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from pystk2_gymnasium import AgentSpec
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from functools import partial

vec_env = make_vec_env("supertuxkart/flattened_multidiscrete-v0", seed=0, n_envs=21, wrapper_class=lambda x : (PreprocessObservationWrapper(x)), env_kwargs={
    'render_mode':None, 'agent':AgentSpec(use_ai=False, name="walid"), 'track':'minigolf', 'laps':2#'difficulty':0, #'num_kart':2, 'difficulty':0
})

tracks = ['abyss',
 'black_forest',
 'candela_city',
 'cocoa_temple',
 'cornfield_crossing',
 'fortmagma',
 'gran_paradiso_island',
 'hacienda',
 'lighthouse',
 'mines',
 'minigolf',
 'olivermath',
 'ravenbridge_mansion',
 'sandtrack',
 'scotland',
 'snowmountain',
 'snowtuxpeak',
 'stk_enterprise',
 'volcano_island',
 'xr591',
 'zengarden']
for i,venv in enumerate(vec_env.envs):
    print(i, tracks[i%len(tracks)])
    venv.env.default_track = tracks[i%len(tracks)]


net_arch=[512,512,512,512]
activation_fn=torch.nn.SiLU
filename = 'policy_512_512_512_512_SiLU_3_statedict'

action_dims = [space.n for space in vec_env.action_space]
unified_policy = UnifiedSACPolicy(
    vec_env.observation_space, 
    action_dims, 
    net_arch=net_arch, 
    activation_fn=activation_fn
)
unified_policy.load_state_dict(torch.load(filename, map_location='cpu'))

steps = [(
    #16384,
    3000,
    3_000_000,
)]
for n_steps, total_timesteps in steps:
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn,),
        device='cpu',
        learning_rate=0.0001,
        batch_size=128,
        n_steps=n_steps,
        tensorboard_log="./outputs/",
        ent_coef=0.001,
        clip_range=0.001,
    )
    model.policy.load_state_dict(PPO.load("ppti_ppo_1500_batch128_clip0001_ent0001", custom_objects={'policy_kwargs' :  dict(net_arch=net_arch, activation_fn=activation_fn,), }).policy.state_dict())
    print('DOING', n_steps, total_timesteps)
    #model.policy.load_state_dict(policy.shared.state_dict())
    #model.policy.load_state_dict(unified_policy.shared.state_dict())
    model.learn(total_timesteps=total_timesteps, progress_bar=True)#callback=SummaryWriterCallback())
    model.save(f'ppti_ppo2_{n_steps}_batch128_clip0001_ent0001')

#model.save("ppti_ppo_8192") 
