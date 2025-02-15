import joblib, torch
import numpy as np
from ..replay_buffer import SACRolloutBuffer
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations, DiscreteActionsWrapper
from pystk2_gymnasium.wrappers import FlattenerWrapper
from ..wrappers import PreprocessObservationWrapper
import gymnasium as gym
from pystk2_gymnasium import AgentSpec

def get_flat_env_stats_for_bhc(state_items = 5, state_karts = 5, state_paths = 5,):

    env = gym.make(
        "supertuxkart/full-v0",
        render_mode=None,
        agent=AgentSpec(use_ai=False, name="walid"),
        track='abyss',
        num_kart=2,
        difficulty=0
    )

    env = PreprocessObservationWrapper(
        FlattenerWrapper(
            DiscreteActionsWrapper(
                PolarObservations(
                    ConstantSizedObservations(
                        env,
                        state_items = state_items,
                        state_karts = state_karts, 
                        state_paths = state_paths,
                    )
                )
            )
        ), ret_dict=False, norm=False,
    )

    env.close()

    observation_space = env.observation_space
    action_space = env.action_space
    action_dims = [space.n for space in env.action_space]
    observation_dim = observation_space.shape[0]

    return observation_space, action_space, observation_dim, action_dims,

def merge_buffers_for_supervision(buffer_names, start_step_id):
    buffers = [joblib.load(x) for x in buffer_names]

    size = sum([b.size for b in buffers])
    observations = torch.cat([b.observations[:b.size] for b in buffers], dim=0)
    track = torch.cat([b.track[:b.size] for b in buffers], dim=0)
    steps = torch.cat([b.steps[:b.size] for b in buffers], dim=0)
    actions = torch.cat([torch.stack([actions for actions in b.actions]) for b in buffers], dim=1)

    for buffer in buffers:
        del buffer

    all_indices = np.arange(0, size)
    all_indices = all_indices[steps[all_indices].flatten()>start_step_id]
    observations = observations[all_indices]
    actions = actions[:,all_indices]
    track = track[all_indices]

    # _,unique_indices = np.unique(observations.numpy(), axis=0, return_index=True)
    # unique_indices = torch.tensor(unique_indices)

    # unique_observations = observations[unique_indices]
    # unique_actions = actions[:, unique_indices]
    # observations = unique_observations
    # actions = unique_actions

    size = observations.size(0)


    return observations, actions, track, size