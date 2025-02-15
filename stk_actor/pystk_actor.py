from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor
from .wrappers import PreprocessObservationWrapper
from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations, DiscreteActionsWrapper
from pystk2_gymnasium.wrappers import FlattenerWrapper
#: The base environment name
env_name = "supertuxkart/full-v0"

#: Player name
player_name = "CherineJSK2"


def get_actor(
    state, observation_space: gym.spaces.Space, action_space: gym.spaces.Space
) -> Agent:
    actor = Actor(observation_space, action_space)

    # Returns a dummy actor
    if state is None:
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())


def get_wrappers(agent_name='normed_a2c_num5_best') -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    ll = lambda env : FlattenerWrapper(DiscreteActionsWrapper(PolarObservations(ConstantSizedObservations(env))))
    return [
        # Example of a custom wrapper
        lambda env: PreprocessObservationWrapper(ll(env), norm=True, ret_dict=True, agent_name=agent_name)
    ]
