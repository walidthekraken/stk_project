from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
import inspect
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import Actor
from .pystk_actor import env_name, get_wrappers, player_name
from .agent import UnifiedSACPolicy

if __name__ == "__main__":


    agent_name = 'normed_a2c_num5_best'
    # Setup the environment
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(agent_name=agent_name),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]

    # (2) Learn
    mod_path = Path(inspect.getfile(get_wrappers)).parent
    actor = Actor(
        env.observation_space, env.action_space,
        net_arch=[1024,1024,1024], 
        activation_fn=torch.nn.Tanh,
        # state_dict_path='policy_normed_1024_1024_1024_Tanh_statedict'
        # state_dict_path='ppo_policy_normed_1024_1024_1024_Tanh_statedict'
        # state_dict_path='a2c_policy_normed_1024_1024_1024_Tanh_statedict'
        # state_dict_path='a2c_2m_policy_normed_1024_1024_1024_Tanh_statedict'
        state_dict_path=mod_path/f'trained_agents/{agent_name}/statedict'
    )
    # ...

    # (3) Save the actor sate
    torch.save(actor.state_dict(), mod_path / "pystk_actor.pth")
