import ipyparallel
import tqdm

def run_episode(arg):
    track, difficulty, lap, num_karts = arg
    import torch
    import torch.nn.functional as F
    from pystk2_gymnasium.stk_wrappers import ConstantSizedObservations, PolarObservations, DiscreteActionsWrapper
    from pystk2_gymnasium.wrappers import FlattenerWrapper

    def preprocess_observation(obs):
        """Convert mixed observation space to flat tensor"""
        continuous_obs, discrete_obs = obs['continuous'], obs['discrete']
        continuous_tensor = torch.FloatTensor(continuous_obs)
        discrete_tensors = [
            F.one_hot(torch.tensor(x), num_classes=num_classes.n) 
            for x, num_classes in zip(discrete_obs, env.observation_space['discrete'])
        ]
        return torch.cat([continuous_tensor] + discrete_tensors)

    import gymnasium as gym
    from pystk2_gymnasium import AgentSpec
    
    records = []

    env = gym.make(
            "supertuxkart/full-v0",
            render_mode=None, 
            agent=AgentSpec(use_ai=True), 
            track=track, 
            difficulty=difficulty,
            laps=lap,
            num_kart=num_karts,
    )
    env = FlattenerWrapper(
        DiscreteActionsWrapper(
            PolarObservations(
                ConstantSizedObservations(
                    env,
                    state_items = 10,
                    state_karts = 10, 
                    state_paths = 10,
                )
            )
        )
    )

    ix = 0
    done = False
    obs, *_ = env.reset()
    prev_obs = obs

    while not done:
        ix += 1
        action = env.action_space.sample()          
        next_obs, reward, done, truncated, _ = env.step(action)
        action = next_obs['action']

        records.append(
            {
                # 'prev_obs':preprocess_observation(prev_obs), 
                'obs':preprocess_observation(obs), 
                'actions':torch.tensor(action), 
                'reward':torch.tensor(reward), 
                # 'next_obs':preprocess_observation(next_obs), 
                'done':torch.tensor(float(done or truncated)),
                'track': track,
                'step': ix-1,
            }
        )
        prev_obs = obs
        obs = next_obs

    env.close()
    return records

def parallel_run_episodes(args,):
    client = ipyparallel.Client()
    dview = client[:]

    dview.push({'run_episode': run_episode})
    results = dview.map(run_episode, args, )
    print('running:', len(args))
    # Flatten results into individual lists
    records = []
    for rec in tqdm.tqdm(results, total=len(args)):
        records.extend(rec)

    client.close()
    return records