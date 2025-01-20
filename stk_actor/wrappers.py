
import gymnasium as gym
import numpy as np
import torch

import torch.functional as F

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
    def __init__(self, env, ret_dict=True, norm=True):
        """
        A Gym wrapper to preprocess mixed observation space (continuous + discrete)
        into a flat tensor.
        
        Args:
            env: The Gym environment to wrap.
        """
        super().__init__(env)
        self.observation_space = self._get_flat_observation_space(env.observation_space)
        self.ret_dict = ret_dict
        self.mean = torch.tensor(
            [1.2639206647872925,
 0.0,
 -0.07462655007839203,
 -0.07617677748203278,
 3.325659990310669,
 -0.6260609030723572,
 686.5313720703125,
 0.90716952085495,
 3.248101347708143e-05,
 -2.2646768229606096e-06,
 0.7184554934501648,
 -0.00979185476899147,
 -0.018949387595057487,
 42.55070114135742,
 -0.0018095501000061631,
 -0.0037681907415390015,
 50.57154083251953,
 0.006202289834618568,
 0.0015260628424584866,
 60.573333740234375,
 -0.0015078107826411724,
 -0.008679982274770737,
 83.0267333984375,
 -0.004257954191416502,
 -0.052518781274557114,
 92.90336608886719,
 -0.04041219875216484,
 -0.07217442244291306,
 29.27079963684082,
 -0.06311116367578506,
 -0.09916721284389496,
 44.90357208251953,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.10005245357751846,
 682.83642578125,
 690.8487548828125,
 685.5414428710938,
 693.1297607421875,
 687.2418212890625,
 694.6862182617188,
 688.3353271484375,
 695.63671875,
 689.5924682617188,
 696.7789306640625,
 0.036972709000110626,
 0.032977644354104996,
 7.680975437164307,
 0.023797351866960526,
 0.004713739734143019,
 14.885013580322266,
 0.01685260608792305,
 -0.003614815417677164,
 22.13359260559082,
 0.010710783302783966,
 -0.006463498808443546,
 29.126462936401367,
 0.005819129757583141,
 -0.007047213148325682,
 35.78960418701172,
 0.08564247190952301,
 0.011537699960172176,
 6.701535701751709,
 0.03396379202604294,
 0.0338205024600029,
 8.146260261535645,
 0.022045161575078964,
 0.0064937700517475605,
 15.329230308532715,
 0.015207406133413315,
 -0.001814433140680194,
 22.549062728881836,
 0.009987653233110905,
 -0.004953742492944002,
 29.498981475830078,
 10.452672958374023,
 10.426018714904785,
 10.403871536254883,
 10.385071754455566,
 10.37171459197998,
 0.2544378638267517,
 1.0111130475997925,
 -0.006203718949109316,
 0.047264423221349716,
 17.174535751342773,
 0.011489897966384888,
 0.003900744253769517,
 0.04766101762652397,
 0.030281564220786095,
 0.0,
 0.0,
 0.04146302491426468,
 0.0,
 0.0,
 0.8652037382125854,
 0.4444541931152344,
 0.23419933021068573,
 0.03508015722036362,
 0.26018109917640686,
 0.026085197925567627,
 0.0,
 0.0,
 0.4107459783554077,
 0.26497969031333923,
 0.03951539844274521,
 0.26854822039604187,
 0.01621069572865963,
 0.0,
 0.0,
 0.4971160888671875,
 0.18721193075180054,
 0.039390929043293,
 0.2605591118335724,
 0.015721958130598068,
 0.0,
 0.0,
 0.41856396198272705,
 0.22104644775390625,
 0.029045993462204933,
 0.30909326672554016,
 0.022250350564718246,
 0.0,
 0.0,
 0.41806697845458984,
 0.23936127126216888,
 0.04056426137685776,
 0.28310322761535645,
 0.018904240801930428,
 0.0,
 0.0,
 0.9955912828445435,
 0.0044087013229727745,
 0.0,
 0.0,
 0.0,
 1.0,
 1.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]
        )
        self.std = torch.tensor(
            [4.425504207611084,
 0.0,
 1.603779673576355,
 1.7964426279067993,
 9.993358612060547,
 10.51357650756836,
 512.2705078125,
 1.6997371912002563,
 0.026459209620952606,
 0.010543138720095158,
 0.02026776410639286,
 0.7250668406486511,
 0.39837801456451416,
 34.88729476928711,
 0.7514939904212952,
 0.44161456823349,
 35.87752914428711,
 0.8336662650108337,
 0.539232075214386,
 41.78753662109375,
 1.0223793983459473,
 0.7299631237983704,
 47.53340530395508,
 1.098395586013794,
 0.8274492025375366,
 47.81513214111328,
 1.6523897647857666,
 1.749129295349121,
 28.97491455078125,
 2.2789063453674316,
 2.5216972827911377,
 40.0125846862793,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.06416473537683487,
 512.6343994140625,
 511.6667175292969,
 511.3175964355469,
 510.3323059082031,
 509.7705383300781,
 508.8245544433594,
 508.5116271972656,
 507.6471252441406,
 507.43463134765625,
 506.58172607421875,
 0.6846545338630676,
 0.5445173382759094,
 13.64755916595459,
 0.44763320684432983,
 0.452096164226532,
 17.358463287353516,
 0.42862439155578613,
 0.4299456775188446,
 20.870670318603516,
 0.451620489358902,
 0.43150416016578674,
 24.120229721069336,
 0.4922274351119995,
 0.4394797086715698,
 27.007665634155273,
 2.4632468223571777,
 2.7507951259613037,
 13.712183952331543,
 0.689609706401825,
 0.5509278774261475,
 17.15859603881836,
 0.45551878213882446,
 0.4613153338432312,
 20.009708404541016,
 0.43676477670669556,
 0.4395506978034973,
 22.933229446411133,
 0.45909419655799866,
 0.4403861463069916,
 25.679624557495117,
 2.8094654083251953,
 2.8031206130981445,
 2.80706524848938,
 2.8221468925476074,
 2.842970848083496,
 1.5037912130355835,
 0.11496783792972565,
 0.9299623370170593,
 1.1188806295394897,
 6.216769218444824,
 0.10657340288162231,
 0.06233403459191322,
 0.2130480855703354,
 0.1713610738515854,
 0.0,
 0.0,
 0.19935867190361023,
 0.0,
 0.0,
 0.3415059745311737,
 0.4969053268432617,
 0.4234975576400757,
 0.18398252129554749,
 0.4387334883213043,
 0.15938878059387207,
 0.0,
 0.0,
 0.4919694662094116,
 0.44132259488105774,
 0.19481778144836426,
 0.44320452213287354,
 0.12628509104251862,
 0.0,
 0.0,
 0.49999192357063293,
 0.3900817334651947,
 0.19452330470085144,
 0.4389398992061615,
 0.12439771741628647,
 0.0,
 0.0,
 0.49332383275032043,
 0.414951890707016,
 0.1679355502128601,
 0.4621199071407318,
 0.14749675989151,
 0.0,
 0.0,
 0.4932415187358856,
 0.4266938269138336,
 0.19727858901023865,
 0.4505063593387604,
 0.13618695735931396,
 0.0,
 0.0,
 0.06625155359506607,
 0.06625155359506607,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0,
 0.0]
        )
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
        
        flat_array = np.concatenate([continuous_array] + discrete_arrays).flatten()
        flat_array = torch.tensor(flat_array)
        normed_array = flat_array
        if self.norm:
            normed_array = (flat_array - self.mean) / (self.std + 1e-8)
            print(f"{normed_array.shape=}")
        if self.ret_dict:
            return {
                'normed_obs':normed_array.numpy(),
                'obs':flat_array.numpy(),
                'normed':1 if self.norm else 0
            }
        return normed_array.numpy()
