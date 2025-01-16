import numpy as np

import gymnasium as gym
from pystk2_gymnasium import AgentSpec
import torch

def calculate_total_obs_dim(observation_space):
    """
    Calculate total observation dimension for mixed observation space
    
    Args:
        observation_space: Dict space with 'continuous' and 'discrete' keys
    Returns:
        Total dimension after flattening and one-hot encoding
    """
    total_dim = 0
    
    # Add continuous dimensions
    continuous_space = observation_space['continuous']
    total_dim += continuous_space.shape[0]
    
    # Add dimensions for one-hot encoded discrete observations
    discrete_space = observation_space['discrete']
    for n in discrete_space.nvec:  # nvec contains the number of values for each discrete dimension
        total_dim += n
        
    return total_dim

class SACRolloutBuffer:
    def __init__(self, buffer_size, obs_dim, action_dims):
        """
        Initialize buffer for multi-discrete actions
        
        Args:
            buffer_size: Maximum size of the buffer
            obs_dim: Dimension of the observation space
            action_dims: List of dimensions for each discrete action space
        """
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.num_actions = len(action_dims)
        self.position = 0
        self.size = 0
        
        # Initialize storage
        self.observations = torch.zeros((buffer_size, obs_dim))
        self.prev_observations = torch.zeros((buffer_size, obs_dim))
        self.actions = [torch.zeros(buffer_size, dtype=torch.long) for _ in action_dims]
        self.rewards = torch.zeros((buffer_size, 1))
        self.next_observations = torch.zeros((buffer_size, obs_dim))
        self.dones = torch.zeros((buffer_size, 1))
        self.track = torch.zeros((buffer_size, 1))
        self.steps = torch.zeros((buffer_size, 1))

        self.tracks = {
            'abyss':0,
            'black_forest':1,
            'candela_city':2,
            'cocoa_temple':3,
            'cornfield_crossing':4,
            'fortmagma':5,
            'gran_paradiso_island':6,
            'hacienda':7,
            'lighthouse':8,
            'mines':9,
            'minigolf':10,
            'olivermath':11,
            'ravenbridge_mansion':12,
            'sandtrack':13,
            'scotland':14,
            'snowmountain':15,
            'snowtuxpeak':16,
            'stk_enterprise':17,
            'volcano_island':18,
            'xr591':19,
            'zengarden':20,
        }
    
    def add(self, obs, actions, reward, next_obs, done, step, track, prev_obs):
        """
        Add a transition to the buffer
        
        Args:
            obs: Current observation
            actions: List of actions, one for each discrete action space
            reward: Reward received
            next_obs: Next observation
            done: Boolean indicating if episode ended
        """
        self.observations[self.position] = obs
        for i, action in enumerate(actions):
            self.actions[i][self.position] = action
        self.rewards[self.position] = reward
        self.next_observations[self.position] = next_obs
        self.prev_observations[self.position] = prev_obs
        self.dones[self.position] = done
        self.steps[self.position] = step
        self.track[self.position] = self.tracks[track]
        
        self.position = (self.position + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            torch.stack([actions[indices] for actions in self.actions]),
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            self.steps[indices],
            self.track[indices],
        )
    
    def get_batches(self, batch_size, start_step_id):
        """
        Retrieve all transitions in batches
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            Batches of transitions
        """
        all_indices = np.arange(0, self.size)
        all_indices = all_indices[self.steps[all_indices].flatten()>start_step_id]
        num_batches = int(np.ceil(len(all_indices) / batch_size))
        np.random.shuffle(all_indices)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_indices))
            indices = all_indices[start_idx:end_idx]
            yield (
                self.observations[indices],
                torch.stack([actions[indices] for actions in self.actions]),
                self.rewards[indices],
                self.next_observations[indices],
                self.prev_observations[indices],
                self.dones[indices],
                self.steps[indices],
                self.track[indices],
            )
    def get_sequences_by_track(self, seq_len, batch_size):
        """
        Generate sequences of transitions grouped by track, with padding for LSTM training.
        
        Args:
            seq_len: Length of each sequence
            batch_size: Number of sequences per batch
        
        Yields:
            Batches of padded sequences grouped by track
        """
        # Group indices by track
        track_indices = {track: [] for track in self.tracks}
        for idx in range(self.size):
            track_name = next(key for key, value in self.tracks.items() if value == int(self.track[idx].item()))
            track_indices[track_name].append(idx)
        
        # Generate sequences for each track
        for track, indices in track_indices.items():
            if not indices:
                continue
            
            sequences = []
            indices = np.array(indices)
            
            for start_idx in range(len(indices)):
                end_idx = start_idx
                seq_indices = indices[max(0, end_idx - seq_len):end_idx]
                
                # Pad with zeros if sequence is shorter than seq_len
                pad_len = seq_len - len(seq_indices)
                obs_seq = torch.zeros((seq_len, self.obs_dim))
                act_seq = torch.zeros((seq_len, self.num_actions), dtype=torch.long)
                rew_seq = torch.zeros((seq_len, 1))
                next_obs_seq = torch.zeros((seq_len, self.obs_dim))
                done_seq = torch.zeros((seq_len, 1))
                step_seq = torch.zeros((seq_len, 1))
                
                obs_seq[pad_len:] = self.observations[seq_indices]
                for i in range(self.num_actions):
                    act_seq[pad_len:, i] = self.actions[i][seq_indices]
                rew_seq[pad_len:] = self.rewards[seq_indices]
                next_obs_seq[pad_len:] = self.next_observations[seq_indices]
                done_seq[pad_len:] = self.dones[seq_indices]
                step_seq[pad_len:] = self.steps[seq_indices]
                
                sequences.append((obs_seq, act_seq, rew_seq, next_obs_seq, done_seq, step_seq))
            
            # Yield sequences in batches
            for batch_start in range(0, len(sequences), batch_size):
                batch = sequences[batch_start:batch_start + batch_size]
                yield (
                    track,
                    torch.stack([seq[0] for seq in batch]),  # Observations
                    torch.stack([seq[1] for seq in batch]),  # Actions
                    torch.stack([seq[2] for seq in batch]),  # Rewards
                    torch.stack([seq[3] for seq in batch]),  # Next observations
                    torch.stack([seq[4] for seq in batch]),  # Dones
                    torch.stack([seq[5] for seq in batch]),  # Steps
                )

