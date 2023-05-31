import pickle
import jax
import jax.numpy as jnp
import numpy as np

from src.types import *

class ReplayBuffer:
    """
    Buffer to store environment transitions. 
    Discrete actions only.
    Works with multiple rewards.
    """
    def __init__(self, obs_shape, n_rewards, capacity):
        self.capacity = capacity

        self.obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
        self.next_obses = jnp.empty((capacity, *obs_shape), dtype=jnp.float32)
        self.actions = jnp.empty((capacity, 1), dtype=jnp.float32)
        self.rewards = jnp.empty((capacity, n_rewards), dtype=jnp.float32)
        self.not_dones = jnp.empty((capacity, 1), dtype=jnp.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx
    
    def add(self, obs, action, reward, next_obs, done):
        self.obses = self.obses.at[self.idx].set(obs)
        self.actions = self.actions.at[self.idx].set(action)
        self.rewards = self.rewards.at[self.idx].set(reward)
        self.next_obses = self.next_obses.at[self.idx].set(next_obs)
        self.not_dones = self.not_dones.at[self.idx].set(not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0


    def add_batch(self, obses, actions, rewards, next_obses, dones):
        self.obses = self.obses.at[self.idx:self.idx+len(obses)].set(obses)
        self.actions = self.actions.at[self.idx:self.idx+len(actions)].set(actions)
        self.rewards = self.rewards.at[self.idx:self.idx+len(rewards)].set(rewards)
        self.next_obses = self.next_obses.at[self.idx:self.idx+len(next_obses)].set(next_obses)
        self.not_dones = self.not_dones.at[self.idx:self.idx+len(dones)].set(jnp.logical_not(dones))

        self.idx = (self.idx + len(obses)) % self.capacity
        self.full = self.full or self.idx == 0


    def sample(self, batch_size, replace=False):
        idxs = np.random.choice(len(self), size=batch_size, replace=replace)

        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_obses = self.next_obses[idxs]
        not_dones = self.not_dones[idxs]

        return Transitions(obses, actions, rewards, next_obses, not_dones)
    
    def save(self, path):
        all_data = (self.idx, self.obses, self.actions, self.rewards, self.next_obses, self.not_dones, self.full)
        
        pickle.dump(all_data, open(path, 'wb'))

    def load(self, path, reset_capacity=True):
        """
        Load the replay buffer from a file.
        
        inputs:
            path (str): path to the file
            reset_capacity (bool): whether to reset the capacity of the replay buffer to the capacity of the loaded replay buffer
        """
        all_data = pickle.load(open(path, 'rb'))
        self.idx, self.obses, self.actions, self.rewards, self.next_obses, self.not_dones, full = all_data
        
        if reset_capacity:
            self.capacity = len(self.obses)
            self.full = True

        self.full = full

        return self