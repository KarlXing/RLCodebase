import torch
import numpy as np
import random
from ..utils import *

# Memory for q networks
class Replay:
    def __init__(self, obs_dim, act_dim, max_size):
        self.state = np.zeros(combined_shape(size, obs_dim)).float()
        self.next_state = np.zeros(combined_shape(size, obs_dim)).float()
        self.action = np.zeros(combined_shape(size, act_dim)).float()
        self.reward = np.zeros(size).float()
        self.done = np.zeros(size).float()
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def store(self, state, action, reward, next_state, done):
        self.state[self.ptr].copy_(state)
        self.next_state[self.ptr].copy_(next_state)
        self.action[self.ptr].copy_(action)
        self.reward[self.ptr].copy_(rew)
        self.done[self.ptr].copy_(done)
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(s=self.state[idxs],
                     next_s=self.next_state[idxs],
                     a=self.action[idxs],
                     r=self.reward[idxs],
                     m=self.done[idxs])
        return batch