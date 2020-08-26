import torch
import numpy as np
import random
from ..utils import *

class Replay:
    def __init__(self, replay_size, num_envs, obs_space, action_space, device=torch.device('cpu')):
        self.replay_size = replay_size // num_envs
        self.current_size = 0
        self.num_envs = num_envs
        self.obs_shape = obs_space.shape
        self.action_dim = get_action_dim(action_space)
        self.discrete_action = is_discrete_action(action_space)
        self.device = device
        self.reset()

    def add(self, data):
        for k,v in data.items():
            getattr(self, k)[self.pos] = to_numpy(v).copy()
        self.pos = (self.pos + 1) % self.replay_size
        self.current_size = min(self.current_size+1, self.replay_size)

    def reset(self):
        self.pos = 0
        self.s = np.zeros((self.replay_size, self.num_envs) + self.obs_shape, dtype=np.float32)
        self.next_s = np.zeros((self.replay_size, self.num_envs) + self.obs_shape, dtype=np.float32)
        if self.discrete_action:
            self.a = np.zeros((self.replay_size, self.num_envs), dtype=np.float32)
        else:
            self.a = np.zeros((self.replay_size, self.num_envs, self.action_dim), dtype=np.float32)
        self.r = np.zeros((self.replay_size, self.num_envs), dtype=np.float32)
        self.d = np.zeros((self.replay_size, self.num_envs), dtype=np.float32)
        self.v = np.zeros((self.replay_size, self.num_envs), dtype=np.float32)