import torch
import numpy as np
import random
from ..utils import *

class Replay:
    def __init__(self, replay_size, num_envs, obs_space, action_space, device=torch.device('cpu')):
        self.replay_size = replay_size // num_envs
        self.current_size = 0
        self.num_envs = num_envs
        self.obs_space = obs_space
        self.obs_shape = obs_space.shape
        self.action_dim = get_action_dim(action_space)
        self.discrete_action = is_discrete_action(action_space)
        self.device = device
        self.reset()

    def add(self, data):
        for k,v in data.items():
            getattr(self, k)[self.pos].copy_(v)
        self.pos = (self.pos + 1) % self.replay_size
        self.current_size = min(self.current_size+1, self.replay_size)

    def reset(self):
        self.pos = 0
        self.s = torch.zeros((self.replay_size, self.num_envs) + self.obs_shape, dtype=convert_dtype(self.obs_space.dtype)).to(self.device)
        self.next_s = torch.zeros((self.replay_size, self.num_envs) + self.obs_shape, dtype=convert_dtype(self.obs_space.dtype)).to(self.device)
        if self.discrete_action:
            self.a = torch.zeros((self.replay_size, self.num_envs), dtype=torch.float).to(self.device)
        else:
            self.a = torch.zeros((self.replay_size, self.num_envs, self.action_dim), dtype=torch.float).to(self.device)
        self.r = torch.zeros((self.replay_size, self.num_envs), dtype=torch.float).to(self.device)
        self.d = torch.zeros((self.replay_size, self.num_envs), dtype=torch.float).to(self.device)
        self.v = torch.zeros((self.replay_size, self.num_envs), dtype=torch.float).to(self.device)
