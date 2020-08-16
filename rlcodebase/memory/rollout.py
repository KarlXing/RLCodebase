import torch
import numpy as np
import random
from ..utils import *

class Rollout:
    def __init__(self, rollout_length, num_envs, obs_space, action_space, device=torch.device('cpu')):
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.obs_shape = obs_space.shape
        self.action_dim = get_action_dim(action_space)
        self.discrete_action = is_discrete_action(action_space)
        self.device = device
        self.reset()

    def add(self, data):
        for k,v in data.items():
            getattr(self, k)[self.pos].copy_(v)
        self.pos = (self.pos + 1) % self.rollout_length

    def reset(self):
        self.pos = 0
        self.s = torch.zeros((self.rollout_length, self.num_envs,) + self.obs_shape, dtype=torch.float).to(self.device)
        if self.discrete_action:
            self.a = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)
        else:
            self.a = torch.zeros((self.rollout_length, self.num_envs, self.action_dim), dtype=torch.float).to(self.device)
        self.r = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)
        self.d = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)
        self.ret = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)
        self.v = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)
        self.log_prob = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)
        self.adv = torch.zeros((self.rollout_length, self.num_envs), dtype=torch.float).to(self.device)


    def compute_return(self, next_v, discount, use_gae=False, gae_lambda=0.95):
        next_v = next_v.detach()
        if not use_gae:
            ret = next_v
            for i in reversed(range(self.rollout_length)):
                ret = self.r[i] + discount * ret * (1-self.d[i])
                adv = ret - self.v[i]
                self.adv[i] = adv
        else:
            gae = 0
            for i in reversed(range(self.rollout_length)):
                next_v = next_v if i == (self.rollout_length-1) else self.v[i+1]
                td_error = self.r[i] + discount * next_v * (1-self.d[i]) - self.v[i]
                gae = td_error + discount * gae_lambda * gae * (1-self.d[i])
                self.adv[i] = gae
        self.ret = self.adv + self.v

    def norm_adv(self):
        self.adv = (self.adv - self.adv.mean()) / (self.adv.std() + 1e-8)