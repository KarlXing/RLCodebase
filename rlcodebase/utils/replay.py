import torch
import numpy as np
import random


# Rollout for actor-critic algos
class Rollout:
    def __init__(self, size, keys=None):
        if keys is None:
            keys = []
        keys = keys + ['s', 'a', 'r', 'm',
                       'v', 'ret', 'adv', 'log_prob']
        self.keys = keys
        self.size = size
        self.reset()

    def add(self, data):
        for k, v in data.items():
            getattr(self, k).append(v)

    def reset(self):
        for key in self.keys:
            setattr(self, key, [])

    def after_fill(self, keys):
        for k in keys:
            setattr(self, k, torch.cat(getattr(self, k)[:self.size], dim=0))

    def compute_returns(self, next_value, discount, use_gae=False, gae_lambda=0.95):
        setattr(self, 'ret', [None] * self.size)
        setattr(self, 'adv', [None] * self.size)
        getattr(self, 'v').append(next_value)
        returns, advantages = next_value, 0
        for i in reversed(range(self.size)):
            returns = self.r[i] + discount * self.m[i] * returns
            if not use_gae:
                advantages = returns - self.v[i]
            else:
                td_error = self.r[i] + discount * self.v[i+1] * self.m[i] - self.v[i]
                advantages = advantages * discount * gae_lambda * self.m[i] +  td_error
            self.ret[i] = returns
            self.adv[i] = advantages
        getattr(self, 'v').pop()

    def norm_adv(self):
        self.adv = (self.adv - self.adv.mean())/self.adv.std()