import torch
import numpy as np
import random


class Storage:
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
            setattr(self, k, torch.cat(getattr(self, k), dim=0))

    # Todo: use_gae
    def compute_returns(self, next_value, discount):
        setattr(self, 'ret', [None] * self.size)
        setattr(self, 'adv', [None] * self.size)
        returns = next_value
        for i in reversed(range(self.size)):
            returns = self.r[i] + discount * storage.m[i] * returns
            advantages = returns - storage.v[i]
            storage.ret[i] = returns
            storage.adv[i] = advantages
