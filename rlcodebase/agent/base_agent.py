import torch
from ..utils.replay import *

class BaseAgent():
    def __init__(self, env, model, config):
        self.env = env
        self.policy = None
        self.storage = None
        self.max_steps = config.max_steps
        self.done_steps = 0
        self.state = self.env.reset()
        self.sample_keys = []
        self.config = config

    def run(self):
        while self.done_steps < self.max_steps:
            self.step()
            self.done_steps += self.config.num_workers

            if (self.done_steps) % self.config.save_interval == 0:
                self.save()

    def step(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError