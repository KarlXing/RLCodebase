import torch
from collections import deque
from ..utils.utils import *

class BaseAgent():
    def __init__(self, config):
        self.config = config
        self.sample_keys = []
        self.policy = None
        self.storage = None
        self.logger = None
        self.done_steps = 0

    def run(self):
        while self.done_steps < self.config.max_steps:
            self.step()
            self.done_steps += self.config.num_envs
            if (self.done_steps) % self.config.save_interval == 0:
                self.save()

            if self.config.echo_interval != 0 and self.done_steps % self.config.echo_interval == 0:
                print("Done steps: ", self.done_steps)
                self.logger.print_last_rewards()
        self.logger.close()

    def step(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def sample(self, indices):
        raise NotImplementedError
