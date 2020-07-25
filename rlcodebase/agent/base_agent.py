import torch
from ..utils.replay import *
from ..utils.utils import *
from collections import deque

class BaseAgent():
    def __init__(self, config, env, logger):
        self.env = env
        self.config = config
        self.logger = logger

        self.sample_keys = []
        self.policy = None
        self.storage = None
        self.done_steps = 0
        self.state = tensor(self.env.reset(), self.config.device)


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