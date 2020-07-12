import torch
from ..utils.replay import *
from ..utils.utils import *
from collections import deque

class BaseAgent():
    def __init__(self, config, env, writer):
        # general attributes for agents
        self.env = env
        self.max_steps = config.max_steps
        self.num_workers = config.num_workers
        self.save_interval = config.save_interval
        self.discount = config.discount
        self.save_path = config.save_path
        self.log_path = config.log_path
        self.log_interval = config.log_interval

        self.sample_keys = []
        self.policy = None
        self.storage = None
        self.done_steps = 0
        self.state = tensor(self.env.reset())
        self.writer = writer
        self.last_rewards = deque(maxlen=20)


    def run(self):
        while self.done_steps < self.max_steps:
            self.step()
            self.done_steps += self.num_workers
            if (self.done_steps) % self.save_interval == 0:
                self.save()

            if (self.done_steps) % self.log_interval == 0:
                print("Done steps: ", self.done_steps)
                print("Average Last Rewards is: ", 'None' if len(self.last_rewards) == 0 else np.mean(self.last_rewards))

    def step(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError