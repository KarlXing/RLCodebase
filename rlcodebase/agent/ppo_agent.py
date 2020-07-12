import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from .base_agent import BaseAgent
from ..utils.replay import Storage
from ..utils.torch_utils import tensor
from ..policy.ppo_policy import PPOPolicy

class PPOAgent(BaseAgent):
    def __init__(self, env, config, model):
        super().__init__(env, config)
        self.policy = PPOPolicy(model, config)
        self.storage = Storage(config.rollout_length)
        self.num_workers = config.num_workers
        self.rollout_length = config.rollout_length
        self.rollout_filled = 0
        self.ppo_epoch = config.ppo_epoch
        self.mini_batch_size = config.mini_batch_size
        self.sample_keys = ['s', 'a', 'log_prob', 'ret', 'adv']

    def step(self):
        self.storage.add({'s': tensor(self.state)})

        with torch.no_grad():
            action, log_prob, v, ent = self.policy.compute_actions(self.state)
        next_state, rwd, done, info = self.env.step(action)
        self.state = tensor(next_state)
        self.rollout_filled += 1
        self.storage.add({'a': action,
                          'log_prob': log_prob,
                          'v': v, 
                          'r': tensor(rwd),
                          'm': tensor(1-done)})

        if self.rollout_filled == self.rollout_length:
            with torch.no_grad():
                _, _, v, _ = self.policy.compute_actions(self.state)
                self.storage.compute_returns(v, self.config.discount)
                self.storage.after_fill(self.sample_keys)

            for i_epoch in range(self.ppo_epoch):
                sampler = BatchSampler(SubsetRandomSampler(range(self.rollout_length * self.num_workers)), 
                                       self.mini_batch_size, 
                                       drop_last=True)
                for indices in sampler:
                    batch = self.sample(indices)
                    self.policy.learn_on_batch(batch)

            self.rollout_filled = 0
            self.storage.reset()


    def save(self):
        pass

    def sample(self, indices):
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[indices]
        return batch








