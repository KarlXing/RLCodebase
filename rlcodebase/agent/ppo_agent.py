import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
from .base_agent import BaseAgent
from ..utils.replay import Rollout
from ..utils.utils import tensor
from ..utils.log import log_rewards, MultiDeque
from ..policy.ppo_policy import PPOPolicy

class PPOAgent(BaseAgent):
    def __init__(self, config, env, model, writer = None):
        super().__init__(config, env, writer)
        self.policy = PPOPolicy(model, config)
        self.storage = Rollout(config.rollout_length)

        self.rollout_length = config.rollout_length
        self.ppo_epoch = config.ppo_epoch
        self.mini_batch_size = config.mini_batch_size

        self.sample_keys = ['s', 'a', 'log_prob', 'ret', 'adv']
        self.rollout_filled = 0

    def step(self):
        self.storage.add({'s': tensor(self.state)})

        with torch.no_grad():
            action, log_prob, v, ent = self.policy.compute_actions(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu().numpy())
        self.state = tensor(next_state)
        self.rollout_filled += 1
        self.storage.add({'a': action,
                          'log_prob': log_prob,
                          'v': v, 
                          'r': tensor(rwd),
                          'm': tensor(1-done)})
        log_rewards(self.writer, info, self.done_steps, self.last_rewards)

        if self.rollout_filled == self.rollout_length:
            with torch.no_grad():
                _, _, v, _ = self.policy.compute_actions(self.state)
                self.storage.compute_returns(v, self.discount)
                self.storage.after_fill(self.sample_keys)
                self.storage.norm_adv()

            mqueue = MultiDeque(tags = ['action_loss', 'value_loss', 'entropy'])
            for i_epoch in range(self.ppo_epoch):
                sampler = BatchSampler(SubsetRandomSampler(range(self.rollout_length * self.num_workers)), 
                                       self.mini_batch_size, 
                                       drop_last=True)
                for indices in sampler:
                    batch = self.sample(indices)
                    loss = self.policy.learn_on_batch(batch)
                    mqueue.add(loss)
            mqueue.write(self.writer, self.done_steps)

            self.rollout_filled = 0
            self.storage.reset()


    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.save_path, 'ppo_model.pt'))

    def sample(self, indices):
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[indices]
        return batch








