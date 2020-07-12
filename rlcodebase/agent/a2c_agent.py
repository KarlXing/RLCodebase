import torch
import os
from .base_agent import BaseAgent
from ..utils.replay import Rollout
from ..utils.utils import tensor
from ..utils.log import log_rewards
from ..policy.a2c_policy import A2CPolicy

class A2CAgent(BaseAgent):
    def __init__(self, config, env, model, writer = None):
        super().__init__(config, env, writer)
        self.policy = A2CPolicy(model, config)
        self.storage = Rollout(config.rollout_length)

        self.rollout_length = config.rollout_length

        self.sample_keys = ['s', 'a', 'ret', 'adv']
        self.rollout_filled = 0

    def step(self):
        self.storage.add({'s': tensor(self.state)})

        with torch.no_grad():
            action, log_prob, v, ent = self.policy.compute_actions(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu())
        self.state = tensor(next_state)
        self.rollout_filled += 1
        self.storage.add({'a': action,
                          'v': v, 
                          'r': tensor(rwd),
                          'm': tensor(1-done)})
        log_rewards(self.writer, info, self.done_steps, self.last_rewards)

        if self.rollout_filled == self.rollout_length:
            with torch.no_grad():
                _, _, v, _ = self.policy.compute_actions(self.state)
                self.storage.compute_returns(v, self.discount)
                self.storage.after_fill(self.sample_keys)

            indices = list(range(self.rollout_length*self.num_workers))
            batch = self.sample(indices)
            loss = self.policy.learn_on_batch(batch)
            self.writer.add_scalar('action_loss', loss[0], self.done_steps)
            self.writer.add_scalar('value_loss', loss[1], self.done_steps)
            self.writer.add_scalar('entropy', loss[2], self.done_steps)

            self.rollout_filled = 0
            self.storage.reset()


    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.save_path, 'a2c_model.pt'))

    def sample(self, indices):
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[indices]
        return batch








