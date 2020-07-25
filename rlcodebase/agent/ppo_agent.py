import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os
from .base_agent import BaseAgent
from ..utils import Rollout
from ..utils import tensor, MultiDeque
from ..policy import PPOPolicy

class PPOAgent(BaseAgent):
    def __init__(self, config, env, model, writer = None):
        super().__init__(config, env, writer)
        self.policy = PPOPolicy(model,
                                config.optimizer,
                                config.lr,
                                config.value_loss_coef,
                                config.entropy_coef,
                                config.ppo_clip_param,
                                config.use_grad_clip,
                                config.max_grad_norm)
        self.storage = Rollout(config.rollout_length)
        self.sample_keys = ['s', 'a', 'log_prob', 'ret', 'adv']
        self.rollout_filled = 0
        self.mini_batch_size = config.rollout_length * config.num_envs // config.num_mini_batch


    def step(self):
        with torch.no_grad():
            action, log_prob, v, ent = self.policy.compute_actions(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu().numpy())
        self.rollout_filled += 1
        self.storage.add({'a': action,
                          'log_prob': log_prob,
                          'v': v, 
                          'r': tensor(rwd, self.config.device),
                          'm': tensor(1-done, self.config.device),
                          's': self.state})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = tensor(next_state, self.config.device)


        if self.rollout_filled == self.config.rollout_length:
            if not self.config.eval:
                with torch.no_grad():
                    _, _, v, _ = self.policy.compute_actions(self.state)
                    self.storage.compute_returns(v, self.config.discount)
                    self.storage.after_fill(self.sample_keys)
                    self.storage.norm_adv() 

                mqueue = MultiDeque(tags = ['action_loss', 'value_loss', 'entropy'])
                for i_epoch in range(self.config.ppo_epoch):
                    sampler = BatchSampler(SubsetRandomSampler(range(self.config.rollout_length * self.config.num_envs)), 
                                           self.mini_batch_size, 
                                           drop_last=True)
                    for indices in sampler:
                        batch = self.sample(indices)
                        loss = self.policy.learn_on_batch(batch)
                        mqueue.add(loss)
                self.logger.add_scalar(*mqueue.return_summary(), self.done_steps)

            self.rollout_filled = 0
            self.storage.reset()


    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.config.save_path, 'model.pt'))

    def sample(self, indices):
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[indices]
        return batch








