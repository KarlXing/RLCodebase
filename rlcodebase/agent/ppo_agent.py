import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os

from .base_agent import BaseAgent
from ..utils import to_tensor, MultiDeque, convert_2dindex
from ..policy import PPOPolicy
from ..memory import Rollout


class PPOAgent(BaseAgent):
    def __init__(self, config, env, model, logger):
        super().__init__(config)
        self.policy = PPOPolicy(model,
                                config.optimizer,
                                config.lr,
                                config.value_loss_coef,
                                config.entropy_coef,
                                config.ppo_clip_param,
                                config.use_grad_clip,
                                config.max_grad_norm)
        self.env = env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        self.storage = Rollout(config.rollout_length, config.num_envs, env.observation_space, env.action_space, config.device)
        self.sample_keys = ['s', 'a', 'log_prob', 'ret', 'adv']
        self.rollout_filled = 0
        self.mini_batch_size = config.rollout_length * config.num_envs // config.num_mini_batch

    def step(self):
        with torch.no_grad():
            action, log_prob, v, ent = self.policy.inference(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu().numpy())
        self.rollout_filled += 1
        self.storage.add({'a': action,
                          'log_prob': log_prob,
                          'v': v, 
                          'r': to_tensor(rwd, self.config.device),
                          'd': to_tensor(done, self.config.device),
                          's': self.state})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)


        if self.rollout_filled == self.config.rollout_length:
            with torch.no_grad():
                _, _, v, _ = self.policy.inference(self.state)
                self.storage.compute_return(v, self.config.discount, self.config.use_gae, self.config.gae_lambda)
                self.storage.norm_adv() 

            mqueue = MultiDeque(tags = ['action_loss', 'value_loss', 'entropy'])
            for i_epoch in range(self.config.ppo_epoch):
                sampler = BatchSampler(SubsetRandomSampler(range(self.config.rollout_length * self.config.num_envs)), 
                                       self.mini_batch_size, 
                                       drop_last=True)
                self.policy.approx_kl = []
                for indices in sampler:
                    batch = self.sample(indices)
                    loss = self.policy.learn_on_batch(batch)
                    mqueue.add(loss)
                if self.config.target_kl is not None and np.mean(self.policy.approx_kl) > 1.5 * self.config.target_kl:
                    break
            
            self.logger.add_scalar(*mqueue.return_summary(), self.done_steps)
            self.rollout_filled = 0
            self.storage.reset()


    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.config.save_path, '%d-model.pt' % self.done_steps))

    def sample(self, indices):
        i1, i2 = convert_2dindex(indices, self.config.num_envs)
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[i1, i2]
        return batch








