import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import os

from .base_trainer import BaseTrainer
from ..utils import to_tensor, MultiDeque, convert_2dindex, update_maxstep_done
from ..agent import PPOAgent
from ..memory import Rollout


class PPOTrainer(BaseTrainer):
    def __init__(self, config, env, model, logger):
        super().__init__(config)
        self.agent = PPOAgent(model,
                                config.optimizer,
                                config.lr,
                                config.value_loss_coef,
                                config.entropy_coef,
                                config.ppo_clip_param,
                                config.use_grad_clip,
                                config.max_grad_norm,
                                config.use_value_clip)
        self.env = env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        self.storage = Rollout(config.rollout_length, config.num_envs, env.observation_space, env.action_space, config.memory_device)
        self.sample_keys = ['s', 'a', 'log_prob', 'ret', 'adv', 'v']
        self.rollout_filled = 0
        self.batch_size = config.rollout_length * config.num_envs // config.num_mini_batch
        self.mini_batch_size = config.mini_batch_size
        if self.batch_size < self.mini_batch_size:
            self.mini_batch_size = self.batch_size
        self.gradient_accumulation_steps = self.batch_size // self.mini_batch_size
        self.gradient_accumulation_cnt = 0

    def step(self):
        with torch.no_grad():
            action, log_prob, v, ent = self.agent.inference(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu().numpy())
        # done = update_maxstep_done(info, done, self.env.max_episode_steps)
        self.rollout_filled += 1
        self.storage.add({'a': action,
                          'log_prob': log_prob,
                          'v': v, 
                          'r': to_tensor(rwd, self.config.memory_device),
                          'd': to_tensor(done, self.config.memory_device),
                          's': self.state})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)


        if self.rollout_filled == self.config.rollout_length:
            with torch.no_grad():
                _, _, v, _ = self.agent.inference(self.state)
                self.storage.compute_return(v.to(self.config.memory_device), self.config.discount, self.config.use_gae, self.config.gae_lambda)
                self.storage.norm_adv() 

            mqueue = MultiDeque(tags = ['action_loss', 'value_loss', 'entropy'])
            for i_epoch in range(self.config.ppo_epoch):
                sampler = BatchSampler(SubsetRandomSampler(range(self.config.rollout_length * self.config.num_envs)), 
                                       self.mini_batch_size, 
                                       drop_last=True)
                self.agent.approx_kl = []
                for indices in sampler:
                    batch = self.sample(indices)
                    self.gradient_accumulation_cnt += 1
                    accumulation = self.gradient_accumulation_cnt % self.gradient_accumulation_steps != 0
                    loss = self.agent.learn_on_batch(batch, accumulation)
                    mqueue.add(loss)
                if self.config.target_kl is not None and np.mean(self.agent.approx_kl) > 1.5 * self.config.target_kl:
                    break
            
            self.logger.add_scalar(*mqueue.return_summary(), self.done_steps)
            self.rollout_filled = 0
            self.storage.reset()

    def save(self):
        torch.save(self.agent.model.state_dict(), os.path.join(self.config.save_path, '%d-model.pt' % self.done_steps))

    def sample(self, indices):
        i1, i2 = convert_2dindex(indices, self.config.num_envs)
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[i1, i2].to(self.config.device)
        return batch








