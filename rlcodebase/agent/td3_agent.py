import torch
import os
import random
import numpy as np

from .base_agent import BaseAgent
from ..memory import Replay, PrioritizedReplay
from ..utils import to_numpy, to_tensor, convert_2dindex
from ..policy import TD3Policy


class TD3Agent(BaseAgent):
    def __init__(self, config, env, eval_env, model, target_model, logger):
        super().__init__(config)
        self.policy = TD3Policy(model,
                                target_model,
                                config.discount,
                                config.optimizer,
                                config.lr,
                                config.soft_update_rate,
                                config.target_noise,
                                config.target_noise_clip,
                                config.policy_delay)
        self.env = env
        self.eval_env = eval_env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        if config.use_per:
            self.storage = PrioritizedReplay(config.replay_size, 
                                             config.num_envs, 
                                             env.observation_space, 
                                             env.action_space, 
                                             config.replay_device,
                                             config.per_max_p,
                                             config.per_alpha,
                                             config.per_eps,
                                             config.per_beta_start,
                                             config.per_beta_end)        
        else:
            self.storage = Replay(config.replay_size, config.num_envs, env.observation_space, env.action_space, config.replay_device)
        self.sample_keys = ['s', 'a', 'r', 'd', 'next_s']
        self.action_limit = {'high':self.env.action_space.high[0], 'low':self.env.action_space.low[0]}

    def step(self):
        if self.done_steps < self.config.warmup_steps:
            action = np.array([self.env.action_space.sample() for _ in range(self.config.num_envs)])
        else:
            with torch.no_grad():
                action = self.policy.inference(self.state).cpu().numpy()
                action += np.random.normal(0, self.config.action_noise, action.shape)
                action = np.clip(action, self.action_limit['low'], self.action_limit['high'])
        next_state, rwd, done, info = self.env.step(action)
        self.storage.add({'s': to_tensor(self.state, self.config.replay_device),
                          'a': to_tensor(action, self.config.replay_device),
                          'r': to_tensor(rwd, self.config.replay_device),
                          'd': to_tensor(done, self.config.replay_device),
                          'next_s': to_tensor(next_state, self.config.replay_device)})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)

        if self.done_steps > self.config.warmup_steps:
            for i in range(self.config.num_envs):
                batch = self.storage.sample(self.config.replay_batch, self.sample_keys, self.config.device)
                loss, td_error = self.policy.learn_on_batch(batch, self.action_limit)
                self.logger.add_scalar(['action_loss', 'value_loss'], loss, self.done_steps+i)
                self.logger.add_scalar(['td_error'], [td_error.mean().item()], self.done_steps)
                if self.config.use_per:
                    self.storage.update_p(batch['indices'], td_error)

        if self.config.use_per:
            self.storage.update_beta(self.config.max_steps, self.done_steps)

    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.config.save_path, '%d-model.pt' % self.done_steps))

    def eval(self):
        eval_returns = []
        state = to_tensor(self.eval_env.reset(), self.config.device)
        while (len(eval_returns) < self.config.eval_episodes):
            with torch.no_grad():
                action = self.policy.inference(state).cpu().numpy()
            next_state, rwd, done, info = self.eval_env.step(action)
            for (i,d) in enumerate(done):
                if d:
                    eval_returns.append(info[i]['episodic_return'])
            state = to_tensor(next_state, self.config.device)
        self.logger.add_scalar(['eval_returns'], [np.mean(eval_returns)], self.done_steps)
