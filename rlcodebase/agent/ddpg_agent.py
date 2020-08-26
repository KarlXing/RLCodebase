import torch
import os
import random
import numpy as np

from .base_agent import BaseAgent
from ..memory import Replay 
from ..utils import to_numpy, to_tensor, convert_2dindex
from ..policy import DDPGPolicy


class DDPGAgent(BaseAgent):
    def __init__(self, config, env, eval_env, model, target_model, logger):
        super().__init__(config)
        self.policy = DDPGPolicy(model,
                                target_model,
                                config.discount,
                                config.optimizer,
                                config.lr,
                                config.soft_update_rate)
        self.env = env
        self.eval_env = eval_env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        self.storage = Replay(config.replay_size, config.num_envs, env.observation_space, env.action_space, config.device)
        self.sample_keys = ['s', 'a', 'r', 'd', 'next_s']

    def step(self):
        if self.done_steps < self.config.warmup_steps:
            action = np.array([self.env.action_space.sample() for _ in range(self.config.num_envs)])
        else:
            with torch.no_grad():
                action = self.policy.inference(self.state).cpu().numpy()
                action += np.random.normal(0, self.config.action_noise, action.shape)
                action = np.clip(action, -1, 1)
        next_state, rwd, done, info = self.env.step(action)
        self.storage.add({'s': self.state,
                          'a': action,
                          'r': rwd,
                          'd': done,
                          'next_s': next_state})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)

        if self.done_steps > self.config.warmup_steps:
            indices = random.sample(list(range(self.storage.current_size * self.config.num_envs)), self.config.replay_batch)
            batch = self.sample(indices)
            loss = self.policy.learn_on_batch(batch)

    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.config.save_path, 'model.pt'))

    def sample(self, indices):
        i1, i2 = convert_2dindex(indices, self.config.num_envs)
        batch = {}
        for k in self.sample_keys:
            batch[k] = to_tensor(getattr(self.storage, k)[i1, i2], self.config.device)
        return batch

    def eval(self):
        eval_returns = []
        state = to_tensor(self.eval_env.reset(), self.config.device)
        while (len(eval_returns) < self.config.eval_episodes):
            with torch.no_grad():
                action = self.policy.inference(state).cpu().numpy()
            next_state, rwd, done, info = self.env.step(action)
            for (i,d) in enumerate(done):
                if d:
                    eval_returns.append(info[i]['episodic_return'])
            state = to_tensor(next_state, self.config.device)
        self.logger.add_scalar(['eval_returns'], [np.mean(eval_returns)], self.done_steps)
