import torch
import os
import random
import numpy as np

from .base_trainer import BaseTrainer
from ..memory import Replay 
from ..utils import to_numpy, to_tensor, convert_2dindex, update_maxstep_done
from ..agent import DDPGAgent


class DDPGTrainer(BaseTrainer):
    def __init__(self, config, env, eval_env, model, target_model, logger):
        super().__init__(config)
        self.agent = DDPGAgent(model,
                                target_model,
                                config.discount,
                                config.optimizer,
                                config.lr,
                                config.soft_update_rate)
        self.env = env
        self.eval_env = eval_env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        self.storage = Replay(config.replay_size, config.num_envs, env.observation_space, env.action_space, config.memory_device)
        self.sample_keys = ['s', 'a', 'r', 'd', 'next_s']
        self.action_limit = {'high':self.env.action_space.high[0], 'low':self.env.action_space.low[0]}

    def step(self):
        if self.done_steps < self.config.warmup_steps:
            action = np.array([self.env.action_space.sample() for _ in range(self.config.num_envs)])
        else:
            with torch.no_grad():
                action = self.agent.inference(self.state).cpu().numpy()
                action += np.random.normal(0, self.config.action_noise, action.shape)
                action = np.clip(action, self.action_limit['low'], self.action_limit['high'])
        next_state, rwd, done, info = self.env.step(action)
        done = update_maxstep_done(info, done, self.env.max_episode_steps)
        self.storage.add({'s': to_tensor(self.state, self.config.memory_device),
                          'a': to_tensor(action, self.config.memory_device),
                          'r': to_tensor(rwd, self.config.memory_device),
                          'd': to_tensor(done, self.config.memory_device),
                          'next_s': to_tensor(next_state, self.config.memory_device)})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)

        if self.done_steps > self.config.warmup_steps:
            for i in range(self.config.num_envs):
                batch = self.storage.sample(self.config.replay_batch, self.sample_keys, self.config.device)
                loss = self.agent.learn_on_batch(batch)
                self.logger.add_scalar(['action_loss', 'value_loss'], loss, self.done_steps+i)

    def save(self):
        torch.save(self.agent.model.state_dict(), os.path.join(self.config.save_path, '%d-model.pt' % self.done_steps))

    def eval(self):
        eval_returns = []
        state = to_tensor(self.eval_env.reset(), self.config.device)
        while (len(eval_returns) < self.config.eval_episodes):
            with torch.no_grad():
                action = self.agent.inference(state).cpu().numpy()
            next_state, rwd, done, info = self.eval_env.step(action)
            for (i,d) in enumerate(done):
                if d:
                    eval_returns.append(info[i]['episodic_return'])
            state = to_tensor(next_state, self.config.device)
        self.logger.add_scalar(['eval_returns'], [np.mean(eval_returns)], self.done_steps)
