import torch
import os
import random
import numpy as np

from .base_agent import BaseAgent
from ..memory import Replay 
from ..utils import to_numpy, to_tensor, convert_2dindex, get_threshold
from ..policy import DQNPolicy


class DQNAgent(BaseAgent):
    def __init__(self, config, env, eval_env, model, target_model, logger):
        super().__init__(config)
        self.policy = DQNPolicy(model,
                                target_model,
                                config.discount,
                                config.optimizer,
                                config.lr,
                                config.soft_update_rate,
                                config.use_grad_clip,
                                config.max_grad_norm)
        self.env = env
        self.eval_env = eval_env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        self.storage = Replay(config.replay_size, config.num_envs, env.observation_space, env.action_space, config.device)
        self.sample_keys = ['s', 'a', 'r', 'd', 'next_s']

    def step(self):
        self.random_action_threshold = get_threshold(self.config.exploration_threshold_start, 
                                                     self.config.exploration_threshold_end, 
                                                     self.config.exploration_steps,
                                                     self.done_steps)
        with torch.no_grad():
            q = self.policy.inference(self.state).cpu().numpy()
            greedy_action = np.argmax(q, axis=-1)
            random_action = np.random.randint(q.shape[1], size=q.shape[0])
            random_val = np.random.rand(q.shape[0])
            action = np.where(random_val > self.random_action_threshold, greedy_action, random_action)

        next_state, rwd, done, info = self.env.step(action)
        self.storage.add({'s': self.state,
                          'a': action,
                          'r': rwd,
                          'd': done,
                          'next_s': next_state})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)

        if self.done_steps > self.config.replay_batch:
            indices = random.sample(list(range(self.storage.current_size * self.config.num_envs)), self.config.replay_batch)
            batch = self.sample(indices)
            loss = self.policy.learn_on_batch(batch)
            self.logger.add_scalar(['q_loss'], loss, self.done_steps)

    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.config.save_path, '%d-model.pt' % self.done_steps))

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
                q = self.policy.inference(state).cpu().numpy()
                action = np.argmax(q, axis=-1)
            next_state, rwd, done, info = self.eval_env.step(action)
            for i in info:
                if i['episodic_return'] is not None:
                    eval_returns.append(i['episodic_return'])
            state = to_tensor(next_state, self.config.device)
        self.logger.add_scalar(['eval_returns'], [np.mean(eval_returns)], self.done_steps)

