import torch
import os
import random
import numpy as np

from .base_trainer import BaseTrainer
from ..memory import Replay, PrioritizedReplay
from ..utils import to_numpy, to_tensor, convert_2dindex, get_threshold, update_maxstep_done
from ..agent import DQNAgent


class DQNTrainer(BaseTrainer):
    def __init__(self, config, env, eval_env, model, target_model, logger):
        super().__init__(config)
        self.agent = DQNAgent(model,
                                target_model,
                                config.discount,
                                config.optimizer,
                                config.lr,
                                config.use_grad_clip,
                                config.max_grad_norm)
        self.env = env
        self.eval_env = eval_env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        if self.config.use_per:
            self.storage = PrioritizedReplay(config.replay_size, 
                                             config.num_envs, 
                                             env.observation_space, 
                                             env.action_space, 
                                             config.memory_device,
                                             config.per_max_p,
                                             config.per_alpha,
                                             config.per_eps,
                                             config.per_beta_start,
                                             config.per_beta_end)
        else:
            self.storage = Replay(config.replay_size, config.num_envs, env.observation_space, env.action_space, config.memory_device)
        self.sample_keys = ['s', 'a', 'r', 'd', 'next_s']

    def step(self):
        self.random_action_threshold = get_threshold(self.config.exploration_threshold_start, 
                                                     self.config.exploration_threshold_end, 
                                                     self.config.exploration_steps,
                                                     self.done_steps)
        with torch.no_grad():
            q = self.agent.inference(self.state).cpu().numpy()
            greedy_action = np.argmax(q, axis=-1)
            random_action = np.random.randint(q.shape[1], size=q.shape[0])
            random_val = np.random.rand(q.shape[0])
            action = np.where(random_val > self.random_action_threshold, greedy_action, random_action)

        next_state, rwd, done, info = self.env.step(action)
        done = update_maxstep_done(info, done, self.env.max_episode_steps)
        self.storage.add({'s': to_tensor(self.state, self.config.memory_device),
                          'a': to_tensor(action, self.config.memory_device),
                          'r': to_tensor(rwd, self.config.memory_device),
                          'd': to_tensor(done, self.config.memory_device),
                          'next_s': to_tensor(next_state, self.config.memory_device)})
        self.logger.save_episodic_return(info, self.done_steps)

        self.state = to_tensor(next_state, self.config.device)

        if self.done_steps > self.config.learning_start:
            batch = self.storage.sample(self.config.replay_batch, self.sample_keys, self.config.device)
            loss, td_error = self.agent.learn_on_batch(batch)
            self.logger.add_scalar(['q_loss'], loss, self.done_steps)
            self.logger.add_scalar(['td_error'], [td_error.mean().item()], self.done_steps)
            if self.config.use_per:
                self.storage.update_p(batch['indices'], td_error)

        if self.done_steps % self.config.target_update_interval == 0:
            self.agent.update_target()

        if self.config.use_per:
            self.storage.update_beta(self.config.max_steps, self.done_steps)

    def save(self):
        torch.save(self.agent.model.state_dict(), os.path.join(self.config.save_path, '%d-model.pt' % self.done_steps))

    def eval(self):
        eval_returns = []
        state = to_tensor(self.eval_env.reset(), self.config.device)
        while (len(eval_returns) < self.config.eval_episodes):
            with torch.no_grad():
                q = self.agent.inference(state).cpu().numpy()
                action = np.argmax(q, axis=-1)
            next_state, rwd, done, info = self.eval_env.step(action)
            for i in info:
                if i['episodic_return'] is not None:
                    eval_returns.append(i['episodic_return'])
            state = to_tensor(next_state, self.config.device)
        self.logger.add_scalar(['eval_returns'], [np.mean(eval_returns)], self.done_steps)

