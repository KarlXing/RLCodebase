import torch
import os

from .base_agent import BaseAgent
from ..memory import Rollout
from ..utils.utils import to_tensor, convert_2dindex
from ..policy.a2c_policy import A2CPolicy


class A2CAgent(BaseAgent):
    def __init__(self, config, env, model, logger):
        super().__init__(config)
        self.policy = A2CPolicy(model,
                                config.optimizer,
                                config.lr,
                                config.value_loss_coef,
                                config.entropy_coef,
                                config.use_grad_clip,
                                config.max_grad_norm)
        self.env = env
        self.state = to_tensor(env.reset(), config.device)
        self.logger = logger
        self.storage = Rollout(config.rollout_length, config.num_envs, env.observation_space, env.action_space, config.device)
        self.sample_keys = ['s', 'a', 'ret', 'adv']
        self.rollout_filled = 0

    def step(self):
        with torch.no_grad():
            action, log_prob, v, ent = self.policy.inference(self.state)
        next_state, rwd, done, info = self.env.step(action.cpu().numpy())
        self.rollout_filled += 1
        self.storage.add({'a': action,
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

            indices = list(range(self.config.rollout_length*self.config.num_envs))
            batch = self.sample(indices)
            loss = self.policy.learn_on_batch(batch)
            self.logger.add_scalar(['action_loss', 'value_loss', 'entropy'], loss, self.done_steps)

            self.rollout_filled = 0
            self.storage.reset()

    def save(self):
        torch.save(self.policy.model.state_dict(), os.path.join(self.config.save_path, 'model.pt'))

    def sample(self, indices):
        i1, i2 = convert_2dindex(indices, self.config.num_envs)
        batch = {}
        for k in self.sample_keys:
            batch[k] = getattr(self.storage, k)[i1, i2]
        return batch








