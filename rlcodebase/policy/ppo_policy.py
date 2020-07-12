import torch
import torch.nn as nn
from .base_policy import BasePolicy


class PPOPolicy(BasePolicy):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.ppo_clip_param = config.ppo_clip_param
        self.entropy_coef = config.entropy_coef
        self.value_loss_coef = config.value_loss_coef
        self.max_grad_norm = config.max_grad_norm

    def compute_actions(self, obs):
        return self.model(obs)

    def learn_on_batch(self, batch):
        state, action, log_prob_old, returns, advantages = batch['s'], batch['a'], batch['log_prob'], batch['ret'], batch['adv']
        _, log_prob, values, entropy = self.model(state, action)
        ratio = torch.exp(log_prob - log_prob_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param,
                                    1.0 + self.ppo_clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), entropy.item()
