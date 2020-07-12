import torch
import torch.nn as nn
from .base_policy import BasePolicy


class A2CPolicy(BasePolicy):
    def __init__(self, model, config):
        super().__init__(model, config)
        self.max_grad_norm = config.max_grad_norm
        self.entropy_coef = config.entropy_coef
        self.value_loss_coef = config.value_loss_coef

    def compute_actions(self, obs):
        return self.model(obs) 

    def learn_on_batch(self, batch):
        state, action, returns, advantages = batch['s'], batch['a'], batch['ret'], batch['adv']

        _, log_prob, values, entropy = self.model(state, action)
        action_loss = -(log_prob * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), entropy.mean().item()