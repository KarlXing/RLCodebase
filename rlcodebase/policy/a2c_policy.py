import torch
import torch.nn as nn
from .base_policy import BasePolicy


class A2CPolicy(BasePolicy):
    def __init__(self, model,
                       optimizer,
                       lr,
                       value_loss_coef,
                       entropy_coef, 
                       use_grad_clip=False, 
                       max_grad_norm=None):
        super().__init__(model, optimizer, lr)
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm

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
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), entropy.mean().item()