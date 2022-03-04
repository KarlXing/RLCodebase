import torch
import torch.nn as nn
from .base_agent import BaseAgent


class A2CAgent(BaseAgent):
    def __init__(self, model,
                       optimizer,
                       lr,
                       value_loss_coef,
                       entropy_coef, 
                       use_grad_clip=False, 
                       max_grad_norm=None):
        super().__init__()
        self.model = model
        if optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Only RMSprop and Adam are supported. Please implement here for other optimizers.")

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm

    def inference(self, obs):
        action, action_log_prob, value, entropy = self.model(obs)
        return action, action_log_prob, value, entropy

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