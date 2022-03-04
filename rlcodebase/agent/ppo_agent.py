import torch
import torch.nn as nn
from .base_agent import BaseAgent


class PPOAgent(BaseAgent):
    def __init__(self, model,
                       optimizer,
                       lr,
                       value_loss_coef,
                       entropy_coef,
                       ppo_clip_param,
                       use_grad_clip = False,
                       max_grad_norm = None,
                       use_value_clip = False):
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
        self.ppo_clip_param = ppo_clip_param
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm
        self.use_value_clip = use_value_clip
        self.approx_kl = []

    def inference(self, obs):
        action, action_log_prob, value, entropy = self.model(obs)
        return action, action_log_prob, value, entropy

    def learn_on_batch(self, batch, accumulation=False):
        state, action, log_prob_old, returns, advantages, old_values = batch['s'], batch['a'], batch['log_prob'], batch['ret'], batch['adv'], batch['v']
        _, log_prob, values, entropy = self.model(state, action)
        self.approx_kl.append(torch.mean(log_prob_old - log_prob).detach().cpu().numpy())
        
        ratio = torch.exp(log_prob - log_prob_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param,
                                    1.0 + self.ppo_clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_value_clip:
            values_clipped = old_values + (values - old_values).clamp(-self.ppo_clip_param, self.ppo_clip_param)
            value_loss1 = (values_clipped - returns).pow(2).mean()
            value_loss2 = (values - returns).pow(2).mean()
            value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        else:
            value_loss = 0.5 * (values - returns).pow(2).mean()
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
        loss.backward()

        # if accumulation, we just accumulate gradients. Otherwise, we update the network based on accumulated gradients
        if not accumulation:
            if self.use_grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return action_loss.item(), value_loss.item(), entropy.mean().item()
