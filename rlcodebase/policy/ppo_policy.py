import torch
import torch.nn as nn
from .base_policy import BasePolicy


class PPOPolicy(BasePolicy):
    def __init__(self, model,
                       optimizer,
                       lr,
                       value_loss_coef,
                       entropy_coef,
                       ppo_clip_param,
                       use_grad_clip = False,
                       max_grad_norm = None):
        super().__init__(model, optimizer, lr)
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.ppo_clip_param = ppo_clip_param
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm
        self.approx_kl = []

    def compute_actions(self, obs):
        result = self.model(obs)
        return result

    def learn_on_batch(self, batch, target_kl=None):
        state, action, log_prob_old, returns, advantages = batch['s'], batch['a'], batch['log_prob'], batch['ret'], batch['adv']
        _, log_prob, values, entropy = self.model(state, action)
        self.approx_kl.append(torch.mean(log_prob_old - log_prob).detach().cpu().numpy())
        
        ratio = torch.exp(log_prob - log_prob_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.ppo_clip_param,
                                    1.0 + self.ppo_clip_param) * advantages
        action_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), entropy.mean().item()
