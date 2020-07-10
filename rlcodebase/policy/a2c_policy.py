import torch
from .base_policy import BasePolicy


class A2CPolicy(BasePolicy):
    def __init__(self, model, config):
        super().__init__(config)

    def computae_actions(self, obs):
        return self.model(obs) 

    def learn_on_batch(self, batch):
        state, action, returns, advantages = batch['s'], batch['a'], batch['ret'], batch['adv']

        values, log_prob, entropy = self.model.evaluate_action(state, action)
        action_loss = log_prob * advantages
        value_loss = 0.5 * (returns - values).pow(2).mean()
        loss = action_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return action_loss.item(), value_loss.item(), entropy.item()