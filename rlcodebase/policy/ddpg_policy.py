import torch
import torch.nn as nn
from .base_policy import BasePolicy


class DDPGPolicy(BasePolicy):
    def __init__(self, model,
                       target_model,
                       discount,
                       optimizer,
                       lr,
                       soft_update_rate):
        super().__init__()
        self.model = model
        self.target_model = target_model
        self.target_model.load_state_dict(self.model.state_dict())
        self.discount = discount
        if optimizer == 'RMSprop':
            self.actor_optimizer = torch.optim.RMSprop(self.model.actor_params, lr=lr)
            self.critic_optimizer = torch.optim.RMSprop(self.model.critic_params, lr=lr)
        else:
            self.actor_optimizer = torch.optim.Adam(self.model.actor_params, lr=lr)
            self.critic_optimizer = torch.optim.Adam(self.model.critic_params, lr=lr)            
        self.soft_update_rate = soft_update_rate 

    def inference(self, obs):
        return self.model.act(obs)

    def learn_on_batch(self, batch):
        state, action, next_state, reward, done = batch['s'], batch['a'], batch['next_s'], batch['r'].unsqueeze(-1), batch['d'].unsqueeze(-1)
        weights = batch['weights'] if 'weights' in batch else torch.ones_like(reward).unsqueeze(-1)
 
        # update critic
        with torch.no_grad():
            next_action = self.target_model.act(next_state)
            target_q = (self.target_model.value(next_state, next_action) * (1-done) * self.discount + reward).detach()
        q = self.model.value(state, action)
        q_loss = ((q - target_q).pow(2)*weights).mean()
        td_error = torch.abs(q.detach() - target_q.detach())

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update policy
        a = self.model.act(state)
        q = self.model.value(state, a)
        a_loss = -q.mean()

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # sync target
        self.soft_update()

        return [a_loss.item(), q_loss.item()], td_error


    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.soft_update_rate) +
                               param * self.soft_update_rate)