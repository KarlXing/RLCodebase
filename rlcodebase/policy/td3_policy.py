import torch
import torch.nn as nn
from .base_policy import BasePolicy


class TD3Policy(BasePolicy):
    def __init__(self, model,
                       target_model,
                       discount,
                       optimizer,
                       lr,
                       soft_update_rate,
                       target_noise,
                       target_noise_clip,
                       policy_delay):
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
        self.target_noise = target_noise
        self.target_noise_clip = target_noise_clip
        self.policy_delay = policy_delay
        self.update_count = 0

    def inference(self, obs):
        return self.model.act(obs)

    def learn_on_batch(self, batch, action_limit):
        state, action, next_state, reward, done = batch['s'], batch['a'], batch['next_s'], batch['r'].unsqueeze(-1), batch['d'].unsqueeze(-1)
        weights = batch['weights'] if 'weights' in batch else torch.ones_like(reward).unsqueeze(-1)

        # update critic
        with torch.no_grad():
            next_action = self.target_model.act(next_state)
            noise = torch.randn_like(next_action) * self.target_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action += noise
            next_action = next_action.clamp(action_limit['low'], action_limit['high'])
            target_q1, target_q2 = self.target_model.value(next_state, next_action)
            target_q = (torch.min(target_q1, target_q2) * (1-done) * self.discount + reward).detach()
        q1, q2 = self.model.value(state, action)
        q_loss = ((q1 - target_q).pow(2)*weights).mean() + ((q2 - target_q).pow(2)*weights).mean()

        td_error = torch.abs(q1 - target_q).detach() + torch.abs(q2 - target_q).detach()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update policy
        a_loss = None
        if self.update_count % self.policy_delay == 0:
            a = self.model.act(state)
            q1 = self.model.value(state, a)[0]
            a_loss = -(q1*weights).mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

        # sync target
        self.soft_update()
        self.update_count += 1

        return [a_loss.item() if a_loss is not None else None, q_loss.item()], td_error


    def learn_critic_on_batch(self, batch, action_limit):
        state, action, next_state, reward, done = batch['s'], batch['a'], batch['next_s'], batch['r'].unsqueeze(-1), batch['d'].unsqueeze(-1)
        weights = batch['weights'] if 'weights' in batch else torch.ones_like(reward).unsqueeze(-1)

        # update critic
        with torch.no_grad():
            next_action = self.target_model.act(next_state)
            noise = torch.randn_like(next_action) * self.target_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action += noise
            next_action = next_action.clamp(action_limit['low'], action_limit['high'])
            target_q1, target_q2 = self.target_model.value(next_state, next_action)
            target_q = (torch.min(target_q1, target_q2) * (1-done) * self.discount + reward).detach()
        q1, q2 = self.model.value(state, action)
        q_loss = ((q1 - target_q).pow(2)*weights).mean() + ((q2 - target_q).pow(2)*weights).mean()

        td_error = torch.abs(q1 - target_q).detach() + torch.abs(q2 - target_q).detach()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        return [q_loss.item()], td_error


    def learn_actor_on_batch(self, batch, action_limit):
        state, action, next_state, reward, done = batch['s'], batch['a'], batch['next_s'], batch['r'].unsqueeze(-1), batch['d'].unsqueeze(-1)

        # update policy
        a_loss = None
        if self.update_count % self.policy_delay == 0:
            a = self.model.act(state)
            q1 = self.model.value(state, a)[0]
            a_loss = -(q1).mean()

            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

        # sync target
        self.update_count += 1

        return [a_loss.item() if a_loss is not None else None]


    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.soft_update_rate) +
                               param * self.soft_update_rate)