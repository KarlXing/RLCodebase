import torch
import torch.nn as nn
from .base_policy import BasePolicy


class SACPolicy(BasePolicy):
    def __init__(self, model,
                       target_model,
                       discount,
                       optimizer,
                       lr,
                       soft_update_rate,
                       alpha,
                       automatic_alpha,
                       action_space = None,
                       device = None):
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
        self.alpha = alpha
        self.automatic_alpha = automatic_alpha
        if self.automatic_alpha:
            assert(action_space is not None)
            assert(device is not None)
            self.target_entropy = -np.prod(action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            if optimizer == 'RMSprop':
                self.alpha_optimizer = torch.optim.RMSprop([self.log_alpha], lr=lr)
            else:
                self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

    def inference(self, obs):
        return self.model.act(obs)

    def learn_on_batch(self, batch):
        state, action, next_state, reward, done = batch['s'], batch['a'], batch['next_s'], batch['r'].unsqueeze(-1), batch['d'].unsqueeze(-1)

        # update critic
        with torch.no_grad():
            next_action, log_p_next, _ = self.model.act(next_state) # here we assume the range of action output is (-1, 1)
            target_q1, target_q2 = self.target_model.value(next_state, next_action)
            target_q = ((torch.min(target_q1, target_q2) - self.alpha * log_p_next) * (1-done) * self.discount + reward).detach()
        q1, q2 = self.model.value(state, action)
        q_loss = (q1 - target_q).pow(2).mean() + (q2 - target_q).pow(2).mean()

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        # update policy
        a, log_p_a, _ = self.model.act(state)
        q1, q2 = self.model.value(state, a)
        q = torch.min(q1, q2)
        a_loss_entropy = (self.alpha * log_p_a).mean()
        a_loss_q = (-q).mean()
        a_loss = a_loss_entropy + a_loss_q

        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        # update alpha
        updated_alpha = None
        if self.automatic_alpha:
            self.alpha_optimizer.zero_grad()
            alpha_loss = -(self.log_alpha * (log_p_a + self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = torch.exp(self.log_alpha)
            updated_alpha = self.alpha.item()

        # sync target
        self.soft_update()

        return a_loss_entropy.item(), a_loss_q.item(), q_loss.item(), updated_alpha


    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.soft_update_rate) +
                               param * self.soft_update_rate)