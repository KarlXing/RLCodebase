import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_policy import BasePolicy


class DQNPolicy(BasePolicy):
    def __init__(self, model,
                       target_model,
                       discount,
                       optimizer,
                       lr,
                       soft_update_rate,
                       use_grad_clip = False,
                       max_grad_norm = None):
        super().__init__()
        self.model = model
        self.target_model = target_model
        self.target_model.load_state_dict(self.model.state_dict())
        self.discount = discount
        if optimizer == 'RMSprop':
            self.q_optimizer = torch.optim.RMSprop(self.model.q_params, lr=lr)
        else:
            self.q_optimizer = torch.optim.Adam(self.model.q_params, lr=lr)
        self.soft_update_rate = soft_update_rate
        self.use_grad_clip = use_grad_clip
        self.max_grad_norm = max_grad_norm

    def inference(self, obs):
        return self.model(obs)

    def learn_on_batch(self, batch):
        state, action, next_state, reward, done = batch['s'], batch['a'], batch['next_s'], batch['r'], batch['d']
        action = action.long()

        # update q net
        with torch.no_grad():
            target_q = self.target_model(next_state).max(1)[0] * (1-done) + reward
        q = self.model(state)
        q = q.gather(1, action.unsqueeze(-1)).squeeze(-1)
        q_loss = F.mse_loss(q, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.use_grad_clip:
            nn.utils.clip_grad_norm_(self.model.q_params, self.max_grad_norm)
        self.q_optimizer.step()

        # sync target
        self.soft_update()

        return [q_loss.item()]


    def soft_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.soft_update_rate) +
                               param * self.soft_update_rate)