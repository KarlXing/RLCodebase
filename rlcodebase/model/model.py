import torch
import torch.nn as nn
from torch.
from .model_utils import *
from ..utils.distributions import FixedCategorical


class CategoricalActorCriticNet(nn.Module):
    def __init__(self, hidden_size, action_space):
        super(ActorCriticNet, self).__init__()
        self.main = ConvBody(hidden_size = hidden_size)
        self.actor = nn.Sequential(nn.Linear(hidden_size, action_space))
        self.critic = nn.Sequential(nn.Linear(hidden_size, 1))

    def forward(self, x):
        features = self.main(x/255.0)
        logits = self.actor(features)
        self.dist = FixedCategorical(logits)
        action = self.dist.sample()
        action_log_prob = self.dist.log_probs(action)
        entropy = self.dist.entropy()
        value = self.critic(features)

        return action, action_log_prob, value, entropy

    def evaluate_action(self, x, action):
        features = self.main(x/255.0)
        logits = self.actor(features)
        self.dist = FixedCategorical(logits)
        return self.dist.log_probs(action)