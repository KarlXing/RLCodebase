import torch
import torch.nn as nn
from .model_utils import *
from torch.distributions import Categorical

class CategoricalActorCriticConvNet(nn.Module):
    def __init__(self, input_channels, action_dim,  hidden_size = 512, flatten_size = 64*7*7):
        super(CategoricalActorCriticConvNet, self).__init__()
        self.main = ConvBody(input_channels = input_channels, hidden_size = hidden_size, flatten_size = flatten_size)
        self.actor = layer_init(nn.Linear(hidden_size, action_dim), gain = 0.01)
        self.critic = layer_init(nn.Linear(hidden_size, 1))

    def forward(self, x, action = None):
        features = self.main(x/255.0)
        logits = self.actor(features)
        self.dist = Categorical(logits = logits)
        if action is None:
            action = self.dist.sample()
        action_log_prob = self.dist.log_prob(action)
        entropy = self.dist.entropy()
        value = self.critic(features)

        return action, action_log_prob, value.squeeze(-1), entropy


class CategoricalActorCriticLinearNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer=[300,400]):
        super(CategoricalActorCriticLinearNet, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer + [action_dim]
        actor_layers = nn.ModuleList([layer_init(nn.Linear(actor_layer_size[i], actor_layer_size[i+1])) for i in range(len(actor_layer_size)-1)])
        self.actor = nn.Sequential(*actor_layers)
    
        critic_layer_size = [input_dim] + hidden_layer + [1]
        critic_layers = nn.ModuleList([layer_init(nn.Linear(critic_layer_size[i], critic_layer_size[i+1])) for i in range(len(critic_layer_size)-1)])
        self.critic = nn.Sequential(*critic_layers)

    def forward(self, x, action=None):
        logits = self.actor(x/255.0)
        self.dist = Categorical(logits = logits)
        if action is None:
            action = self.dist.sample()
        action_log_prob = self.dist.log_prob(action)
        entropy = self.dist.entropy()
        value = self.critic(x/255.0)

        return action, action_log_prob, value.squeeze(-1), entropy