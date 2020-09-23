import torch
import torch.nn as nn
from .model_utils import *
from torch.distributions import Categorical

# Categorical-Actor-Critic-Convolutional-Net
class CatACConvNet(nn.Module):
    def __init__(self, input_channels, action_dim,  hidden_size = 512, flatten_size = 64*7*7):
        super(CatACConvNet, self).__init__()
        self.main = ConvBody(input_channels = input_channels, hidden_size = hidden_size, flatten_size = flatten_size)
        self.actor = layer_init(nn.Linear(hidden_size, action_dim), gain = 0.01)
        self.critic = layer_init(nn.Linear(hidden_size, 1))
        self.actor_params = list(self.main.parameters()) + list(self.actor.parameters())
        self.critic_params = list(self.main.parameters()) + list(self.critic.parameters())

    def forward(self, x, action = None):
        actor_features = critic_features = self.main(x/255)
        logits = self.actor(actor_features)
        self.dist = Categorical(logits = logits)
        if action is None:
            action = self.dist.sample()
        action_log_prob = self.dist.log_prob(action)
        entropy = self.dist.entropy()
        value = self.critic(critic_features)

        return action, action_log_prob, value.squeeze(-1), entropy

# Categorical-Actor-Critic-Linear-Net
class CatACLinearNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer = [400,300]):
        super(CatACLinearNet, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer + [action_dim]
        self.actor = make_linear_model(actor_layer_size, output_gain = 0.01)
        critic_layer_size = [input_dim] + hidden_layer + [1]
        self.critic = make_linear_model(critic_layer_size)
        self.actor_params =  list(self.actor.parameters())
        self.critic_params =  list(self.critic.parameters())

    def forward(self, x, action=None):
        logits = self.actor(x)
        self.dist = Categorical(logits = logits)
        if action is None:
            action = self.dist.sample()
        action_log_prob = self.dist.log_prob(action)
        entropy = self.dist.entropy()
        value = self.critic(x)

        return action, action_log_prob, value.squeeze(-1), entropy


# Continuous-Deterministic-Actor-Critic-Convolutional-Net
class ConDetACConvNet(nn.Module):
    def __init__(self, input_channels, action_dim,  hidden_size = 512, flatten_size = 64*7*7):
        super(ConDetACConvNet, self).__init__()
        self.main_actor = ConvBody(input_channels = input_channels, hidden_size = hidden_size, flatten_size = flatten_size)
        self.main_critic = ConvBody(input_channels = input_channels, hidden_size = hidden_size, flatten_size = flatten_size)
        self.actor = layer_init(nn.Linear(hidden_size, action_dim), gain = 0.01)
        self.critic = layer_init(nn.Linear(hidden_size + action_dim, 1), gain = 0.01)
        self.actor_params = list(self.main_actor.parameters()) + list(self.actor.parameters())
        self.critic_params = list(self.main_critic.parameters()) + list(self.critic.parameters())

    def act(self, x):
        return self.actor(self.main_actor(x/255))

    def value(self, x, a):
        feature = self.main_critic(x/255)
        inputs = torch.cat([feature, a], dim=1)
        return self.critic(inputs)

# Continuous-Deterministic-Actor-Critic-Linear-Net
class ConDetACLinearNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer = [400, 300]):
        super(ConDetACLinearNet, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer + [action_dim]
        self.actor = make_linear_model(actor_layer_size, output_activation = nn.Tanh, output_gain = 0.01)
        critic_layer_size = [input_dim + action_dim] + hidden_layer + [1]
        self.critic = make_linear_model(critic_layer_size, output_gain = 0.01)
        self.actor_params =  list(self.actor.parameters())
        self.critic_params =  list(self.critic.parameters())

    def act(self, x):
        return self.actor(x)

    def value(self, x, a):
        inputs = torch.cat([x, a], dim=1)
        return self.critic(inputs)

# Continuous-Deterministic-Actor-Double-Critic-Linear-Net (e.g. TD3)
class ConDetADCLinearNet(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_layer = [400, 300]):
        super(ConDetADCLinearNet, self).__init__()
        actor_layer_size = [input_dim] + hidden_layer + [action_dim]
        self.actor = make_linear_model(actor_layer_size, output_activation = nn.Tanh, output_gain = 0.01)
        critic_layer_size = [input_dim + action_dim] + hidden_layer + [1]
        self.critic1 = make_linear_model(critic_layer_size, output_gain = 0.01)
        self.critic2 = make_linear_model(critic_layer_size, output_gain = 0.01)
        self.actor_params =  list(self.actor.parameters())
        self.critic_params =  list(self.critic1.parameters()) + list(self.critic2.parameters())

    def act(self, x):
        return self.actor(x)

    def value(self, x, a):
        inputs = torch.cat([x, a], dim=1)
        return self.critic1(inputs), self.critic2(inputs)


