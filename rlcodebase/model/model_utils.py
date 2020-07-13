import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def layer_init(layer, gain = 1):
    nn.init.orthogonal_(layer.weight.data, gain = gain)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class ConvBody(nn.Module):
    def __init__(self, input_channels = 1, hidden_size = 512, flatten_size = 64*7*7):
        super(ConvBody, self).__init__()

        self.main = nn.Sequential(
            layer_init(nn.Conv2d(input_channels, 32, 8, stride=4), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(), Flatten(),
            layer_init(nn.Linear(flatten_size, hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU())

    def forward(self, x):
        return self.main(x)