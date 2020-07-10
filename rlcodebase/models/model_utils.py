import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class ConvBody(nn.Module):
    def __init__(self, hidden_size = 512, flatten_size = 32*7*7):
        super(ConvBody, self).__init__()

        self.main = nn.Sequential(
            layer_init(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            layer_init(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            layer_init(nn.Linear(flatten_size, hidden_size)), nn.ReLU())

    def forward(self, x):
        return self.main(x)

