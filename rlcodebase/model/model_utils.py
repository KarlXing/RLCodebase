import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def orthogonal_init(module, gain = 1):
    nn.init.orthogonal_(module.weight.data, gain = gain)
    nn.init.constant_(module.bias.data, 0)
    return module

def xavier_uniform_init(module, gain=1):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight.data, gain = gain)
        nn.init.constant_(module.bias.data, 0)
    return module

class ConvBody(nn.Module):
    def __init__(self, input_channels = 1, hidden_size = 512, flatten_size = 64*7*7):
        super(ConvBody, self).__init__()

        self.main = nn.Sequential(
            orthogonal_init(nn.Conv2d(input_channels, 32, 8, stride=4), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            orthogonal_init(nn.Conv2d(32, 64, 4, stride=2), gain=nn.init.calculate_gain('relu')), nn.ReLU(),
            orthogonal_init(nn.Conv2d(64, 64, 3, stride=1), gain=nn.init.calculate_gain('relu')), nn.ReLU(), Flatten(),
            orthogonal_init(nn.Linear(flatten_size, hidden_size), gain=nn.init.calculate_gain('relu')), nn.ReLU())

    def forward(self, x):
        return self.main(x)


def make_linear_model(layer_size, intermediate_activation=nn.ReLU, output_activation=nn.Identity, output_gain=1):
    layers = nn.ModuleList([])
    for i in range(len(layer_size)-2):
        layers.append(orthogonal_init(nn.Linear(layer_size[i], layer_size[i+1])))
        layers.append(intermediate_activation())
    layers.append(orthogonal_init(nn.Linear(layer_size[-2], layer_size[-1]), gain=output_gain))
    layers.append(output_activation())
    return nn.Sequential(*layers)

