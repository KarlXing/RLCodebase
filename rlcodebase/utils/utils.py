import torch
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


def tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_action_dim(action_space):
    if isinstance(action_space, Discrete):
        return action_space.n
    elif isinstance(action_space, Box):
        return action_space.shape[0]
    else:
        print("Action dim of %s is not implemented yet." % str(type(action_space)))