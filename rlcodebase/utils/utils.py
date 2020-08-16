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

def is_discrete_action(action_space):
    return isinstance(action_space, Discrete)


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def convert_2dindex(indices, d):
    idx1, idx2 = [], []
    for i in indices:
        idx1.append(i//d)
        idx2.append(i%d)
    return idx1, idx2