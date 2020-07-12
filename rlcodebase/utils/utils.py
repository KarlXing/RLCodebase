import torch
import numpy as np
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from .config import *


def select_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        CommonConfig.DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        CommonConfig.DEVICE = torch.device('cpu')


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(CommonConfig.DEVICE)
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