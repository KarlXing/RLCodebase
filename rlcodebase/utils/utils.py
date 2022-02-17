import torch
import numpy as np
import math
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete


def update_maxstep_done(info, done, max_episode_steps):
    for i in range(len(info)):
        if done[i] and info[i]['eplsodic_len'] >= max_episode_steps:
            done[i] = False
    return done

def to_tensor(x, device):
    if isinstance(x, torch.Tensor):
        return x.type(torch.float).to(device)
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(device)
    return x

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.array(x, dtype=np.float32)

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

def get_threshold(threshold_start, threshold_end, decay_steps, done_steps):
    if done_steps > decay_steps:
        return threshold_end
    else:
        return threshold_start - (threshold_start - threshold_end) * done_steps / decay_steps

def convert_dtype(np_dtype):
    if str(np_dtype) == 'float64':  # replace double with float
        return torch.float
    elif str(np_dtype) == 'float32':
        return torch.float
    elif str(np_dtype) == 'float16':
        return torch.half
    elif str(np_dtype) == 'uint8':
        return torch.uint8
    elif str(np_dtype) == 'int8':
        return torch.int8
    elif str(np_dtype) == 'int16':
        return torch.short
    elif str(np_dtype) == 'int':
        return torch.int
    elif str(np_dtype) == 'int32':
        return torch.long
    else:
        return None