import torch
import numpy as np
from .config import *


def select_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        Config.DEVICE = torch.device('cuda:%d' % (gpu_id))
    else:
        Config.DEVICE = torch.device('cpu')

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float32)
    x = torch.from_numpy(x).to(Config.DEVICE)
    return x

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)