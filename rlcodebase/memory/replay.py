import torch
import numpy as np
import random
from ..utils import *

class Replay:
    def __init__(self, replay_size, num_envs, obs_space, action_space, device=torch.device('cpu')):
        self.buffer_size = replay_size // num_envs
        self.current_size = 0
        self.num_envs = num_envs
        self.obs_space = obs_space
        self.obs_shape = obs_space.shape
        self.action_dim = get_action_dim(action_space)
        self.discrete_action = is_discrete_action(action_space)
        self.device = device
        self.reset()

    def add(self, data):
        for k,v in data.items():
            getattr(self, k)[self.pos].copy_(v)
        self.pos = (self.pos + 1) % self.buffer_size
        self.current_size = min(self.current_size+1, self.buffer_size)

    def reset(self):
        self.pos = 0
        self.s = torch.zeros((self.buffer_size, self.num_envs) + self.obs_shape, dtype=convert_dtype(self.obs_space.dtype)).to(self.device)
        self.next_s = torch.zeros((self.buffer_size, self.num_envs) + self.obs_shape, dtype=convert_dtype(self.obs_space.dtype)).to(self.device)
        if self.discrete_action:
            self.a = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float).to(self.device)
        else:
            self.a = torch.zeros((self.buffer_size, self.num_envs, self.action_dim), dtype=torch.float).to(self.device)
        self.r = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float).to(self.device)
        self.d = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float).to(self.device)
        self.v = torch.zeros((self.buffer_size, self.num_envs), dtype=torch.float).to(self.device)

    def sample(self, sample_size, sample_keys, device):
        indices = random.sample(list(range(self.current_size * self.num_envs)), sample_size)
        i1, i2 = convert_2dindex(indices, self.num_envs)
        batch = {}
        for k in sample_keys:
            batch[k] = to_tensor(getattr(self, k)[i1, i2], device)
        return batch


class PrioritizedReplay(Replay):
    def __init__(self, replay_size, 
                       num_envs, 
                       obs_space, 
                       action_space, 
                       device=torch.device('cpu'), 
                       max_p=1, 
                       alpha=0.5, 
                       eps=0.01, 
                       beta_start = 0.4,
                       beta_end = 1):
        super(PrioritizedReplay, self).__init__(replay_size, num_envs, obs_space, action_space, device)
        self.tree = SumTree(replay_size)
        self.max_p = max_p
        self.alpha = alpha
        self.eps = eps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta = beta_start

    def add(self, data):
        # store data
        for k,v in data.items():
            getattr(self, k)[self.pos].copy_(v)

        # update priority
        base_index = self.pos * self.num_envs
        for i in range(self.num_envs):
            self.tree.update(base_index+i, self.max_p)

        self.pos = (self.pos + 1) % self.buffer_size
        self.current_size = min(self.current_size+1, self.buffer_size)

    def sample(self, sample_size, sample_keys, device):
        assert(sample_size > 0)

        # get indices and priority
        segment = self.tree.total() / sample_size
        indices, priority = [], []
        for i in range(sample_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            _, p, data_idx = self.tree.get(s)
            indices.append(data_idx)
            priority.append(p/self.tree.total())

        # prepare data batch with indices and weights of importance sampling
        i1, i2 = convert_2dindex(indices, self.num_envs)
        batch = {}
        for k in sample_keys:
            batch[k] = to_tensor(getattr(self, k)[i1, i2], device)

        batch['indices'] = indices
        priority = np.asarray(priority)
        weights = ((priority*sample_size)+1e-6)**(-self.beta)
        weights = weights/weights.max()
        batch['weights'] = to_tensor(weights, device).unsqueeze(-1)
        return batch

    def update_p(self, indices, td_error):
        if isinstance(td_error, torch.Tensor):
            td_error = td_error.cpu().numpy()
        
        priority = (td_error + self.eps)**(self.alpha)

        for data_idx, p in zip(indices, priority):
            self.tree.update(data_idx, min(p, self.max_p))

    def update_beta(self, max_steps, done_steps):
        self.beta = self.beta_start + (self.beta_end - self.beta_start) * done_steps / max_steps





