import torch
import argparse

class CommonConfig:
    DEVICE = torch.device('cpu')
    def __init__(self):
        # Common config for algorithms
        self.optimizer = "RMSprop"
        self.lr = 0.0007
        self.discount = 0.99
        self.use_gae = False
        self.max_steps = int(10e5)
        self.max_grad_norm = 0.5

        # Common config for simulation
        self.save_interval = int(1e5)
        self.save_path = './'
        self.log_interval = int(1e4)
        self.log_path = './'
        self.num_workers = 1
        self.seed = 1

    def update(self, custom_args):
        for arg in vars(custom_args):
            assert(hasattr(self, arg))
            setattr(self, arg, getattr(args, arg))

class ActorCriticConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.01
        self.rollout_length = 32

class PPOConfig(ActorCriticConfig):
    def __init__(self):
        super().__init__()
        self.ppo_clip_param = 0.2
        self.ppo_epoch = 10
        self.mini_batch_size = 32

class A2CConfig(ActorCriticConfig):
    def __init__(self):
        super().__init__()