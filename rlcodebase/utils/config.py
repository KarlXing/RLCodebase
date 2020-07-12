import torch
import argparse

class CommonConfig:
    DEVICE = torch.device('cpu')
    def __init__(self):
        self.optimizer = "RMSprop"
        self.lr = 0.0007
        self.discount = None
        self.gradient_clip = None
        self.use_gae = False
        self.max_steps = 0
        self.rollout_length = None
        self.log_interval = int(1e3)
        self.save_interval = 0
        self.max_grad_norm = None

    def update(self, custom_args):
        for arg in vars(custom_args):
            assert(hasattr(self, arg))
            setattr(self, arg, getattr(args, arg))

class ActorCriticConfig(CommonConfig):
    def __init__(self):
        super().__init__()
        self.entropy_coef = 0.01
        self.value_loss_coef = 0.01
        self.num_workers = 1

class PPOConfig(ActorCriticConfig):
    def __init__(self):
        super().__init__()
        self.ppo_clip_param = 0.2
        self.ppo_epoch = 10
        self.mini_batch_size = 32