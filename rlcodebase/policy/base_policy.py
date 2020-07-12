import torch

class BasePolicy():
    def __init__(self, model, config):
        self.config = config
        self.model = model
        if config.optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=config.lr)
        elif config.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        else:
            raise NotImplementedError("Only RMSprop and Adam are supported. Please implement here for other optimizers.")

    def compute_actions(self, obs):
        raise NotImplementedError

    def learn_on_batch(self, batch):
        raise NotImplementedError