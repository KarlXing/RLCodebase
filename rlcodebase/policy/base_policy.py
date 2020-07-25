import torch

class BasePolicy():
    def __init__(self, model, optimizer, lr):
        self.model = model
        if optimizer == 'RMSprop':
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=lr)
        elif optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise NotImplementedError("Only RMSprop and Adam are supported. Please implement here for other optimizers.")

    def compute_actions(self, obs):
        raise NotImplementedError

    def learn_on_batch(self, batch):
        raise NotImplementedError