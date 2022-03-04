# all policies should implement inference and learn_on_batch functions
class BaseAgent():
    def __init__(self):
        pass

    def inference(self, obs):
        raise NotImplementedError

    def learn_on_batch(self, batch):
        raise NotImplementedError