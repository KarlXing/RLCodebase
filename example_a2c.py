import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.agent import A2CAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import CatACConvNet
from torch.utils.tensorboard import SummaryWriter

def main():
    # create config
    config = Config()
    config.game = 'BreakoutNoFrameskip-v4'
    config.algo = 'a2c'
    config.max_steps = int(2e7)
    config.num_envs = 16
    config.optimizer = 'RMSprop'
    config.lr = 0.0001
    config.discount = 0.99
    config.use_gae = True
    config.gae_lambda = 0.95
    config.use_grad_clip = True
    config.max_grad_norm = 5
    config.rollout_length = 5
    config.value_loss_coef = 0.5
    config.entropy_coef = 0.01
    config.use_gpu = True
    config.seed = 1
    config.num_frame_stack = 4
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed, num_frame_stack= config.num_frame_stack)
    model = CatACConvNet(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = A2CAgent(config, env, model, logger)
    agent.run()

if __name__ == '__main__':
    main()
