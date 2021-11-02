import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.agent import A2CAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import CatACConvNet
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--game', default='BreakoutNoFrameskip-v4', type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

def main():
    # create config with basic parameters for a2c
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
    config.num_frame_stack = 4
    config.seed = 1

    # update config with argparse object
    config.update(args)
    # add extra tag to the log/save file name (log/save file name includes game and algo information in default)
    config.tag = 'seed%d' % (config.seed)
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
