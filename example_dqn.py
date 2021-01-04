import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.agent import DQNAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import CatQConvNet
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--game', default='BreakoutNoFrameskip-v4', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--replay-on-gpu', default=False, action='store_true')
parser.add_argument('--use-per', default=False, action='store_true')
args = parser.parse_args()

def main():
    # create config
    config = Config()
    config.game = 'BreakoutNoFrameskip-v4'
    config.algo = 'dqn'
    config.max_steps = int(1e7)
    config.num_envs = 4
    config.optimizer = 'RMSprop'
    config.lr = 0.00025
    config.discount = 0.99
    config.use_grad_clip = True
    config.max_grad_norm = 5
    config.replay_size = int(1e5)
    config.replay_batch = 32
    config.replay_on_gpu = False
    config.exploration_threshold_start = 1
    config.exploration_threshold_end = 0.01
    config.exploration_steps = int(1e6)
    config.target_update_interval = int(1e4)
    config.learning_start = int(5e4)
    config.intermediate_eval = True
    config.eval_interval = int(1e5)
    config.use_gpu = True
    config.num_frame_stack = 4
    config.seed = 0
    config.log_episodes_avg_window = 10000

    # update config with argparse object (pass game and seed from command line)
    config.update(args)
    config.tag = '%s-%s-%d' % (config.game, config.algo, config.seed)
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed, num_frame_stack= config.num_frame_stack)
    eval_env = make_vec_envs(config.game, num_envs = 1, seed = config.seed, num_frame_stack= config.num_frame_stack)
    model = CatQConvNet(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    target_model = CatQConvNet(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes, config.log_episodes_avg_window)

    # create agent and run
    agent = DQNAgent(config, env, eval_env, model, target_model, logger)
    agent.run()

if __name__ == '__main__':
    main()
