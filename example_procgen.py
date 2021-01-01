import rlcodebase
from rlcodebase.env import make_vec_envs_procgen
from rlcodebase.agent import PPOAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import ImpalaCNN, SeparateImpalaCNN
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--game', default='starpilot', type=str)
parser.add_argument('--num-envs', default=256, type=int)
parser.add_argument('--start-level', default=0, type=int)
parser.add_argument('--num-levels', default=500, type=int)
parser.add_argument('--distribution-mode', default='easy', type=str)
parser.add_argument('--num-frame-stack', default=1, type=int)
parser.add_argument('--separate-actor-critic', default=False, action='store_true')
parser.add_argument('--mini-batch-size', default=2048, type=int, help='reduce this if gpu memory is not enough')
parser.add_argument('--tag', default=None, type=str)
args = parser.parse_args()

def main():
    # create config
    config = Config()
    config.game = args.game
    config.algo = 'ppo'
    config.max_steps = int(1e8)
    config.num_envs = args.num_envs
    config.optimizer = 'Adam'
    config.lr = 0.0005
    config.discount = 0.99
    config.use_gae = True
    config.gae_lambda = 0.95
    config.use_grad_clip = True
    config.max_grad_norm = 0.5
    config.rollout_length = 256
    config.value_loss_coef = 0.5
    config.entropy_coef = 0.01
    config.ppo_epoch = 4
    config.ppo_clip_param = 0.2
    config.num_mini_batch = 8
    config.mini_batch_size = args.mini_batch_size
    config.use_gpu = True
    config.num_frame_stack = args.num_frame_stack

    # update config with argparse object
    config.update(args)
    config.tag = '%s-%s' % (config.game, config.algo)
    if args.tag:
        config.tag += '-' + args.tag 
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs_procgen(env_name = config.game, 
                                num_envs = config.num_envs, 
                                start_level = config.start_level, 
                                num_levels = config.num_levels,
                                distribution_mode = config.distribution_mode,
                                num_frame_stack= config.num_frame_stack)
    Model = SeparateImpalaCNN if args.separate_actor_critic else ImpalaCNN
    model = Model(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = PPOAgent(config, env, model, logger)
    agent.run()

if __name__ == '__main__':
    main()
