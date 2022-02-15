import rlcodebase
from rlcodebase.env import make_vec_envs_dmcontrol
from rlcodebase.agent import SACAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import ConStoSGADCLinearNet
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--task-name', default='cheetah', type=str)
parser.add_argument('--domain-name', default='run', type=str)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--tag', default='', type=str)
args = parser.parse_args()

def main():
    # create config with basic parameters for sac
    config = Config()
    config.game = args.task_name + '-' + args.domain_name
    config.algo = 'sac'
    config.max_steps = int(1e6)
    config.num_envs = 1
    config.optimizer = 'Adam'
    config.lr = 0.001
    config.discount = 0.99
    config.replay_size = int(1e6)
    config.replay_batch = 100
    config.replay_on_gpu = True
    config.warmup_steps = 10000
    config.soft_update_rate = 0.005
    config.sac_alpha = 0.2
    config.automatic_alpha = True
    config.intermediate_eval = True
    config.eval_interval = int(1e4)
    config.use_gpu = True
    config.seed = 0

    # update config with argparse object (pass game and seed from command line)
    config.update(args)
    # add extra tag to the log/save file name (log/save file name includes game and algo information in default)
    config.tag = 'seed%d' % (config.seed)
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs_dmcontrol(config.task_name, config.domain_name, num_envs = config.num_envs, seed = config.seed, from_pixels=config.from_pixels)
    eval_env = make_vec_envs_dmcontrol(config.task_name, config.domain_name, num_envs = 1, seed = config.seed, from_pixels=config.from_pixels)
    model = ConStoSGADCLinearNet(input_dim = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    target_model = ConStoSGADCLinearNet(input_dim = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = SACAgent(config, env, eval_env, model, target_model, logger)
    agent.run()

if __name__ == '__main__':
    main()
