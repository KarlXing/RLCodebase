import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.trainer import TD3Trainer
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import ConDetADCLinearNet
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser
import pybullet_envs

parser = ArgumentParser()
parser.add_argument('--game', default='HalfCheetah-v2', type=str)
parser.add_argument('--seed', default=0, type=int)
args = parser.parse_args()

def main():
    # create config with basic parameters for td3
    config = Config()
    config.game = 'HalfCheetah-v2'
    config.algo = 'td3'
    config.max_steps = int(1e6)
    config.num_envs = 1
    config.optimizer = 'Adam'
    config.lr = 0.001
    config.discount = 0.99
    config.replay_size = int(1e6)
    config.replay_batch = 100
    config.replay_on_gpu = True
    config.warmup_steps = 10000
    config.action_noise = 0.1
    config.target_noise = 0.2
    config.target_noise_clip = 0.5
    config.policy_delay = 2
    config.soft_update_rate = 0.005
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
    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed)
    eval_env = make_vec_envs(config.game, num_envs = 1, seed = config.seed)
    model = ConDetADCLinearNet(input_dim = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    target_model = ConDetADCLinearNet(input_dim = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create trainer and run
    trainer = TD3Trainer(config, env, eval_env, model, target_model, logger)
    trainer.run()

if __name__ == '__main__':
    main()
