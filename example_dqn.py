import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.agent import DQNAgent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import CatQConvNet
from torch.utils.tensorboard import SummaryWriter

def main():
    # create config
    config = Config()
    config.game = 'BreakoutNoFrameskip-v4'
    config.algo = 'dqn'
    config.max_steps = int(2e7)
    config.num_envs = 4
    config.optimizer = 'RMSprop'
    config.lr = 0.00025
    config.discount = 0.99
    config.use_grad_clip = True
    config.max_grad_norm = 5
    config.replay_size = int(1e5)
    config.replay_batch = 32
    config.soft_update_rate = 0.001
    config.exploration_threshold_start = 1
    config.exploration_threshold_end = 0.01
    config.exploration_steps = int(1e6)
    config.intermediate_eval = True
    config.eval_interval = int(1e5)
    config.use_gpu = True
    config.seed = 0
    config.num_frame_stack = 4
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed, num_frame_stack= config.num_frame_stack)
    eval_env = make_vec_envs(config.game, num_envs = 1, seed = config.seed)
    model = CatQConvNet(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    target_model = CatQConvNet(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = DQNAgent(config, env, eval_env, model, target_model, logger)
    agent.run()

if __name__ == '__main__':
    main()
