import rlcodebase
from rlcodebase.env import make_vec_envs
from rlcodebase.agent import TD3Agent
from rlcodebase.utils import get_action_dim, init_parser, Config, Logger
from rlcodebase.model import ConDetADCLinearNet
from torch.utils.tensorboard import SummaryWriter

def main():
    # create config
    config = Config()
    config.game = 'HalfCheetah-v2'
    config.algo = 'td3'
    config.max_steps = int(1e6)
    config.num_envs = 2
    config.optimizer = 'Adam'
    config.lr = 0.001
    config.discount = 0.99
    config.use_gae = True
    config.gae_lambda = 0.95
    config.rollout_length = 5
    config.replay_size = int(1e6)
    config.replay_batch = 100
    config.warmup_steps = 10000
    config.action_noise = 0.1
    config.soft_update_rate = 0.005
    config.intermediate_eval = True
    config.use_gpu = True
    config.seed = 1
    config.after_set()
    print(config)

    # prepare env, model and logger
    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed)
    eval_env = make_vec_envs(config.game, num_envs = 1, seed = config.seed)
    model = ConDetADCLinearNet(input_dim = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    target_model = ConDetADCLinearNet(input_dim = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)
    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    # create agent and run
    agent = TD3Agent(config, env, eval_env, model, target_model, logger)
    agent.run()

if __name__ == '__main__':
    main()
