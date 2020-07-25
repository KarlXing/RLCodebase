import rlcodebase
from rlcodebase.env.envs import make_vec_envs
from rlcodebase.agent import A2CAgent
from rlcodebase.utils import get_action_dim, init_config, Logger
from rlcodebase.model import CategoricalActorCriticNet
from torch.utils.tensorboard import SummaryWriter

def main():
	# get config
    config = init_config()
    print(config)

    env = make_vec_envs(config.game, num_envs = config.num_envs, seed = config.seed, num_frame_stack= config.num_frame_stack)
    model = CategoricalActorCriticNet(input_channels = env.observation_space.shape[0], action_dim = get_action_dim(env.action_space)).to(config.device)

    logger =  Logger(SummaryWriter(config.save_path), config.num_echo_episodes)

    agent = A2CAgent(config, env, model, logger)
    agent.run()

if __name__ == '__main__':
    main()
