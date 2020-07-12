import rlcodebase
from rlcodebase.env.envs import make_vec_envs
from rlcodebase.agent.ppo_agent import PPOAgent
from rlcodebase.utils.config import PPOConfig
from rlcodebase.model.model import CategoricalActorCriticNet

def main():
    model = CategoricalActorCriticNet(input_channels = 1, action_space = 4)
    game = 'BreakoutNoFrameskip-v4'
    env = make_vec_envs(game, num_workers = 3, seed = 1)

    config = PPOConfig()
    config.rollout_length = 10
    config.discount = 0.99
    config.max_steps = 100
    config.num_workers = 3
    config.mini_batch_size = 10
    config.max_grad_norm = 0.5

    agent = PPOAgent(env, config, model)
    for i in range(10):
        agent.step()

if __name__ == '__main__':
    main()


