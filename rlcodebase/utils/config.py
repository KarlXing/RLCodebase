import argparse
import torch
import time
import datetime
import os

class Config:
    def __init__(self):
        self.general_rl_config = ['algo', 'game', 'max_steps', 'num_envs', 'num_frame_stack', 'optimizer', 'lr', 'discount', 
                                  'use_gae', 'gae_lambda', 'use_grad_clip', 'max_grad_norm']
        self.general_exp_config = ['echo_interval', 'num_echo_episodes', 'log_episodes_avg_window', 'save_interval', 'save_path', 'intermediate_eval','eval_interval', 'eval_episodes', 'use_gpu', 'seed', 'tag']

        temp_config = ['value_loss_coef', 'entropy_coef', 'rollout_length', 'memory_on_gpu', 'ppo_epoch', 'ppo_clip_param', 'use_value_clip', 'num_mini_batch', 'mini_batch_size', 'target_kl']
        self.ppo = self.general_rl_config + temp_config + self.general_exp_config

        temp_config = ['value_loss_coef', 'entropy_coef', 'rollout_length', 'memory_on_gpu']
        self.a2c = self.general_rl_config + temp_config + self.general_exp_config

        temp_config = ['replay_size', 'warmup_steps', 'replay_batch', 'memory_on_gpu', 'action_noise', 'soft_update_rate']
        self.ddpg = self.general_rl_config + temp_config + self.general_exp_config

        temp_config = ['replay_size', 'warmup_steps', 'replay_batch', 'memory_on_gpu', 'action_noise', 'soft_update_rate', 'target_noise', 'target_noise_clip', 'policy_delay']
        self.td3 = self.general_rl_config + temp_config + self.general_exp_config

        temp_config = ['replay_size', 'warmup_steps', 'replay_batch', 'memory_on_gpu', 'sac_alpha', 'automatic_alpha', 'soft_update_rate']
        self.sac = self.general_rl_config + temp_config + self.general_exp_config

        temp_config = ['replay_size', 'replay_batch', 'memory_on_gpu',  'exploration_threshold_start', 'exploration_threshold_end', 'exploration_steps', 'target_update_interval', 'learning_start']
        per_config = ['use_per', 'per_alpha', 'per_beta_start', 'per_beta_end', 'per_eps', 'per_max_p']
        self.dqn = self.general_rl_config + temp_config + self.general_exp_config + per_config


        # set default attributes of config from default parser
        default_parser = init_parser()
        args = default_parser.parse_args([])
        for k,v in vars(args).items():
            setattr(self, k, v)

    def __str__(self):
        result = ''
        for k in getattr(self, self.algo):
            result += '%s: %s\n' %  (k, str(getattr(self, k)))
        return result

    def after_set(self):
        self.device = torch.device('cuda') if self.use_gpu and torch.cuda.is_available() else torch.device('cpu')
        self.memory_device = torch.device('cuda') if self.memory_on_gpu and torch.cuda.is_available() else torch.device('cpu')

        if self.save_path == 'default':
            path = '%s-%s-%s' % (self.algo, self.game, datetime.datetime.fromtimestamp(time.time()).strftime('%d.%m.%Y-%H:%M:%S.%f'))
            if self.tag:
                path = '-'.join([path, self.tag])
            self.save_path = os.path.join('./runs', path)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        with open(os.path.join(self.save_path, 'config'), 'w') as f:
            f.write(str(self))

    # user could update config with parsed args
    def update(self, args):
        for k,v in vars(args).items():
            setattr(self, k, v)


def init_parser():
    parser = argparse.ArgumentParser()
    # General RL parameters
    parser.add_argument('--algo',
                        default='a2c', type=str,
                        help='type of rl algos; support dqn, a2c, ppo, ddpg, td3 and sac for now')
    parser.add_argument('--game',
                        default='BreakoutNoFrameskip-v4', type=str,
                        help='name of game')
    parser.add_argument('--max-steps',
                        default=int(1e7), type=int,
                        help='total number of steps to run')
    parser.add_argument('--num-envs',
                        default=16, type=int,
                        help='number of parallel environments')
    parser.add_argument('--num-frame-stack',
                        default=1, type=int,
                        help='number of frames stack')
    parser.add_argument('--optimizer', 
                        default='RMSprop', type=str,
                        help='Optimizer: RMSprop | Adam')
    parser.add_argument('--lr',
                        default=0.0001, type=float,
                        help='learning rate')
    parser.add_argument('--discount',
                        default=0.99, type=float,
                        help='discount factor for rewards')
    parser.add_argument('--use-gae',
                        default=False, action='store_true',
                        help='use generalized advantage estimation or not')
    parser.add_argument('--gae-lambda',
                        default=0.95, type=float,
                        help='lambda parameter used in GAE')
    parser.add_argument('--use-grad-clip',
                        default=False, action='store_true',
                        help='clip gradients or not')
    parser.add_argument('--max-grad-norm',
                        default=5, type=float,
                        help='max norm of gradients')


    # Algo RL parameters
    parser.add_argument('--value-loss-coef',
                        default=0.5, type=float,
                        help='coefficient of value loss')
    parser.add_argument('--entropy-coef',
                        default=0.01, type=float,
                        help='coefficient of entropy loss')
    parser.add_argument('--rollout-length',
                        default=5, type=int,
                        help='number of steps in one forward computation')
    parser.add_argument('--ppo-clip-param',
                        default=0.1, type=float,
                        help='PPO: clip parameter')
    parser.add_argument('--use-value-clip',
                        default=False, action='store_true',
                        help='PPO: clip critic values or not (the clip parameter is the same as ppo policy clip parameter)')
    parser.add_argument('--ppo-epoch',
                        default=4, type=int,
                        help='PPO: number of epochs')
    parser.add_argument('--target-kl',
                        default=None, type=float,
                        help='skip training in PPO if kl divergence between the current dist and original dist goes beyond target_kl')
    parser.add_argument('--num-mini-batch',
                        default=4, type=int,
                        help='PPO: number of mini batches in each epco, mini_batch_size = num_envs * rollout_length / num_mini_batch')
    parser.add_argument('--mini-batch-size',
                        default=2048, type=int,
                        help='PPO: maximum size of one batch of data, if the desired batch size is bigger than mini_batch_size, then gradients accumulation is used')
    parser.add_argument('--replay-size',
                        default=int(1e6), type=int,
                        help='replay memory size for off policy algos')
    parser.add_argument('--warmup-steps',
                        default=int(1e4), type=int,
                        help='warm up steps before sampling actions from policy and do learning')
    parser.add_argument('--replay-batch',
                        default=100, type=int,
                        help='batch size of sampled data from replay memory')
    parser.add_argument('--memory-on-gpu',
                        default=False, action='store_true',
                        help='place replay on gpu or cpu')
    parser.add_argument('--action-noise',
                        default=0.1, type=float,
                        help='std of zero-mean normal distrituion noise added to action')
    parser.add_argument('--soft-update-rate',
                        default=0.005, type=float,
                        help='soft update rate for synchronize target network with training network')
    parser.add_argument('--target-noise',
                        default=0.2, type=float,
                        help='std of zero-mean normal distrituion noise added to next action when evaluating target q value')
    parser.add_argument('--target-noise-clip',
                        default=0.5, type=float,
                        help='limit for absolute value of td3-target-noise')
    parser.add_argument('--policy-delay',
                        default=2, type=int,
                        help='the delay of policy update compared to value update')
    parser.add_argument('--sac-alpha',
                        default=0.2, type=int,
                        help='alpha parameter to control the exploration of soft actor-critic')
    parser.add_argument('--automatic-alpha',
                        default=False, action='store_true',
                        help='automatically adjusting the alpha parameter of soft actor-critic')
    parser.add_argument('--exploration-threshold-start',
                        default=1, type=float,
                        help='the beginning threshold of dqn exploration')
    parser.add_argument('--exploration-threshold-end',
                        default=0.01, type=float,
                        help='the end threshold of dqn exploration')
    parser.add_argument('--exploration-steps',
                        default=int(1e6), type=float,
                        help='the number of steps for dqn exploration, the threshold decreases from start to end linearly within steps')
    parser.add_argument('--target-update-interval',
                        default=int(1e4), type=int,
                        help='the interval of synchronizing the target network')
    parser.add_argument('--learning-start',
                        default=int(5e4), type=int,
                        help='the number of steps before the start of learning')
    parser.add_argument('--use-per',
                        default=False, action='store_true',
                        help='use prioritized experience replay or not')
    parser.add_argument('--per-alpha',
                        default=0.5, type=float,
                        help='alpha for priority calculation in PER')
    parser.add_argument('--per-beta-start',
                        default=0.4, type=float,
                        help='the initial value of beta in importance sampling of PER')
    parser.add_argument('--per-beta-end',
                        default=1, type=float,
                        help='the final value of beta in importance sampling of PER')
    parser.add_argument('--per-eps',
                        default=0.01, type=float,
                        help='the small value added to td error in priority calculation')
    parser.add_argument('--per-max-p',
                        default=1, type=float,
                        help='the maximum priority in PER')


    # General Experiment Config
    parser.add_argument('--echo-interval',
                        default=int(1e4), type=int,
                        help='the interval of printing average episodic return of recent episodes, set as 0 to avoid printing')
    parser.add_argument('--num-echo-episodes',
                        default=20, type=int,
                        help='the number of recent episodes whose episodic return will be averaged and then printed')
    parser.add_argument('--log-episodes-avg-window',
                        default=-1, type=int, 
                        help='log the average episodic return within each window of global steps. -1 means log every episodic return')
    parser.add_argument('--save-interval',
                        default=None, type=int,
                        help='number of steps between two saving')
    parser.add_argument('--save-path',
                        default='default', type=str,
                        help='directory to save models and also the logs')
    parser.add_argument('--intermediate-eval',
                        default=False, action='store_true',
                        help='do evaluation during the training; designed for off policy algos; need eval_env')
    parser.add_argument('--eval-interval',
                        default=int(1e4), type=int,
                        help='number of steps between two evaluation; used only when intermediate_eval is True')
    parser.add_argument('--eval-episodes',
                        default=20, type=int,
                        help='number of episodes for intermediate evaluation')
    parser.add_argument('--use-gpu',
                        default=False, action='store_true',
                        help='use gpu or cpu')
    parser.add_argument('--seed',
                        default=1, type=int,
                        help='random seed for numpy and torch')
    parser.add_argument('--tag',
                        default=None, type=str,
                        help='tag used in creating default save path; easier for user to distinguish saved results')
    return parser