import argparse
import torch
import time
import os

class Config:
    def __init__(self):
        self.general_rl_config = ['algo', 'game', 'max_steps', 'num_envs', 'num_frame_stack', 'optimizer', 'lr', 'discount', 
                                  'use_gae', 'gae_lambda', 'use_grad_clip', 'max_grad_norm']
        self.general_exp_config = ['echo_interval', 'num_echo_episodes', 'save_interval', 'save_path', 'use_gpu', 'seed', 'eval', 'tag']

        temp_config = ['value_loss_coef', 'entropy_coef', 'rollout_length', 'ppo_epoch', 'ppo_clip_param', 'num_mini_batch', 'target_kl']
        self.ppo = self.general_rl_config + temp_config + self.general_exp_config

        temp_config = ['value_loss_coef', 'entropy_coef', 'rollout_length']
        self.a2c = self.general_rl_config + temp_config + self.general_exp_config

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
    
        if self.save_path == 'default':
            path = '%s-%s-%s' % (self.algo, self.game, time.time())
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
                        help='type of reinforcement learning algorithm; support a2c and ppo for now')
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
                        default=0.0007, type=float,
                        help='learning rate')
    parser.add_argument('--discount',
                        default=0.99, type=float,
                        help='discount factor for rewards')
    parser.add_argument('--use-gae',
                        default=False, action='store_true',
                        help='use generalized advantage estimation or not'
                        )
    parser.add_argument('--gae-lambda',
                        default=0.95, type=float,
                        help='lambda parameter used in GAE')
    parser.add_argument('--use-grad-clip',
                        default=False, action='store_true',
                        help='clip gradients or not')
    parser.add_argument('--max-grad-norm',
                        default=0.5, type=float,
                        help='max norm of gradients')

    # Actor-Critic RL parameters
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
    parser.add_argument('--ppo-epoch',
                        default=4, type=int,
                        help='PPO: number of epochs')
    parser.add_argument('--target-kl',
                        default=None, type=float,
                        help='avoid training in PPO if kl divergence between the current dist and original dist goes beyond target_kl')
    parser.add_argument('--num-mini-batch',
                        default=4, type=int,
                        help='PPO: number of mini batches in each epco, mini_batch_size = num_envs * rollout_length / num_mini_batch')

    # General Experiment Config
    parser.add_argument('--echo-interval',
                        default=int(1e4), type=int,
                        help='the interval of printing average episodic return of recent episodes, set as 0 to avoid printing')
    parser.add_argument('--num-echo-episodes',
                        default=20, type=int,
                        help='the number of recent episodes whose episodic return will be averaged and then printed')
    parser.add_argument('--save-interval',
                        default=int(1e5), type=int,
                        help='number of steps between two saving')
    parser.add_argument('--save-path',
                        default='default', type=str,
                        help='directory to save models and also the logs')
    parser.add_argument('--use-gpu',
                        default=False, action='store_true',
                        help='use gpu or cpu')
    parser.add_argument('--seed',
                        default=1, type=int,
                        help='random seed for numpy and torch')
    parser.add_argument('--eval',
                        default=False, action='store_true',
                        help='evaluate the model without training')
    parser.add_argument('--tag',
                        default=None, type=str,
                        help='tag used in creating default save path; easier for user to distinguish saved results')

    return parser