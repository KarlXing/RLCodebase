import os
import warnings

import gym
import numpy as np
import torch
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env import VecEnvWrapper, VecEnvObservationWrapper
from procgen import ProcgenEnv
import dmc2gym

from ..utils import *


def make_env(env_id, seed, rank, action_repeat_freq=1, rwd_delay=1):
    def _thunk():
        set_random_seed(seed)
        env = gym.make(env_id)

        is_atari = 'atari' in type(env.unwrapped).__name__.lower()
        if is_atari:
            env = make_atari(env_id)
        env.seed(seed + rank)
        env.action_space.seed(seed + rank)

        env = OriginalReturnWrapper(env)

        if is_atari:
            env = wrap_deepmind(env)

        env = ActionRepeatWrapper(env, action_repeat_freq)
        env = RewardDelayWrapper(env, rwd_delay)

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            env = TransposeImage(env)

        return env

    return _thunk

# vec envs based on openai gym
def make_vec_envs(env_name, num_envs, seed=1, num_frame_stack=1, action_repeat_freq=1, rwd_delay=1):
    envs = [make_env(env_name, seed, i, action_repeat_freq, rwd_delay) for i in range(num_envs)]
    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecFrameStack(envs, num_frame_stack)

    temp_env = make_env(env_name, seed, 0)()
    max_episode_steps = (temp_env.spec.max_episode_steps + action_repeat_freq - 1) // action_repeat_freq
    del temp_env
    envs = VecMonitor(envs, max_episode_steps)
    return envs


def make_env_dmcontrol(domain_name, task_name, seed, rank, from_pixels=False, height=84, width=84, action_repeat_freq=1, rwd_delay=1):
    def _thunk():
        set_random_seed(seed)
        env = dmc2gym.make(domain_name=domain_name, task_name=task_name, seed=seed+rank, from_pixels=from_pixels, height=height, width=width, visualize_reward=False, frame_skip=action_repeat_freq)

        env = RewardDelayWrapper(env, rwd_delay)
        env = OriginalReturnWrapper(env)

        env.seed(seed + rank)
        env.action_space.seed(seed + rank)
        return env

    return _thunk

# vec envs for dmcontrol
def make_vec_envs_dmcontrol(domain_name, task_name, num_envs, seed=1, from_pixels=False, height=84, width=84, num_frame_stack=1, action_repeat_freq=1, rwd_delay=1):
    envs = [make_env_dmcontrol(domain_name, task_name, seed, i, from_pixels, height, width, action_repeat_freq, rwd_delay) for i in range(num_envs)]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    envs = VecFrameStack(envs, num_frame_stack)
    temp_env = make_env_dmcontrol(domain_name, task_name, seed, 0, from_pixels, height, width, action_repeat_freq, rwd_delay)()
    max_episode_steps = temp_env.spec.max_episode_steps
    del temp_env
    envs = VecMonitor(envs, max_episode_steps)
    return envs

# vec envs for procgen
def make_vec_envs_procgen(env_name, num_envs, start_level=0, num_levels=0, distribution_mode='hard', normalize_obs=False, normalize_ret=True, num_frame_stack=1):
    env = ProcgenEnv(num_envs=num_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode)
    env = VecExtractDictTransposedObs(env, 'rgb')
    env = VecFrameStack(env, num_frame_stack)
    env = VecMonitor(env)
    env = VecNormalize(env, obs=normalize_obs, ret=normalize_ret)
    return env

class OriginalReturnWrapper(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.total_rewards = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.total_rewards += reward
        if done:
            info['episodic_return'] = self.total_rewards
            self.total_rewards = 0
        else:
            info['episodic_return'] = None
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat_freq=1):
        gym.Wrapper.__init__(self, env)
        self.action_repeat_freq = action_repeat_freq

    def step(self, action):
        action_repeat_rwd = 0
        for _ in range(self.action_repeat_freq):
            obs, reward, done, info = self.env.step(action)
            action_repeat_rwd += reward
            if done:
                break
        reward = action_repeat_rwd
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()

class RewardDelayWrapper(gym.Wrapper):
    def __init__(self, env, rwd_delay=1):
        gym.Wrapper.__init__(self, env)
        self.rwd_delay = rwd_delay
        self.rwd_step = 0
        self.rwd_sum = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        self.rwd_step = (self.rwd_step + 1) % self.rwd_delay
        if self.rwd_step == 0 or done:
            reward += self.rwd_sum
            self.rwd_sum = 0
        else:
            self.rwd_sum += reward
            reward = 0
        
        return obs, reward, done, info

    def reset(self):
        return self.env.reset()


class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)


class VecFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stackedobs = np.zeros((venv.num_envs,) + low.shape, low.dtype)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stackedobs[:, :-self.shape_dim0] = self.stackedobs[:, self.shape_dim0:]
        for (i, new) in enumerate(news):
            if new:
                self.stackedobs[i] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        self.stackedobs[...] = 0
        self.stackedobs[:, -self.shape_dim0:] = obs
        return self.stackedobs

    def close(self):
        self.venv.close()


class VecExtractDictTransposedObs(VecEnvObservationWrapper):
    def __init__(self, venv, key):
        self.key = key
        observation_space = venv.observation_space.spaces[self.key]
        obs_shape = observation_space.shape
        observation_space = Box(
            observation_space.low[0, 0, 0],
            observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=observation_space.dtype)
        super().__init__(venv=venv,
            observation_space=observation_space)

    def process(self, obs):
        return np.transpose(obs[self.key], [0,3,1,2])


class VecMonitor(VecEnvWrapper):
    def __init__(self, venv, max_episode_steps=float('inf')):
        VecEnvWrapper.__init__(self, venv)
        self.episodic_rets = None
        self.episodic_lens = None
        self.max_episode_steps = max_episode_steps

    def reset(self):
        obs = self.venv.reset()
        self.episodic_rets = np.zeros(self.num_envs, 'f')
        self.episodic_lens = np.zeros(self.num_envs, 'i')
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.episodic_rets += rews
        self.episodic_lens += 1

        for i in range(len(dones)):
            if dones[i]:
                info = infos[i]
                info['episodic_return'] = self.episodic_rets[i]
                info['eplsodic_len'] = self.episodic_lens[i]

                self.episodic_rets[i] = 0
                self.episodic_lens[i] = 0
            else:
                info = infos[i]
                info['episodic_return'] = None
                info['eplsodic_len'] = None

        return obs, rews, dones, infos


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, obs=False, ret=False, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)

        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape) if obs else None
        self.ret_rms = RunningMeanStd(shape=()) if ret else None
        
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return obs, rews, news, infos

    def _obfilt(self, obs):
        if self.obs_rms:
            self.obs_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)
