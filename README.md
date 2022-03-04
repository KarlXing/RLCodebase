# RLCodebase
RLCodebase is a modularized codebase for deep reinforcement learning algorithms based on PyTorch. This repo aims to provide an user-friendly reinforcement learning codebase for beginners to get started and for researchers to try their ideas quickly and efficiently. 

For now, it has implemented DQN(PER), A2C, PPO, DDPG, TD3 and SAC algorithms, and has been tested on Atari, Procgen, Mujoco, PyBullet and DMControl Suite environments.

## Introduction
The design of RLCodebase is shown as below. 


![RLCodebase](imgs/RLCodebase.png)
* **Config**: Config is a class that contains parameters for reinforcement learning algorithms such as discount factor, learning rate, etc. and general configurations such as random seed, saving path, etc.
* **Trainer**: Trainer is a wrapped class that controls the workflow of reinforcement learning training. It manages the interactions between submodules (Agent, Env, memory). 
* **Agent**: Agent chooses actions to take given states. It also defines how to update the model given a batch of data.
* **Model**: Model gathers all neural networks to train.
* **Env**: Env is a vectorized gym environment. 
* **Memory**: Memory stores experiences utilized for RL training.

## Installtion
All required packages have been included in setup.py and requirements.txt. Mujoco is needed for mujoco_py and dm control suite. To support mujoco_py and dm control, please refer to https://github.com/openai/mujoco-py and https://github.com/deepmind/dm_control. For mujoco_py 2.1.2.14 and dm_control (commit fe44496), you may download mujoco like below

````
cd ~  
mkdir .mujoco  
cd .mujoco  
# for mujoco_py
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz  
# for dm control
wget https://github.com/deepmind/mujoco/releases/download/2.1.1/mujoco-2.1.1-linux-x86_64.tar.gz
tar -xf mujoco-2.1.1-linux-x86_64.tar.gz
````


To install RLCodebase, follow
````
# create virtual env
conda create -n rlcodebase python=3.8
conda activate rlcodebase

# install rlcodebase
git clone git@github.com:KarlXing/RLCodebase.git RLCodebase
cd RLCodebase
pip install -e .
pip install -r requirements.txt

# try it
python examples/example_ppo.py
````

## Supported Algorithms
* DQN (PER)
* A2C
* PPO
* DDPG
* TD3
* SAC

## Supported Environments (tested)
* Atari 
* Mujoco
* PyBullet
* Procgen

## Results
### 1. PPO & A2C In Atari Games
<img src="https://github.com/KarlXing/RLCodebase/blob/master/imgs/A2C&PPO.png">

### 2. DDPG & TD3 & SAC In PyBullet Environments
<img src="https://github.com/KarlXing/RLCodebase/blob/master/imgs/DDPG&TD3&SAC.png">

### 3. DQN & DQN+PER In PongNoFrameskip-v4
<img src="https://github.com/KarlXing/RLCodebase/blob/master/imgs/DQN&DQN+PER.png" width="500" class="center">  

### 4. Procgen
<img src="https://github.com/KarlXing/RLCodebase/blob/master/imgs/procgen.png">  


## Citation
Please use the bibtex below if you want to cite this repository in your publications:
````
@misc{rlcodebase,
  author = {Jinwei Xing},
  title = {RLCodebase: PyTorch Codebase For Deep Reinforcement Learning Algorithms},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/KarlXing/RLCodebase}},
}
````


## References for implementation and design
RLCodebase is inspired by resources below.
* https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
* https://github.com/ShangtongZhang/DeepRL
* https://github.com/ray-project/ray/tree/master/rllib
