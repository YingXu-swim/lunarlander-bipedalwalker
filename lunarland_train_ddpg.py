import sys, os
sys.path.insert(0, os.path.abspath(".."))
# os.environ["MUJOCO_GL"] = "egl" # for mujoco rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from common import helper as h
from common import logger as logger
import numpy as np
import argparse
# from DDPG import DDPG
from ddpg_edit import DDPG
import torch.nn.functional as F
from make_env import create_env
from itertools import count
# from utils import scale_action
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_numpy(tensor):
    # return tensor.cpu().numpy().flatten()
    # return tensor.detach().numpy().flatten()
    return tensor.cpu().data.numpy().flatten()

def train(agent, env):
    reward_sum, timesteps, done = 0, 0, False
    observation = env.reset()
    while not done:
        action, _ = agent.get_action(observation, evaluation=False)
        """
        action_ = to_numpy(action)
        observation_, reward, done, _ = env.step(action_)
        """
        observation_, reward, done, _ = env.step(action)
        agent.record(observation, action, reward, observation_, done)
        agent.update()
        reward_sum += reward
        timesteps += 1
        observation = observation_

    train_info = {'timesteps': timesteps,
                'ep_reward': reward_sum,
                'last_reward': reward}
    return train_info 

def test(agent, env, num_episode=50):
    total_test_reward = 0
    for ep in range(num_episode):
        obs, done= env.reset(), False
        test_reward = 0

        while not done:
            # Similar to the training loop above -
            # get the action, act on the environment, save total reward
            # (evaluation=True makes the agent always return what it thinks to be
            # the best action - there is no exploration at this point)
            action, _ = agent.get_action(obs, evaluation=True)
            # action_ = scale_action(action.copy(), env.action_space.high, env.action_space.low)
            obs_, reward, done, _ = env.step(action)
            obs = obs_
            test_reward += reward
        print("last reward", reward)

        total_test_reward += test_reward
        print("Test ep_reward:", test_reward)

    print("Average test reward:", total_test_reward/num_episode)


@hydra.main(config_path='configs/', config_name='lunarlander_continuous_easy_ddpg')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())
    # create folders if needed
    work_dir = Path().cwd()/'results'# /f'{cfg.env_name}'
    if cfg.save_model: h.make_dir(work_dir/"model")
    if cfg.save_logging: 
        h.make_dir(work_dir/"logging")
        L = logger.Logger() # create a simple logger to record stats

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'
    """
    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)
    """
    env = create_env(config_file_name='./env/lunarlander_continuous_easy', seed=cfg.seed)

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    agent = DDPG(state_shape=state_shape, 
                action_dim=action_dim,
                max_action = max_action, 
                actor_lr=cfg.actor_lr,
                critic_lr=cfg.critic_lr,
                gamma=cfg.gamma, 
                tau=cfg.tau,
                batch_size=cfg.batch_size,
                buffer_size=1e6)

    if not cfg.testing:# training
        threshold = 200
        reward_history = []
        avg_reward_history = []
        # agent.load(cfg.model_path, ep)# if want to continue training 
        for episode in range(cfg.train_episodes):
            train_info = train(agent, env)

            reward_history.append(train_info['ep_reward'])
            avg_reward = np.mean(reward_history[-100:])
            avg_reward_history.append(avg_reward)

            if cfg.use_wandb:
                wandb.log(train_info)
            if cfg.save_logging:
                L.log(**train_info)
            if (not cfg.silent) and (episode % 100 == 0):
                print("ep", episode, train_info)
            # print({"ep": episode, **train_info})############
            if cfg.save_model and episode in [300, 1000, 1300, 1400, 1500]:
                agent.save(cfg.model_path, episode, cfg.seed)
                print("saved")
        """
        if cfg.save_model:
            agent.save(cfg.model_path, cfg.train_episodes)
            print("finish, model saved")
        """
    else: # testing
        if cfg.model_path == 'default':
            cfg.model_path = work_dir/'model'

        print("Loading model from", cfg.model_path, "...")

        
        agent.load(cfg.model_path, 1500, cfg.seed)
        print('Testing ...')
        test(agent, env, num_episode=50)


if __name__ == '__main__':
    main()


