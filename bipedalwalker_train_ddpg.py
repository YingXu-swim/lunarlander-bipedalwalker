import sys, os
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
os.environ["MUJOCO_GL"] = "glfw" # for windows-enabled rendering
import time
from pathlib import Path

import torch
import gym
import hydra
import wandb
import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

from pg import PG
from ddpg import DDPG
from make_env import create_env
from common import helper as h
from common import logger as logger

def to_numpy(tensor):
    return tensor.cpu().numpy().flatten()

def train(agent, env, max_episode_steps=1000):
    # Run actual training        
    reward_sum, timesteps, done, episode_timesteps = 0, 0, False, 0
    # Reset the environment and observe the initial state
    obs = env.reset()
  
    while not done:
        episode_timesteps += 1
        
        # Sample action from policy
        action, act_logprob = agent.get_action(obs)

        # Perform the action on the environment, get new state and reward
        next_obs, reward, done, _ = env.step(to_numpy(action))

        # Trick: distinguish dead or done
        # if (reward<=-100): done = True
        # else: done = False  

        # Trick
        if reward <= -100: reward = -1

        # Store action's outcome (so that the agent can improve its policy)
        if isinstance(agent, PG):
            done_bool = done
            agent.record(action, reward)
        elif isinstance(agent, DDPG):
            # ignore the time truncated terminal signal
            done_bool = float(done) if episode_timesteps < max_episode_steps else 0 
            agent.record(obs, action, next_obs, reward, done_bool)
        else: raise ValueError

        # Store total episode reward
        reward_sum += reward
        timesteps += 1

        # update observation
        obs = next_obs.copy()

    # update the policy after one episode
    info = agent.update()

    # Return stats of training
    info.update({'timesteps': timesteps,
                'ep_reward': reward_sum,})
    return info


# Function to test a trained policy
@torch.no_grad()
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
            obs, reward, done, info = env.step(to_numpy(action))
            
            test_reward += reward

        total_test_reward += test_reward
        print("Test ep_reward:", test_reward)

    print("Average test reward:", total_test_reward/num_episode)


# The main function
@hydra.main(config_path='configs', config_name='bipedalwalker_easy_ddpg')
def main(cfg):
    # sed seed
    h.set_seed(cfg.seed)
    cfg.run_id = int(time.time())

    # create folders if needed
    work_dir = Path().cwd()/'results'/cfg.env_name
    if cfg.save_logging:
        logging_path = work_dir / 'logging'
        h.make_dir(logging_path)
        L = logger.Logger() # create a simple logger to record stats
    if cfg.save_model:
        model_path = work_dir / 'model'
        h.make_dir(model_path)

    # Model filename
    if cfg.model_path == 'default':
        cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'

    # use wandb to store stats; we aren't currently logging anything into wandb during testing
    """
    if cfg.use_wandb and not cfg.testing:
        wandb.init(project="rl_aalto",
                    name=f'{cfg.exp_name}-{cfg.env_name}-{str(cfg.seed)}-{str(cfg.run_id)}',
                    group=f'{cfg.exp_name}-{cfg.env_name}',
                    config=cfg)
    """
    # create a env
    #env = gym.make(cfg.env_name)
    env = create_env(config_file_name='./env/bipedalwalker_easy', seed=cfg.seed)

    if cfg.save_video:
        # During testing, save every episode
        if cfg.testing:
            ep_trigger = 1
            video_path = work_dir/'video'/'test'
        # During training, save every 50th episode
        else:
            ep_trigger = 50
            video_path = work_dir/'video'/'train'
        env = gym.wrappers.RecordVideo(env, video_path,
                                        episode_trigger=lambda x: x % ep_trigger == 0,
                                        name_prefix=cfg.exp_name) # save video every 50 episode

    state_shape = env.observation_space.shape
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # init agent
    if cfg.agent_name == "pg":
        agent = PG(state_shape, action_dim, cfg.lr, cfg.gamma)
    else: # ddpg
        agent = DDPG(state_shape, action_dim, max_action,
                    cfg.lr, cfg.gamma, cfg.tau, cfg.batch_size, cfg.buffer_size)

    if not cfg.testing: # training
        for ep in range(cfg.train_episodes + 1):
            # collect data and update the policy
            train_info = train(agent, env)

            # if cfg.use_wandb:
            #     wandb.log(train_info)
            if cfg.save_logging:
                L.save(logging_path/'logging.pkl')
            if (not cfg.silent) and (ep % 100 == 0):
                print({"ep": ep, **train_info})

        
        if cfg.save_model:
            agent.save(model_path)

    else: # testing
        if cfg.model_path == 'default':
            cfg.model_path = work_dir/'model'/f'{cfg.env_name}_params.pt'
        print("Loading model from", cfg.model_path, "...")

        # load model
        agent.load(model_path)
        
        print('Testing ...')
        test(agent, env, num_episode=50)


# Entry point of the script
if __name__ == "__main__":
    main()


