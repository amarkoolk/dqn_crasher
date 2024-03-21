import wandb
import gymnasium as gym

import tyro
import os
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from arguments import Args
from buffers import ReplayMemory, PrioritizedExperienceReplay, Transition
from create_env import make_env, make_vector_env
from crash_wrappers import CrashResetWrapper, CrashRewardWrapper
from dqn_agent import DQN, DQN_Agent
from config import load_config
from wandb_logging import initialize_logging



# from itertools import count
# import warnings




if __name__ == "__main__":
    # Parse command line arguments
    args = tyro.cli(Args)
    
    # Check Argument Inputs
    assert args.num_envs > 0
    assert args.total_timesteps > 0
    assert args.learning_rate > 0
    assert args.buffer_size > 0
    assert args.gamma > 0
    assert args.tau > 0
    assert args.batch_size > 0
    assert args.start_e > 0
    assert args.buffer_type in ["ER", "PER"]
    assert args.model_type in ["DQN"]

    assert args.max_duration > 0

    # Use wandb to log training runs
    if args.track:
        wandb_run = initialize_logging(args, ego_version=0)

    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    else:
        device = torch.device("cpu")

    # Create Trajectories Folder
    if args.save_trajectories:
        os.makedirs(args.trajectories_folder, exist_ok=True)

    # Load Environment Configurations
    # Non-Adversarial Environment Config
    na_env_cfg = load_config("env_configs/single_agent.yaml")

    # Adversarial Environment Config
    # adv_env_cfg = load_config("env_configs/single_agent_crash.yaml")

    # Multi-Agent Environment Config
    ma_config = load_config("env_configs/multi_agent.yaml")

    # Create Vector Env with Non-Adversarial Rewards
    # na_env = make_vector_env(na_env_cfg, num_envs = args.num_envs, record_video=False, record_dir='na_videos', record_every=100)

    # # # 1. Teach Ego Vehicle to Drive Safely in Highway against Non-Adversarial Vehicle
    # ego_agent = DQN_Agent(na_env, args, device, save_trajectories=args.save_trajectories, multi_agent=False, trajectory_path=args.trajectories_folder+'/E0_MOBIL')

    # # # Load Ego Model
    # if args.load_model:
    #     ego_agent.load_model(path = 'ego_model.pth')

    # # # Learn Ego Model Initially
    # if args.learn:
    #     ego_agent.learn(na_env, args.total_timesteps)
    # na_env.close()

    # # # Save Non-Adversarial Collision Trajectories
    # if args.save_trajectories:
    #     ego_agent.trajectory_store.write(args.trajectories_folder+'/trajectories_E0_MOBIL', 'json')

    # # # Save Ego Model
    # if args.save_model:
    #     ego_agent.save_model(path = 'ego_model.pth')

    # 2. Test Ego Vehicle in Non-Adversarial Environment
    
    ego_model0 = "E0_MOBIL.pth"
    ego_models = [f"E{i}_V{i}_TrainEgo_True.pth" for i in range(1,6)]
    ego_models = [ego_model0] + ego_models
    na_env = gym.make('crash-v0', config=na_env_cfg, render_mode='rgb_array')
    na_env.configure({'adversarial' : False})
    args.num_envs = 1
    for ego_version in range(0,6):
        ego_agent = DQN_Agent(na_env, args, device, adversarial = False, save_trajectories=args.save_trajectories)
        ego_agent.load_model(path = ego_models[ego_version])
        while True:
            done = truncated = False
            obs, info = na_env.reset()
            ego_state = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)
            while not (done or truncated):
                with torch.no_grad():
                    ego_action = torch.argmax(ego_agent.policy_net(ego_state))
                obs, reward, done, truncated, info = na_env.step(ego_action)
                ego_state = torch.tensor(obs.flatten(), dtype=torch.float32, device=device)

    na_env.close()

    # 3. Test Ego Vehicle in Adversarial Environment/ Train Adversarial Agent
    # env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
    # args.num_envs = 1
    # args.track = False
    # ego_agent = DQN_Agent(env, args, device, adversarial = False, save_trajectories=args.save_trajectories, multi_agent=True)
    # ego_agent.load_model(path = 'ego_model.pth')
    # npc_agent = DQN_Agent(env, args, device, adversarial = True, save_trajectories = args.save_trajectories, multi_agent=True)

    # if wandb.run is not None:
    #     wandb.finish()
    #     run = initialize_logging(args)
    # else:
    #     run = initialize_logging(args)

    # num_crashes = []
    # episode_rewards = 0.0
    # duration = 0.0
    # ep_rew_mean = np.zeros(0)
    # ep_len_mean = np.zeros(0)

    # t_step = 0
    # ep_num = 0
    
    # # Testing Loop
    # while True:
    #     done = False
    #     obs, info = env.reset()
    #     ego_state = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=device)
    #     npc_state = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=device)
    #     while not done:
    #         with torch.no_grad():
    #             ego_action = torch.argmax(ego_agent.policy_net(ego_state))
    #         npc_action = torch.squeeze(npc_agent.select_action(npc_state, env, t_step))
    #         obs, reward, terminated, truncated, info = env.step((ego_action, npc_action))
    #         reward = torch.tensor(reward, dtype = torch.float32, device=device)
    #         done = terminated | truncated


    #         ego_state = torch.tensor(obs[0].flatten(), dtype=torch.float32, device=device)

    #         npc_next_state = torch.tensor(obs[1].flatten(), dtype=torch.float32, device=device)
            
    #         if terminated:
    #             npc_agent.memory.push(npc_state.view(1,npc_agent.n_observations), npc_action.view(1,1), None, reward.view(1,1))
    #         else:
    #             npc_agent.memory.push(npc_state.view(1,npc_agent.n_observations), npc_action.view(1,1), npc_next_state.view(1,npc_agent.n_observations), reward.view(1,1))
            
    #         npc_state = npc_next_state

    #         episode_rewards += reward.cpu().numpy()
    #         duration += 1

    #         # Perform one step of the optimization (on the policy network)
    #         npc_agent.optimize_model()

    #         # Soft update of the target network's weights
    #         # θ′ ← τ θ + (1 −τ )θ′
    #         target_net_state_dict = npc_agent.target_net.state_dict()
    #         policy_net_state_dict = npc_agent.policy_net.state_dict()
    #         for key in policy_net_state_dict:
    #             target_net_state_dict[key] = policy_net_state_dict[key]*npc_agent.tau + target_net_state_dict[key]*(1-npc_agent.tau)
    #         npc_agent.target_net.load_state_dict(target_net_state_dict)

    #         if done:
    #             num_crashes.append(float(info['crashed']))
    #             ep_rew_mean = np.append(ep_rew_mean, episode_rewards)
    #             ep_len_mean = np.append(ep_len_mean, duration)
    #             if ep_rew_mean.size > 100:
    #                 ep_rew_mean = np.delete(ep_rew_mean, 0)
    #             if ep_len_mean.size > 100:
    #                 ep_len_mean = np.delete(ep_len_mean, 0)

    #             wandb.log({"rollout/ep_rew_mean": ep_rew_mean.mean(),
    #                     "rollout/ep_len_mean": ep_len_mean.mean(),
    #                     "rollout/num_crashes": num_crashes[-1],
    #                     "rollout/num_crashes_mean": np.mean(num_crashes)},
    #                     step = ep_num)
                
    #             episode_rewards = 0.0
    #             duration = 0.0
    #             ep_num += 1

    #         t_step += 1
    #         env.render()

    # env.close()
    