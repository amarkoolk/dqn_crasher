import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import random
import json
import math
import numpy as np
from typing import TypeAlias, List, Tuple
from collections import namedtuple

from buffers import ReplayMemory, PrioritizedExperienceReplay, Transition

from tqdm import tqdm
import wandb
import gymnasium as gym
from dqn_agent import DQN, DQN_Agent
from wandb_logging import initialize_logging
from config import load_config
from create_env import make_env, make_vector_env
import tyro
from arguments import Args


def multi_agent_training_loop(env_config, args, device, use_pbar = True):

    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None
    
    env = make_vector_env(env_config, args.num_envs)
    ego_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True)
    ego_agent.load_model(path = 'ego_model.pth')
    npc_agent = DQN_Agent(env, args, device, save_trajectories = args.save_trajectories, multi_agent=True)

    if args.track:
        if wandb.run is not None:
            wandb.finish()
            run = initialize_logging(args)
        else:
            run = initialize_logging(args)

    
    num_crashes = []
    episode_rewards = np.zeros(args.num_envs)
    duration = np.zeros(args.num_envs)
    episode_speed = np.zeros(args.num_envs)
    ep_rew_total = np.zeros(0)
    ep_len_total = np.zeros(0)
    ep_speed_total = np.zeros(0)

    t_step = 0
    ep_num = 0


    obs, info = env.reset()
    ego_state = torch.tensor(obs[0].reshape(args.num_envs,ego_agent.n_observations), dtype=torch.float32, device=device)
    npc_state = torch.tensor(obs[1].reshape(args.num_envs,npc_agent.n_observations), dtype=torch.float32, device=device)
    
    # Testing Loop
    while t_step < args.total_timesteps:
        ego_action = torch.squeeze(ego_agent.predict(ego_state))
        npc_action = torch.squeeze(npc_agent.select_action(npc_state, env, t_step))
        obs, reward, terminated, truncated, info = env.step((ego_action, npc_action))

        reward = torch.tensor(reward, dtype = torch.float32, device=device)
        done = terminated | truncated

        ego_state = torch.tensor(obs[0].reshape(args.num_envs,ego_agent.n_observations), dtype=torch.float32, device=device)
        npc_state = npc_agent.update(npc_state, npc_action, obs[1], reward, terminated)

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration = duration + np.ones(args.num_envs)
        episode_speed = episode_speed + np.linalg.norm(ego_state[:,3:5].cpu().numpy(), axis=1)


        for worker in range(args.num_envs):
            if done[worker]:
                # Save Trajectories that end in a Crash
                # if self.save_trajectories:
                #     if info['final_info'][worker]['crashed']:
                #         self.trajectory_store.save(worker, ep_num)
                #     else:
                #         self.trajectory_store.clear(worker)

                num_crashes.append(float(info['final_info'][worker]['crashed']))
                if args.track:
                    ep_rew_total = np.append(ep_rew_total, episode_rewards[worker])
                    ep_len_total = np.append(ep_len_total, duration[worker])
                    ep_speed_total = np.append(ep_speed_total, episode_speed[worker]/duration[worker])
                    if ep_rew_total.size > 100:
                        ep_rew_total = np.delete(ep_rew_total, 0)
                    if ep_len_total.size > 100:
                        ep_len_total = np.delete(ep_len_total, 0)
                    if ep_speed_total.size > 100:
                        ep_speed_total = np.delete(ep_speed_total, 0)

                    wandb.log({"rollout/ep_rew_mean": ep_rew_total.mean(),
                            "rollout/ep_len_mean": ep_len_total.mean(),
                            "rollout/num_crashes": num_crashes[-1],
                            "rollout/sr100": np.mean(num_crashes[-100:]),
                            "rollout/ego_speed_mean": ep_speed_total.mean()},
                            step = ep_num)

                episode_rewards[worker] = 0
                duration[worker] = 0
                episode_speed[worker] = 0
                ep_num += 1

            t_step += 1
            pbar.update(1)


    if use_pbar:
        pbar.close()
    env.close()

if __name__ == "__main__":

    args = tyro.cli(Args)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    else:
        device = torch.device("cpu")
    
    ma_config = load_config("env_configs/multi_agent.yaml")
    multi_agent_training_loop(ma_config, args, device)