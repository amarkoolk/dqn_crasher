import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
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
from dqn_agent import DQN, DQN_Agent, TrajectoryStore
from wandb_logging import initialize_logging
from config import load_config
from create_env import make_env, make_vector_env
import tyro

from arguments import Args


def multi_agent_training_loop(cycle, ego_model, npc_model, train_ego, env_config, args, device, trajectory_path, record_video = False, use_pbar = True):

    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None

    if train_ego:
        env_config['adversarial'] = False
        env_config['normalize_reward'] = True
        env_config['collision_reward'] = -1
    else:
        env_config['adversarial'] = True
        env_config['normalize_reward'] = False
        env_config['collision_reward'] = 400

    
    video_dir = f'videos_{cycle}_{train_ego}'
    env = make_vector_env(env_config, args.num_envs, record_video=record_video, record_dir=video_dir, record_every=100)
    ego_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True)
    ego_agent.load_model(path = ego_model)
    npc_agent = DQN_Agent(env, args, device, save_trajectories = args.save_trajectories, multi_agent=True)
    npc_agent.load_model(path = npc_model)

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
        if train_ego:
            ego_action = torch.squeeze(ego_agent.select_action(ego_state, env, t_step))
            npc_action = torch.squeeze(npc_agent.predict(npc_state))
        else:
            ego_action = torch.squeeze(ego_agent.predict(ego_state))
            npc_action = torch.squeeze(npc_agent.select_action(npc_state, env, t_step))

        if args.num_envs == 1:
            ego_action = ego_action.view(1,1)
            npc_action = npc_action.view(1,1)

        obs, reward, terminated, truncated, info = env.step((ego_action, npc_action))

        reward = torch.tensor(reward, dtype = torch.float32, device=device)
        done = terminated | truncated

        if done:
            int_frames = info['final_info'][0]['int_frames']
        else:
            int_frames = info['int_frames'][0]

        if args.save_trajectories:
            for worker in range(args.num_envs):
                if terminated[worker]:
                    ego_agent.trajectory_store.add(worker, Transition(ego_state[worker].cpu().numpy(), ego_action[worker].cpu().numpy(), None, reward[worker].cpu().numpy()), int_frames)
                else:
                    ego_agent.trajectory_store.add(worker, Transition(ego_state[worker].cpu().numpy(), ego_action[worker].cpu().numpy(), obs[0][worker].flatten(), reward[worker].cpu().numpy()), int_frames)


        if train_ego:
            ego_state = ego_agent.update(ego_state, ego_action, obs[0], reward, terminated)
            npc_state = torch.tensor(obs[1].reshape(args.num_envs,npc_agent.n_observations), dtype=torch.float32, device=device)
        else:
            ego_state = torch.tensor(obs[0].reshape(args.num_envs,ego_agent.n_observations), dtype=torch.float32, device=device)
            npc_state = npc_agent.update(npc_state, npc_action, obs[1], reward, terminated)

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration = duration + np.ones(args.num_envs)
        episode_speed = episode_speed + np.linalg.norm(ego_state[:,3:5].cpu().numpy(), axis=1)


        for worker in range(args.num_envs):
            if done[worker]:
                # Save Trajectories that end in a Crash
                if args.save_trajectories:
                    if info['final_info'][worker]['crashed']:
                        ego_agent.trajectory_store.save(worker, ep_num)
                    else:
                        ego_agent.trajectory_store.clear(worker)

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

    if train_ego:
        ego_agent.save_model(path=f'{cycle}_ego_model.pth')
    else:
        npc_agent.save_model(path=f'{cycle}_npc_model.pth')

    if args.save_trajectories:
        ego_agent.trajectory_store.write(trajectory_path, 'json')

    wandb.finish()
    env.close()

def multi_agent_eval(ego_model, npc_model, env_config, args, device, record_video = False, use_pbar = True):

    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None
    
    env = make_env(env_config, record_video, record_dir=f'eval_videos', record_every=10)
    ego_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True)
    ego_agent.load_model(path = ego_model)
    npc_agent = DQN_Agent(env, args, device, save_trajectories = args.save_trajectories, multi_agent=True)
    npc_agent.load_model(path = npc_model)

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
        npc_action = torch.squeeze(npc_agent.predict(npc_state))

        if args.num_envs == 1:
            ego_action = ego_action.view(1,1)
            npc_action = npc_action.view(1,1)
            
        obs, reward, terminated, truncated, info = env.step((ego_action, npc_action))

        reward = torch.tensor(reward, dtype = torch.float32, device=device)
        done = terminated | truncated

        if args.save_trajectories:
            for worker in range(args.num_envs):
                if terminated[worker]:
                    ego_agent.trajectory_store.add(worker, Transition(ego_state[worker].cpu().numpy(), ego_action[worker].cpu().numpy(), None, reward[worker].cpu().numpy()))
                else:
                    ego_agent.trajectory_store.add(worker, Transition(ego_state[worker].cpu().numpy(), ego_action[worker].cpu().numpy(), obs[0][worker].flatten(), reward[worker].cpu().numpy()))


        ego_state = torch.tensor(obs[0].reshape(args.num_envs,ego_agent.n_observations), dtype=torch.float32, device=device)
        npc_state = torch.tensor(obs[1].reshape(args.num_envs,npc_agent.n_observations), dtype=torch.float32, device=device)

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

    wandb.finish()
    env.close()

if __name__ == "__main__":

    args = tyro.cli(Args)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    else:
        device = torch.device("cpu")

    args.num_envs = min(args.num_envs, os.cpu_count())
    
    ma_config = load_config("env_configs/multi_agent.yaml")
    ego_model = "ego_model.pth"
    npc_model = "npc_model.pth"

    cycles = 2
    for cycle in range(cycles):
        trajectory_path = args.trajectories_folder+'/trajectories_ego_' + str(cycle)
        multi_agent_training_loop(cycle, ego_model, npc_model, False, ma_config, args, device, trajectory_path)
        npc_model = f"{cycle}_npc_model.pth"

        trajectory_path = args.trajectories_folder+'/trajectories_npc_' + str(cycle)
        multi_agent_training_loop(cycle, ego_model, npc_model, True, ma_config, args, device, trajectory_path)
        ego_model = f"{cycle}_ego_model.pth"
