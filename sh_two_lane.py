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
from model_pool import ModelPool
from multi_agent_dqn import multi_agent_training_loop, multi_agent_eval, ego_vs_npc_pool, npc_vs_ego_pool, pool_evaluation, agent_vs_mobil
from multi_agent_pool import multi_agent_loop
from wandb_logging import initialize_logging
from config import load_config
from create_env import make_env, make_vector_env

import tyro
from arguments import Args


if __name__ == "__main__":

    args = tyro.cli(Args)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    else:
        device = torch.device("cpu")

    args.num_envs = min(args.num_envs, os.cpu_count())

    # If model folder not created yet, create it
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)
    
    ma_config = load_config("env_configs/multi_agent.yaml")

    if args.eval == False and args.use_pool == False:
        ego_model = "E0_MOBIL.pth"
        npc_model = "NPC_V_MOBIL.pth"

        cycles = args.cycles
        ego_version = 0
        npc_version = 0
        for cycle in range(cycles):
            train_ego = False
            npc_version += 1
            trajectory_path = args.trajectories_folder+ f'/E{ego_version}_V{npc_version}_TrainEgo_{train_ego}'
            multi_agent_training_loop(cycle, ego_version, npc_version, ego_model, npc_model, train_ego, ma_config, args, device, trajectory_path)
            npc_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"

            train_ego = True
            ego_version += 1
            trajectory_path = args.trajectories_folder+ f'/E{ego_version}_V{npc_version}_TrainEgo_{train_ego}'
            multi_agent_training_loop(cycle, ego_version, npc_version, ego_model, npc_model, train_ego, ma_config, args, device, trajectory_path)
            ego_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"

    elif args.eval == False and args.use_pool == True:

        ego_model = "E0_MOBIL.pth"
        # ego_model = ""
        npc_model = "NPC_V_MOBIL.pth"
        cycles = args.cycles
        ego_version = 0
        npc_version = 0

        ego_pool = ModelPool(args.sampling, args.adjustable_k)
        npc_pool = ModelPool(args.sampling, args.adjustable_k)
        ego_elo = 1000
        npc_elo = 1000

        baseline_model = 'mobil'
        ego_pool.add_model(baseline_model, 1000.0, 1000.0)
        if args.sampling == "two_model":
            npc_pool.add_model(baseline_model, 1000.0, 1000.0)

        
        



        for cycle in range(cycles):
            

            ma_config['adversarial'] = False
            ma_config['normalize_reward'] = True
            ma_config['collision_reward'] = -1
            env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
            train_ego = True
            ego_version +=1
            trajectory_path = args.trajectories_folder+ f'/E{ego_version}_V{npc_version}_TrainEgo_True'
            ego_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True, trajectory_path=trajectory_path, ego_or_npc='EGO')
            ego_agent.load_model(path = ego_model)
            trajectory_path = args.trajectories_folder+ f'/E{ego_version}_V{npc_version}_TrainEgo_False'
            npc_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True, trajectory_path=trajectory_path, ego_or_npc='NPC')
            npc_agent.load_model(path = npc_model)
            npc_agent.cycle = cycle

            if cycle == 0:
                npc_pool.add_model(npc_agent, 1000.0, 1000.0)
                print("Evaluation...")
                env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
                pool_evaluation(env, -1, ego_pool, npc_pool, args, device, False)
                env.close()
                print(f"Ego ELOS: {ego_pool.model_elo}")
                print(f"NPC ELOS: {npc_pool.model_elo}")

                env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')

            
            ego_model = os.path.join(args.model_folder, f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth")
            ego_vs_npc_pool(env, ego_agent, npc_pool, args, device, ego_version, npc_version, ego_model)
            ego_elo = npc_pool.opponent_elo
            ego_pool.add_model(ego_agent, 1000.0, 1000.0)
            print(f"EGO ELO: {ego_elo}")

            env.close()
            print("Evaluation...")
            env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
            pool_evaluation(env, cycle+1, ego_pool, npc_pool, args, device, True)
            env.close()

            ego_pool.set_opponent_elo(npc_pool.model_elo[-1])
            npc_pool.set_opponent_elo(ego_pool.model_elo[-1])
            ego_pool.update_probabilities(True)
            npc_pool.update_probabilities(False)

            print(f"Ego ELOS: {ego_pool.model_elo}")
            print(f"NPC ELOS: {npc_pool.model_elo}")

            # Change Environment to Reward Collision Avoidance
            ma_config['adversarial'] = True
            ma_config['normalize_reward'] = False
            ma_config['collision_reward'] = 400
            env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')

            train_ego = False
            npc_version += 1
            npc_model = os.path.join(args.model_folder, f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth")
            npc_vs_ego_pool(env, npc_agent, ego_pool, args, device, ego_version, npc_version, npc_model)
            npc_elo = ego_pool.opponent_elo
            npc_pool.add_model(npc_agent, 1000.0, 1000.0)
            print(f"NPC ELO: {npc_elo}")

            print("Evaluation...")
            env.close()
            env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
            pool_evaluation(env, cycle+1+0.5, ego_pool, npc_pool, args, device, False)

            ego_pool.set_opponent_elo(npc_pool.model_elo[-1])
            npc_pool.set_opponent_elo(ego_pool.model_elo[-1])
            ego_pool.update_probabilities(True)
            npc_pool.update_probabilities(False)

            print(f"Ego ELOS: {ego_pool.model_elo}")
            print(f"NPC ELOS: {npc_pool.model_elo}")

        ego_pool.end_pool()
        npc_pool.end_pool()

        ego_pool.write_model_pool_log(f'ego_pool_adj_bl_{args.start_e}_{args.sampling}.json')
        npc_pool.write_model_pool_log(f'npc_pool_adj_bl_{args.start_e}_{args.sampling}.json')

        env.close()

    else:
        model = "E0_MOBIL.pth"
        model_dir = ""

        ma_config['use_mobil'] = True
        ma_config['ego_vs_mobil'] = True
        ma_config['adversarial'] = False
        ma_config['normalize_reward'] = True
        ma_config['collision_reward'] = -1
        env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
        agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True, ego_or_npc='EGO')
        agent.load_model(path = os.path.join(model_dir, model))

        agent_vs_mobil(env, agent, args, device)