import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import shutil
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
from multi_agent_dqn import multi_agent_training_loop, multi_agent_eval, ego_vs_npc_pool, npc_vs_ego_pool, pool_evaluation, agent_vs_mobil, ego_vs_npc, npc_vs_ego, agent_eval
from multi_agent_pool import multi_agent_loop
from wandb_logging import initialize_logging
from config import load_config
from create_env import make_env, make_vector_env
import highway_env


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

    ego_model = "testingv1/E0_V0_TrainEgo_True.pth"
    npc_model = "NPC_v_MOBIL.pth"

    ego_version = 0
    npc_version = 0

    # Check if Trajectories Folder Exists

    trajectories_folder = args.model_folder + "/trajectories"
    if not os.path.exists(trajectories_folder):
        os.makedirs(trajectories_folder)

    env = gym.make('crash-v0', config=ma_config, render_mode='rgb_array')
    ego_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True, trajectory_path=trajectories_folder, ego_or_npc='EGO', override_obs=10)
    npc_agent = DQN_Agent(env, args, device, save_trajectories=args.save_trajectories, multi_agent=True, trajectory_path=trajectories_folder, ego_or_npc='NPC', override_obs=10)
    ego_agent.load_model(path = ego_model)
    npc_agent.load_model(path = npc_model)
    # agent_eval(env, ego_agent, npc_agent, args, device, ego_version, npc_version)
    agent_vs_mobil(env, ego_agent, args, device)
    env.close()