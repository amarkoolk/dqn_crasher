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
from multi_agent_dqn import multi_agent_training_loop, multi_agent_eval
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
    
    ma_config = load_config("env_configs/multi_agent.yaml")
    ego_model = "E0_MOBIL.pth"
    npc_model = "npc_model.pth"

    cycles = 2
    ego_version = 0
    npc_version = 0
    for cycle in range(cycles):
        train_ego = False
        trajectory_path = args.trajectories_folder+ f'/E{ego_version}_V{npc_version}_TrainEgo_{train_ego}'
        multi_agent_training_loop(cycle, ego_version, npc_version, ego_model, npc_model, train_ego, ma_config, args, device, trajectory_path)
        npc_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"
        npc_version += 1

        train_ego = True
        trajectory_path = args.trajectories_folder+ f'/E{ego_version}_V{npc_version}_TrainEgo_{train_ego}'
        multi_agent_training_loop(cycle, ego_version, npc_version, ego_model, npc_model, train_ego, ma_config, args, device, trajectory_path)
        ego_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"
        ego_version += 1