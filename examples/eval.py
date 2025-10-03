import json
import math
import os
import random
import shutil
from collections import namedtuple
from typing import List, Tuple, TypeAlias

import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from arguments import Args
from buffers import PrioritizedExperienceReplay, ReplayMemory, Transition
from config import load_config
from create_env import make_env, make_vector_env
from dqn_agent import DQN, DQN_Agent, TrajectoryStore
from dqn_crasher.utils.model_pool import ModelPool
from multi_agent_dqn import (
    agent_eval,
    agent_vs_mobil,
    ego_vs_npc,
    ego_vs_npc_pool,
    multi_agent_eval,
    multi_agent_training_loop,
    npc_vs_ego,
    npc_vs_ego_pool,
    pool_evaluation,
)
from multi_agent_pool import multi_agent_loop
from tqdm import tqdm
from wandb_logging import initialize_logging

import wandb

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    args.num_envs = min(args.num_envs, os.cpu_count())

    # If model folder not created yet, create it
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    ma_config = load_config("env_configs/multi_agent.yaml")

    ego_model = "sanity-check-bl/E1_V1_TrainEgo_True.pth"
    npc_model = "sanity-check-bl/E0_V1_TrainEgo_False.pth"

    ego_version = 1
    npc_version = 1

    # Check if Trajectories Folder Exists

    trajectories_folder = args.model_folder + "/trajectories_e1v1"
    if not os.path.exists(trajectories_folder):
        os.makedirs(trajectories_folder)

    env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
    ego_agent = DQN_Agent(
        env,
        args,
        device,
        save_trajectories=args.save_trajectories,
        multi_agent=True,
        trajectory_path=trajectories_folder,
        ego_or_npc="EGO",
        override_obs=10,
    )
    npc_agent = DQN_Agent(
        env,
        args,
        device,
        save_trajectories=args.save_trajectories,
        multi_agent=True,
        trajectory_path=trajectories_folder,
        ego_or_npc="NPC",
        override_obs=10,
    )
    ego_agent.load_model(path=ego_model)
    npc_agent.load_model(path=npc_model)
    agent_eval(env, ego_agent, npc_agent, args, device, ego_version, npc_version)
    env.close()
