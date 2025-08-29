import json
import math
import os
import random
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
import wandb
from arguments import Args
from buffers import PrioritizedExperienceReplay, ReplayMemory, Transition
from config import load_config
from create_env import make_env, make_vector_env
from dqn_agent import DQN, DQN_Agent, TrajectoryStore
from model_pool import ModelPool
from multi_agent_dqn import train_ego
from multi_agent_pool import multi_agent_loop
from tqdm import tqdm
from wandb_logging import initialize_logging

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

    ego_model = "E0_MOBIL.pth"
    npc_model = "NPC_V_MOBIL.pth"
    cycles = args.cycles
    ego_version = 0
    npc_version = 0

    ma_config["adversarial"] = False
    ma_config["normalize_reward"] = True
    ma_config["collision_reward"] = -1
    env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
    trajectory_path = (
        args.trajectories_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_False"
    )
    ego_agent = DQN_Agent(
        env,
        args,
        device,
        save_trajectories=args.save_trajectories,
        multi_agent=True,
        trajectory_path=trajectory_path,
        ego_or_npc="EGO",
        override_obs=10,
    )

    # Random UUID
    run_id = wandb.util.generate_id()
    train_ego(env, ego_agent, args, device, model_path=f"ego_models/train_ego.pth")

    env.close()
