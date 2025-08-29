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
from multi_agent_dqn import (
    multi_agent_training_loop,
    multi_agent_eval,
    ego_vs_npc_pool,
    npc_vs_ego_pool,
    pool_evaluation,
    agent_vs_mobil,
    ego_vs_npc,
    npc_vs_ego,
    agent_eval,
)
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

    ego_model = "ego_models/baseline_20_30/train_ego_0kdk0foe.pth"
    npc_model = ""  # Don't use baseline model to seed weights

    cycles = args.cycles
    ego_version = 0
    npc_version = 0

    # Check if Model Folder Exists
    if not os.path.exists(args.model_folder):
        os.makedirs(args.model_folder)

    # Copy Ego Model to Model Folder
    shutil.copy(
        ego_model,
        os.path.join(
            args.model_folder, f"E{ego_version}_V{npc_version}_TrainEgo_True.pth"
        ),
    )

    # Check if Trajectories Folder Exists
    traj_folder = os.path.join(args.model_folder, args.trajectories_folder)
    if not os.path.exists(traj_folder):
        os.makedirs(traj_folder)

    ma_config["adversarial"] = True
    ma_config["normalize_reward"] = False
    ma_config["collision_reward"] = 400
    env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
    train_ego = False
    trajectory_path = traj_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_True"
    original_ego = DQN_Agent(
        env,
        args,
        device,
        save_trajectories=args.save_trajectories,
        multi_agent=True,
        trajectory_path=trajectory_path,
        ego_or_npc="EGO",
        override_obs=10,
    )
    original_ego.load_model(path=ego_model)
    trajectory_path = traj_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_False"
    npc_agent = DQN_Agent(
        env,
        args,
        device,
        save_trajectories=args.save_trajectories,
        multi_agent=True,
        trajectory_path=trajectory_path,
        ego_or_npc="NPC",
        override_obs=10,
    )
    npc_agent.load_model(path=npc_model)

    # Falsification
    npc_version += 1
    npc_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"
    npc_model = os.path.join(args.model_folder, npc_model)
    npc_vs_ego(
        env, original_ego, npc_agent, args, device, ego_version, npc_version, npc_model
    )
    agent_eval(env, original_ego, npc_agent, args, device, ego_version, npc_version)
    env.close()

    # Hardening
    ma_config["adversarial"] = False
    ma_config["normalize_reward"] = True
    ma_config["collision_reward"] = -100
    env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")

    train_ego = True
    ego_version += 1
    ego_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"
    ego_model = os.path.join(args.model_folder, ego_model)
    ego_vs_npc(
        env, original_ego, npc_agent, args, device, ego_version, npc_version, ego_model
    )
    agent_eval(env, original_ego, npc_agent, args, device, ego_version, npc_version)
    env.close()
