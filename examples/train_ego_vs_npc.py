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
from model_pool import ModelPool
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

    ego_model = "E0_V0_TrainEgo_True.pth"
    npc_model = "NPC_v_MOBIL.pth"
    ego_version = 0
    npc_version = 1

    # Hardening
    ma_config["adversarial"] = False
    ma_config["normalize_reward"] = True
    ma_config["collision_reward"] = -1
    env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")

    trajectory_path = (
        args.trajectories_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_True"
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
    ego_agent.load_model(path=os.path.join(args.model_folder, ego_model))
    trajectory_path = (
        args.trajectories_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_False"
    )
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
    npc_agent.load_model(path=os.path.join(args.model_folder, npc_model))

    train_ego = True
    ego_version = 1
    ego_model = f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth"
    ego_model = os.path.join(args.model_folder, ego_model)
    ego_vs_npc(
        env, ego_agent, npc_agent, args, device, ego_version, npc_version, ego_model
    )
    env.close()
