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
from multi_agent_dqn import (
    multi_agent_training_loop,
    multi_agent_eval,
    ego_vs_npc_pool,
    npc_vs_ego_pool,
    pool_evaluation,
)
from multi_agent_pool import multi_agent_loop
from wandb_logging import initialize_logging
from config import load_config
from create_env import make_env, make_vector_env

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
    env_config = load_config("env_configs/multi_agent.yaml")
    env_config["use_mobil"] = True
    if args.eval:
        if args.track:
            if wandb.run is not None:
                wandb.finish()
                initialize_logging(
                    args,
                    ego_version="NPC_v_MOBIL",
                    npc_version="NPC_v_MOBIL",
                    train_ego=False,
                    eval=False,
                    npc_pool_size=None,
                    ego_pool_size=None,
                    sampling=None,
                )
            else:
                initialize_logging(
                    args,
                    ego_version="NPC_v_MOBIL",
                    npc_version="NPC_v_MOBIL",
                    train_ego=False,
                    eval=False,
                    npc_pool_size=None,
                    ego_pool_size=None,
                    sampling=None,
                )

        env = make_vector_env(
            env_config,
            args.num_envs,
            record_video=False,
            record_dir="",
            record_every=100,
        )
        npc_agent: DQN_Agent = DQN_Agent(env, args, device, ego_or_npc="NPC")
        npc_agent.test(env, args.total_timesteps)
        env.close()
    else:
        if args.track:
            if wandb.run is not None:
                wandb.finish()
                initialize_logging(
                    args,
                    ego_version="NPC_v_MOBIL",
                    npc_version="NPC_v_MOBIL",
                    train_ego=False,
                    eval=False,
                    npc_pool_size=None,
                    ego_pool_size=None,
                    sampling=None,
                )
            else:
                initialize_logging(
                    args,
                    ego_version="NPC_v_MOBIL",
                    npc_version="NPC_v_MOBIL",
                    train_ego=False,
                    eval=False,
                    npc_pool_size=None,
                    ego_pool_size=None,
                    sampling=None,
                )

        env = make_vector_env(
            env_config,
            args.num_envs,
            record_video=False,
            record_dir="",
            record_every=100,
        )
        npc_agent: DQN_Agent = DQN_Agent(
            env, args, device, ego_or_npc="NPC", multi_agent=True, override_obs=10
        )
        npc_agent.learn(env, args)
        npc_agent.save_model("NPC_v_MOBIL.pth")
        env.close()
