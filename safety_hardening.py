import wandb
import gymnasium as gym

import tyro
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from arguments import Args
from buffers import ReplayMemory, PrioritizedExperienceReplay, Transition
from create_env import make_env, make_vector_env
from models import DQN, DQN_Agent
from config import load_config



# from itertools import count
# import warnings




if __name__ == "__main__":
    # Parse command line arguments
    args = tyro.cli(Args)
    print(args)
    
    # Check Argument Inputs
    assert args.num_envs > 0
    assert args.total_timesteps > 0
    assert args.learning_rate > 0
    assert args.buffer_size > 0
    assert args.gamma > 0
    assert args.tau > 0
    assert args.batch_size > 0
    assert args.start_e > 0
    assert args.buffer_type in ["ER", "PER", "HER"]
    assert args.model_type in ["DQN"]

    assert args.max_duration > 0

    if args.buffer_type == "HER":
        raise NotImplementedError("HER is not implemented yet")

    # Use wandb to log training runs
    if args.track:
        wandb.init(
            # set the wandb project where this run will be logged
            project="rl_crash_course",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.learning_rate,
            "architecture": args.model_type,
            "max_duration": args.max_duration,
            "dataset": "Highway-Env",
            "max_steps": args.total_timesteps,
            "collision_reward": args.crash_reward,
            "ttc_x_reward": args.ttc_x_reward,
            "ttc_y_reward": args.ttc_y_reward,
            "BATCH_SIZE": args.batch_size,
            "GAMMA": args.gamma,
            "EPS_START": args.start_e,
            "EPS_END": args.end_e,
            "EPS_DECAY": args.decay_e,
            "TAU": args.tau,
            "ReplayBuffer": args.buffer_type
            }
        )

    # Load Environment Configurations
    env_config = load_config("env_configs/single_agent.yaml")

    # Create Vector Env with Adversarial Rewards
    env = make_vector_env(env_config, num_envs = args.num_envs, adversarial = args.adversarial)

    # 1. Teach Ego Vehicle to Drive Safely in Highway against Non-Adversarial Vehicle
    # 2. Teach Adversarial Vehicle to Drive Unsafely in Highway against Ego Vehicle



    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    else:
        device = torch.device("cpu")

    # Initialize DQN Agent
    agent = DQN_Agent(env, args, device)
    agent.learn(env, args.total_timesteps)
    env.close()

    # Save the model
    if args.save_model:
        agent.save_model()
