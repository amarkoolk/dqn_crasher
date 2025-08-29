import os
from typing import List, Tuple, TypeAlias

import numpy as np
import torch
import tyro
from arguments import Args
from config import load_config
from create_env import make_vector_env
from dqn_agent import DQN_Agent
from model_pool import ModelPool
from multi_agent_pool import multi_agent_loop

if __name__ == "__main__":
    args = tyro.cli(Args)

    # Determine the computation device
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    args.num_envs = min(args.num_envs, os.cpu_count())
    ma_config = load_config("env_configs/multi_agent.yaml")

    # Initialize ModelPools if using
    if args.use_pool:
        ego_pool = ModelPool(args.sampling, args.adjustable_k)
        npc_pool = ModelPool(args.sampling, args.adjustable_k)
        ego_elo = 1000
        npc_elo = 1000

    ego_model = "E0_MOBIL.pth"
    npc_model = "npc_model.pth"
    cycles = args.cycles
    ego_version = 0
    npc_version = 0

    for cycle in range(cycles):
        train_mode = "train_npc" if cycle % 2 == 0 else "train_ego"
        npc_version += cycle % 2 == 0
        ego_version += cycle % 2 != 0
        trajectory_path = f"{args.trajectories_folder}/E{ego_version}_V{npc_version}_Train{train_mode.split('_')[1].capitalize()}"

        # Update config for training modes
        ma_config["adversarial"] = train_mode == "train_npc"
        ma_config["normalize_reward"] = train_mode != "train_npc"
        ma_config["collision_reward"] = 400 if train_mode == "train_npc" else -1

        if train_mode == "train_ego":
            npc_pool.add_model(npc_model, npc_elo)
        elif train_mode == "train_npc":
            ego_pool.add_model(ego_model, ego_elo)

        if train_mode == "train_ego":
            ma_config["adversarial"] = False
            ma_config["normalize_reward"] = True
            ma_config["collision_reward"] = -1
            train_ego = True
            video_dir = f"videos_{ego_version}_{npc_version}_{train_ego}"
        elif train_mode == "train_npc":
            ma_config["adversarial"] = True
            ma_config["normalize_reward"] = False
            ma_config["collision_reward"] = 400
            train_ego = False
            video_dir = f"videos_{ego_version}_{npc_version}_{train_ego}"
        elif train_mode == "eval":
            video_dir = f"videos_eval_{ego_version}_{npc_version}"
            train_ego = False
        else:
            raise ValueError(
                "Invalid mode. Please choose 'train_ego', 'train_npc', or 'eval'."
            )

        env = make_vector_env(
            ma_config,
            args.num_envs,
            record_video=False,
            record_dir="",
            record_every=100,
        )
        ego_agent = DQN_Agent(
            env,
            args,
            device,
            save_trajectories=args.save_trajectories,
            multi_agent=True,
            trajectory_path=trajectory_path,
        )
        ego_agent.load_model(path=ego_model)
        npc_agent = DQN_Agent(
            env,
            args,
            device,
            save_trajectories=args.save_trajectories,
            multi_agent=True,
        )
        npc_agent.load_model(path=npc_model)

        if train_mode == "train_npc":
            ego_pool.add_model(ego_agent, ego_elo)
        elif train_mode == "train_ego":
            npc_pool.add_model(npc_agent, npc_elo)

        # Run the training or evaluation loop
        multi_agent_loop(
            env=env,
            mode=train_mode,
            train_ego=train_ego,
            ego_version=ego_version,
            npc_version=npc_version,
            ego_agent=ego_agent,
            npc_agent=npc_agent,
            env_config=ma_config,
            args=args,
            device=device,
            trajectory_path=trajectory_path,
            record_video=False,
            use_pbar=True,
            ego_pool=ego_pool if args.use_pool else None,
            npc_pool=npc_pool if args.use_pool else None,
        )

        # Update model paths for the next cycle
        if train_mode == "train_npc":
            npc_model = f"E{ego_version}_V{npc_version}_TrainEgo_False.pth"
        else:
            ego_model = f"E{ego_version}_V{npc_version}_TrainEgo_True.pth"
