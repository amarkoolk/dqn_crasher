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
import wandb
from arguments import Args
from buffers import PrioritizedExperienceReplay, ReplayMemory, Transition
from config import load_config
from create_env import make_env, make_vector_env
from dqn_agent import DQN, DQN_Agent, TrajectoryStore
from model_pool import ModelPool
from multi_agent_dqn import (ego_vs_npc_pool, npc_vs_ego_pool, pool_evaluation,
                             test_agents, train_agents)
from multi_agent_pool import multi_agent_loop
from tqdm import tqdm
from train_agent import train_agent
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

    config = load_config("model_configs/training_config.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config["observation"]["observation_config"]["frame_stack"] = args.frame_stack

    config["gym_config"] = gym_config

    if args.eval == False and args.use_pool == False:
        ego_model = "ego_models/baseline_20_30/train_ego_0kdk0foe.pth"
        # npc_model = "NPC_v_MOBIL.pth"
        npc_model = ""  # Don't use baseline model to seed weights

        cycles = args.cycles
        ego_version = 0
        npc_version = 0

        # Check if Model Folder Exists
        if not os.path.exists(args.model_folder):
            os.makedirs(args.model_folder)

        # Copy Models to Model Folder
        # If no existing models, don't copy
        if ego_model != "":
            shutil.copy(
                ego_model,
                os.path.join(
                    args.model_folder,
                    f"E{ego_version}_V{npc_version}_TrainEgo_True.pth",
                ),
            )
        if npc_model != "":
            shutil.copy(
                npc_model,
                os.path.join(
                    args.model_folder,
                    f"E{ego_version}_V{npc_version}_TrainEgo_False.pth",
                ),
            )

        # Check if Trajectories Folder Exists
        traj_folder = os.path.join(args.model_folder, args.trajectories_folder)
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)

        config["env_name"] = "crash-v0"
        config["ego_version"] = ego_version
        config["npc_version"] = npc_version

        for cycle in range(cycles):
            gym_config["adversarial"] = True
            gym_config["normalize_reward"] = False
            gym_config["collision_reward"] = 400
            config["gym_config"] = gym_config
            config["train_ego"] = False
            config["trajectory_path"] = (
                traj_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_True"
            )
            config["ego_model"] = os.path.join(
                args.model_folder, f"E{ego_version}_V{npc_version}_TrainEgo_True.pth"
            )
            config["npc_model"] = os.path.join(
                args.model_folder, f"E{ego_version}_V{npc_version}_TrainEgo_False.pth"
            )
            config["npc_version"] += 1

            train_agent(config)

            # Hardening
            gym_config["adversarial"] = False
            gym_config["normalize_reward"] = True
            gym_config["collision_reward"] = -100
            config["gym_config"] = gym_config
            config["train_ego"] = True
            config["trajectory_path"] = (
                traj_folder + f"/E{ego_version}_V{npc_version}_TrainEgo_False"
            )
            config["ego_version"] += 1

            train_agent(config)

    elif args.eval == True and args.use_pool == False:
        model_folder = args.model_folder
        cycles = args.cycles

        # Check if Trajectories Folder Exists

        trajectories_folder = args.model_folder + "/trajectories"
        if not os.path.exists(trajectories_folder):
            os.makedirs(trajectories_folder)

        # Grab all models in the model folder and Parse by False/True String
        models = os.listdir(model_folder)
        ego_models = [model for model in models if "True" in model]
        npc_models = [model for model in models if "False" in model]

        # Sort Models by Version (Ego by E#, NPC by V#)
        ego_models.sort(key=lambda x: int(x.split("_")[0][1:]))
        npc_models.sort(key=lambda x: int(x.split("_")[1][1:]))

        # Evaluate Each Model Pair
        for ego_version, ego_model in enumerate(ego_models):
            for npc_version, npc_model in enumerate(npc_models):
                print(f"Ego Mode:l: {ego_model}, Ego Version: {ego_version}")
                print(f"NPC Model: {npc_model}, NPC Version: {npc_version}")
                env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")

                action_space = env.action_space[0]
                n_actions = action_space[0].n
                n_obs = 10 * args.frame_stack

                trajectory_path = (
                    trajectories_folder + f"/E{ego_version}_V{npc_version}_Eval"
                )
                ego_agent = DQN_Agent(
                    n_obs,
                    n_actions,
                    action_space,
                    args,
                    device,
                    trajectory_path=trajectory_path,
                    ego_or_npc="EGO",
                    override_obs=10,
                )
                npc_agent = DQN_Agent(
                    n_obs,
                    n_actions,
                    action_space,
                    args,
                    device,
                    trajectory_path=trajectory_path,
                    ego_or_npc="NPC",
                    override_obs=10,
                )
                ego_agent.load_model(path=os.path.join(model_folder, ego_model))
                npc_agent.load_model(path=os.path.join(model_folder, npc_model))
                test_agents(
                    env, ego_agent, npc_agent, args, device, ego_version, npc_version
                )
                env.close()

    elif args.eval == False and args.use_pool == True:
        ego_model = "E0_MOBIL.pth"
        # ego_model = ""
        npc_model = "NPC_V_MOBIL.pth"
        cycles = args.cycles
        ego_version = 0
        npc_version = 0

        ego_pool = ModelPool(args.sampling, args.adjustable_k, n_obs=10)
        npc_pool = ModelPool(args.sampling, args.adjustable_k, n_obs=10)
        ego_elo = 1000
        npc_elo = 1000

        baseline_model = "mobil"
        ego_pool.add_model(baseline_model, 1000.0, 1000.0)
        if args.sampling == "two_model":
            npc_pool.add_model(baseline_model, 1000.0, 1000.0)

        for cycle in range(cycles):
            ma_config["adversarial"] = True
            ma_config["normalize_reward"] = False
            ma_config["collision_reward"] = 400

            env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
            trajectory_path = (
                args.trajectories_folder
                + f"/E{ego_version}_V{npc_version}_TrainEgo_True"
            )
            ego_agent = DQN_Agent(
                env,
                args,
                device,
                multi_agent=True,
                trajectory_path=trajectory_path,
                ego_or_npc="EGO",
                override_obs=10,
            )
            ego_agent.load_model(path=ego_model)
            trajectory_path = (
                args.trajectories_folder
                + f"/E{ego_version}_V{npc_version}_TrainEgo_False"
            )
            npc_agent = DQN_Agent(
                env,
                args,
                device,
                multi_agent=True,
                trajectory_path=trajectory_path,
                ego_or_npc="NPC",
                override_obs=10,
            )
            npc_agent.load_model(path=npc_model)
            npc_agent.cycle = cycle

            if cycle == 0:
                if args.sampling == "two_model":
                    ego_pool.add_model(ego_agent, 1000.0, 1000.0)
                npc_pool.add_model(npc_agent, 1000.0, 1000.0)
                print("Evaluation...")
                env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
                pool_evaluation(env, -1, ego_pool, npc_pool, args, device, False)
                env.close()
                print(f"Ego ELOS: {ego_pool.model_elo}")
                print(f"NPC ELOS: {npc_pool.model_elo}")

            # Change Environment to Reward Collision Avoidance
            ma_config["adversarial"] = True
            ma_config["normalize_reward"] = False
            ma_config["collision_reward"] = 400
            env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")

            train_ego = False
            npc_version += 1
            npc_model = os.path.join(
                args.model_folder,
                f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth",
            )
            npc_vs_ego_pool(
                env,
                npc_agent,
                ego_pool,
                args,
                device,
                ego_version,
                npc_version,
                npc_model,
            )
            npc_elo = ego_pool.opponent_elo
            npc_pool.add_model(npc_agent, 1000.0, 1000.0)
            print(f"NPC ELO: {npc_elo}")

            print("Evaluation...")
            env.close()
            env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
            pool_evaluation(
                env, cycle + 1 + 0.5, ego_pool, npc_pool, args, device, False
            )

            ego_pool.set_opponent_elo(npc_pool.model_elo[-1])
            npc_pool.set_opponent_elo(ego_pool.model_elo[-1])
            ego_pool.update_probabilities(False)
            npc_pool.update_probabilities(True)

            print(f"Ego ELOS: {ego_pool.model_elo}")
            print(f"NPC ELOS: {npc_pool.model_elo}")

            # Change Environment to Reward Collision
            ma_config["adversarial"] = False
            ma_config["normalize_reward"] = True
            ma_config["collision_reward"] = -1000
            env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")

            train_ego = True
            ego_version += 1
            ego_model = os.path.join(
                args.model_folder,
                f"E{ego_version}_V{npc_version}_TrainEgo_{train_ego}.pth",
            )
            ego_vs_npc_pool(
                env,
                ego_agent,
                npc_pool,
                args,
                device,
                ego_version,
                npc_version,
                ego_model,
            )
            ego_elo = npc_pool.opponent_elo
            ego_pool.add_model(ego_agent, 1000.0, 1000.0)
            print(f"EGO ELO: {ego_elo}")

            env.close()
            print("Evaluation...")
            env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
            pool_evaluation(env, cycle + 1, ego_pool, npc_pool, args, device, True)
            env.close()

            ego_pool.set_opponent_elo(npc_pool.model_elo[-1])
            npc_pool.set_opponent_elo(ego_pool.model_elo[-1])
            ego_pool.update_probabilities(True)
            npc_pool.update_probabilities(False)

            print(f"Ego ELOS: {ego_pool.model_elo}")
            print(f"NPC ELOS: {npc_pool.model_elo}")

        ego_pool.end_pool()
        npc_pool.end_pool()

        ego_pool.write_model_pool_log(
            os.path.join(
                args.model_folder, f"ego_pool_{args.start_e}_{args.sampling}.json"
            )
        )
        npc_pool.write_model_pool_log(
            os.path.join(
                args.model_folder, f"npc_pool_{args.start_e}_{args.sampling}.json"
            )
        )

        env.close()

    else:
        model = "E0_MOBIL.pth"
        model_dir = ""

        ma_config["use_mobil"] = True
        ma_config["ego_vs_mobil"] = True
        ma_config["adversarial"] = False
        ma_config["normalize_reward"] = True
        ma_config["collision_reward"] = -1000
        env = gym.make("crash-v0", config=ma_config, render_mode="rgb_array")
        agent = DQN_Agent(env, args, device, multi_agent=True, ego_or_npc="EGO")
        agent.load_model(path=os.path.join(model_dir, model))

        agent_vs_mobil(env, agent, args, device)
