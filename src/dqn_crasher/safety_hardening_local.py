import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import gymnasium as gym
import highway_env
import torch
from dqn_crasher.utils.config import load_pkg_yaml
from dqn_crasher.utils.helpers import make_players
from dqn_crasher.utils.wandb_logging import initialize_logging

import dqn_crasher.scenarios.policies as policies
import dqn_crasher.scenarios.scenarios as scenarios
import wandb
from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.training.runner import MultiAgentRunner
from dqn_crasher.utils.utils import DeviceHelper

def run_local_training(config, device, cycle, model_dir, init_model_path_A, init_model_path_B, train_ego: bool = False):

    ego_trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
        f"EGO_{cycle}"
    )

    npc_trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
        f"NPC_{cycle}"
    )


    ego_gym_config = load_pkg_yaml("configs/env/multi_agent_old.yaml")
    npc_gym_config = load_pkg_yaml("configs/env/multi_agent_npc.yaml")

    ego_gym_config["observation"]["observation_config"]["frame_stack"] = config.get("frame_stack", 1)
    npc_gym_config["observation"]["observation_config"]["frame_stack"] = config.get("frame_stack", 1)

    tmp = gym.make(config["env_name"], config=npc_gym_config)
    act_space = tmp.action_space[0]
    n_act = act_space.n
    n_obs = 10 * config.get("frame_stack", 1)
    tmp.close()

    ego_agent = DQN_Agent(n_obs, n_act, act_space, config, device)
    print(f"SETTING EGO POLICY TRAIN: {train_ego}")
    print(f"INITIAL MODEL: {init_model_path_A}")
    ego_policy = policies.DQNPolicy(ego_agent, ego_trajectory_save_path, train = train_ego, init_model=init_model_path_A)

    npc_agent = DQN_Agent(n_obs, n_act, act_space, config, device)
    print(f"SETTING NPC POLICY TRAIN: {not train_ego}")
    print(f"INITIAL MODEL: {init_model_path_B}")
    npc_policy = policies.DQNPolicy(npc_agent, npc_trajectory_save_path, train = not train_ego, init_model=init_model_path_B)


    if train_ego:
        config["gym_config"] = ego_gym_config
        wandb_name = "EGO"
        p_A = ego_policy
        p_B = npc_policy
    else:
        config["gym_config"] = npc_gym_config
        wandb_name = "NPC"

        p_A = npc_policy
        p_B = ego_policy

    

    run_name = f"{wandb_name}{cycle}_" + config["run_name"]
    config["model_save_path"] = model_dir + f"/{wandb_name}_{cycle}"


    run = initialize_logging(config, train_ego, run_name = run_name)

    runner = MultiAgentRunner(config, device, p_A, p_B)
    runner.train(train_player="A")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("env_config")
    args = parser.parse_args()

    config = load_pkg_yaml(f"configs/model/{args.env_config}.yaml")
    device = DeviceHelper.get(config)

    ego_model = config.get("initial_model_A")
    npc_model = ""

    model_dir = os.path.join(
                    config.get("root_directory", ""),
                    config.get("model_save_path", "models/model")
                )

    for cycle in range(1, int(config.get("cycles")) + 1):

        # Train NPC
        run_local_training(config, device, cycle, model_dir, ego_model, npc_model, train_ego = False)

        npc_model = os.path.join(
                    model_dir,
                    f"NPC_{cycle}/model_{config.get("total_timesteps")}.pth"
                )
        
        run_local_training(config, device, cycle, model_dir, ego_model, npc_model, train_ego = True)

        ego_model = os.path.join(
                    model_dir,
                    f"EGO_{cycle}/model_{config.get("total_timesteps")}.pth"
                )
        
    
    


if __name__ == "__main__":
    main()
