import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import gymnasium as gym
import highway_env
import torch
from utils.config import load_pkg_yaml
from utils.helpers import make_players
from utils.wandb_logging import initialize_logging

import dqn_crasher.scenarios.policies as policies
import dqn_crasher.scenarios.scenarios as scenarios
import wandb
from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.training.runner import MultiAgentRunner
from dqn_crasher.utils.utils import DeviceHelper


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("env_config")
    args = parser.parse_args()


    config = load_pkg_yaml(f"configs/model/{args.env_config}.yaml")
    gym_config = load_pkg_yaml("configs/env/multi_agent.yaml")
    gym_config["adversarial"] = False
    gym_config["normalize_reward"] = True
    gym_config["collision_reward"] = -1
    gym_config["ttc_x_reward"] = 0
    gym_config["ttc_y_reward"] = 0
    gym_config["observation"]["observation_config"]["frame_stack"] = config[
        "frame_stack"
    ]
    config["spawn_config"] = gym_config["spawn_configs"]
    config["gym_config"] = gym_config

    run = initialize_logging(config, train_ego=True)

    is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))
    if is_sweep:
        config["root_directory"] = os.path.join(
            config["root_directory"],
            f"{config['lr']}_{config['batch_size']}_{config['num_hidden_layers']}_{config['hidden_layer']}",
        )

    device = DeviceHelper.get(config)
    p_A, p_B = make_players(config, gym_config, device)

    runner = MultiAgentRunner(config["env_name"], config, gym_config, device, p_A, p_B)

    if config.get("train_ego", False):
        runner.train(train_player="A")
    else:
        runner.test()


if __name__ == "__main__":
    main()
