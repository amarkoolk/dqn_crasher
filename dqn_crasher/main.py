import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from runner      import MultiAgentRunner
from dqn_agent   import DQN_Agent
from config      import load_config
import utils
from helpers import make_players
import wandb
from wandb_logging import initialize_logging

import scenarios, helpers, utils
import gymnasium as gym
import highway_env

if __name__=="__main__":

    config = load_config("model_configs/dqn_vs_scenarios.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config['adversarial'] = False
    gym_config['normalize_reward'] = True
    gym_config['collision_reward'] = -100
    gym_config['observation']['observation_config']['frame_stack'] = config['frame_stack']
    config['spawn_config'] = gym_config['spawn_configs']
    config['gym_config'] = gym_config

    run = initialize_logging(config, train_ego=True)

    is_sweep = bool(os.getenv("WANDB_SWEEP_ID"))
    if is_sweep:
        config['root_directory'] = os.path.join(config['root_directory'], f"{config['batch_size']}_{config['num_hidden_layers']}_{config['hidden_layer']}")


    device = utils.DeviceHelper.get(config)
    p_A, p_B = make_players(config, gym_config, device)

    runner = MultiAgentRunner(config["env_name"], config, gym_config, device, p_A, p_B)

    if config.get('train_ego', False):
        runner.train(train_player="A")
    else:
        runner.test()
