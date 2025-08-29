import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
from scenarios import Slowdown, SlowdownSameLane, SpeedUp, CutIn
from config import load_config
import torch
import tyro
from tqdm import tqdm
from arguments import Args
from dqn_agent import DQN_Agent
from create_env import make_vector_env
import json
from helpers import obs_to_state
from train_agent import test_scenarios, train_scenarios, train_vs_mobil

import highway_env

if __name__ == "__main__":
    config = load_config("model_configs/test_scenario_config.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config["vs_mobil"] = True
    gym_config["controlled_vehicles"] = 1
    gym_config["other_vehicles"] = 1
    config["gym_config"] = gym_config

    test_scenarios(config)
