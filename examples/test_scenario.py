import json

import gymnasium as gym
import highway_env
import numpy as np
import torch
import tyro
from arguments import Args
from config import load_config
from create_env import make_vector_env
from dqn_agent import DQN_Agent
from gymnasium.vector import AsyncVectorEnv
from helpers import obs_to_state
from scenarios import CutIn, Slowdown, SlowdownSameLane, SpeedUp
from tqdm import tqdm
from train_agent import test_scenarios, train_scenarios, train_vs_mobil

if __name__ == "__main__":
    config = load_config("model_configs/test_scenario_config.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config["vs_mobil"] = True
    gym_config["controlled_vehicles"] = 1
    gym_config["other_vehicles"] = 1
    config["gym_config"] = gym_config

    test_scenarios(config)
