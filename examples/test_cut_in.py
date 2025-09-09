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
from helpers import make_step_actions, unpack_states
from scenarios import CutIn, IdleFaster, IdleSlower, Slowdown, SlowdownSameLane, SpeedUp
from tqdm import tqdm

if __name__ == "__main__":
    args = tyro.cli(Args)
    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    #   scenarios = [IdleSlower(), SlowdownSameLane(), IdleFaster(), CutIn()]
    scenarios = [CutIn()]  # , SlowdownSameLane(), SpeedUp(), CutIn()]

    config = load_config("model_configs/training_config.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config["observation"]["observation_config"]["frame_stack"] = args.frame_stack

    vs_mobil = True
    gym_config["vs_mobil"] = vs_mobil

    for scenario in tqdm(scenarios):
        scenario.set_config(gym_config)
        gym_config["initial_speed"] = 20
        if vs_mobil:
            gym_config["controlled_vehicles"] = 1
            gym_config["other_vehicles"] = 1

        env = gym.make("crash-v0", config=gym_config, render_mode="rgb_array")

        ego_model0 = "E0_MOBIL.pth"

        action_space = env.action_space[0]
        n_actions = action_space.n
        n_obs = 10 * args.frame_stack

        ego_agent = DQN_Agent(n_obs, n_actions, action_space, config, device)
        ego_agent.load_model(path=ego_model0)

        for _ in tqdm(range(3), leave=False):
            done = truncated = False
            obs, info = env.reset()
            ego_state, npc_state = unpack_states(obs, ego_agent, device, vs_mobil)
            scenario.reset(ego_state, npc_state, info)
            while not (done or truncated):  # or scenario.end_frames > 15):
                action = scenario.get_action()
                npc_action = (
                    torch.squeeze(torch.tensor([action])).view(1, 1).cpu().numpy()
                )
                ego_action = (
                    torch.squeeze(ego_agent.predict(ego_state)).view(1, 1).cpu().numpy()
                )
                actions = make_step_actions(ego_action, npc_action, vs_mobil)
                obs, reward, terminated, truncated, info = env.step(actions)
                done = terminated or truncated

                ego_state, npc_state = unpack_states(obs, ego_agent, device, vs_mobil)

                scenario.set_state(
                    ego_state[0, 1], ego_state[0, 2], npc_state[0, 1], npc_state[0, 2]
                )
                env.render()

env.close()
