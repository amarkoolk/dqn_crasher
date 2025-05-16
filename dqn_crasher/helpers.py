from collections import deque
import torch
from typing import Callable, List, Sequence, Tuple, Union
import importlib
import numpy as np

def class_from_path(path: str) -> Callable:
    module_name, class_name = path.rsplit(".", 1)
    class_object = getattr(importlib.import_module(module_name), class_name)
    return class_object

def initialize_stats(queue_len = 100) -> dict:
    return {
        "num_crashes": {},
        "total_crashes": [],
        "episode_rewards": 0,
        "episode_duration": 0,
        "ego_speed": 0,
        "npc_speed": 0,
        "ep_rew_total": deque([], maxlen=queue_len),
        "ep_len_total": deque([], maxlen=queue_len),
        "ego_speed_total": deque([], maxlen=queue_len),
        "npc_speed_total": deque([], maxlen=queue_len),
        "right_lane_reward": 0.0,
        "high_speed_reward": 0.0,
        "collision_reward": 0.0,
        "ttc_x_reward": 0.0,
        "ttc_y_reward": 0.0,
        "epsilon": 0.0,
        "spawn_config": None,
        "episode_num": 0
    }

def populate_stats(info, agent, ego_state, npc_state, reward, episode_statistics: dict, is_ego: bool = True, scenario = None):

    episode_statistics['spawn_config'] = info['spawn_config']
    episode_statistics['scenario'] = scenario
    episode_statistics['episode_rewards'] += reward
    episode_statistics['episode_duration'] += 1
    episode_statistics['ego_speed'] += ego_state[0,3].cpu().numpy()
    episode_statistics['npc_speed'] += npc_state[0,3].cpu().numpy()
    if is_ego:
        episode_statistics['right_lane_reward'] += info['rewards']['right_lane_reward']
        episode_statistics['high_speed_reward'] += info['rewards']['high_speed_reward']
    else:
        episode_statistics['ttc_x_reward'] += info['rewards']['ttc_x_reward']
        episode_statistics['ttc_y_reward'] += info['rewards']['ttc_y_reward']
    episode_statistics['collision_reward'] += info['rewards']['collision_reward']
    if agent is not None:
        episode_statistics['epsilon'] = agent.eps_threshold

def reset_stats(stats: dict):
    stats["episode_rewards"] = 0
    stats["episode_duration"] = 0
    stats["ego_speed"] = 0
    stats["npc_speed"] = 0
    stats["right_lane_reward"] = 0.0
    stats["high_speed_reward"] = 0.0
    stats["collision_reward"] = 0.0
    stats["ttc_x_reward"] = 0.0
    stats["ttc_y_reward"] = 0.0

    stats['episode_num'] += 1

def interleave_stacked_frames(obs, n_cars, n_features, n_stack):
    if n_stack > 1:
        reshaped_obs = obs.reshape((obs.shape[0], n_features, n_stack))
        interleaved_obs = np.empty((n_cars * n_stack, n_features), dtype = obs.dtype)
        for i in range(n_cars):
            interleaved_obs[i::n_cars] = reshaped_obs[i]
    else:
        interleaved_obs = obs

    return interleaved_obs.flatten()

def obs_to_state(obs, n_observations, device, num_features = 5, frame_stack = 1):

    if(len(obs) == 2):
        flattened_ego_obs = obs[0].flatten()
        ego_obs = interleave_stacked_frames(obs[0], len(obs), num_features, frame_stack)
        ego_state = torch.tensor(ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device)
        npc_obs = interleave_stacked_frames(obs[1], len(obs), num_features, frame_stack)
        npc_state = torch.tensor(npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device)
    elif(len(obs) == 1):
        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[:n_observations]
        ego_state = torch.tensor(ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device)
        npc_state = torch.tensor(ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device)
        npc_state[0, 0] = 1.0
        npc_state[0, 1] = ego_state[0, 6]
        npc_state[0, 2] = ego_state[0, 7]
        npc_state[0, 3] = ego_state[0, 8]
        npc_state[0, 4] = ego_state[0, 9]
        npc_state[0, 5] = ego_state[0, 0]
        npc_state[0, 6] = ego_state[0, 1]
        npc_state[0, 7] = ego_state[0, 2]
        npc_state[0, 8] = ego_state[0, 3]
        npc_state[0, 9] = ego_state[0, 4]

    return ego_state, npc_state

def make_step_actions(ego_action, npc_action, vs_mobil=False):
    # if we’re “versus MOBIL” then MOBIL is our ego,
    # so we send (npc, npc)
    return (npc_action, npc_action) if vs_mobil else (ego_action, npc_action)
