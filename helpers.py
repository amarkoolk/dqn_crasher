from collections import deque
import torch
from typing import Callable, List, Sequence, Tuple, Union
import importlib

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

def populate_stats(info, agent, ego_state, npc_state, reward, episode_statistics: dict, is_ego: bool = True):

    episode_statistics['spawn_config'] = info['spawn_config']
    episode_statistics['episode_rewards'] += reward
    episode_statistics['episode_duration'] += 1
    episode_statistics['ego_speed'] += ego_state[0,3].cpu().numpy()
    episode_statistics['npc_speed'] += npc_state[0,3].cpu().numpy() + npc_state[0,8].cpu().numpy()
    if is_ego:
        episode_statistics['right_lane_reward'] += info['rewards']['right_lane_reward']
        episode_statistics['high_speed_reward'] += info['rewards']['high_speed_reward']
    else:
        episode_statistics['ttc_x_reward'] += info['rewards']['ttc_x_reward']
        episode_statistics['ttc_y_reward'] += info['rewards']['ttc_y_reward']
    episode_statistics['collision_reward'] += info['rewards']['collision_reward']
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

def obs_to_state(obs, ego_agent, npc_agent, device):
    if(len(obs) == 2):
        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[:ego_agent.n_observations]
        ego_state = torch.tensor(ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device)
        flattened_npc_obs = obs[1].flatten()
        npc_obs = flattened_npc_obs[:npc_agent.n_observations]
        npc_state = torch.tensor(npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device)
    elif(len(obs) == 1):
        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[:ego_agent.n_observations]
        ego_state = torch.tensor(ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device)
        npc_state = torch.tensor(ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device)
        npc_state[0, 0] = 1.0
        npc_state[0, 1] = ego_state[0, 1] + ego_state[0, 6]
        npc_state[0, 2] = ego_state[0, 2] + ego_state[0, 7]
        npc_state[0, 3] = ego_state[0, 3] + ego_state[0, 8]
        npc_state[0, 4] = ego_state[0, 4] + ego_state[0, 9]

    return ego_state, npc_state