from collections import deque
import copy
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
        "episode_num": 0,
        "policy_name": None,
        "specific_policy_name": None,
        "policy_type": None,
        "training_step": 0,
        "testing_step": 0,
        "checkpoint_step": 0,
        "metrics_type": "training",
        "checkpoint_testing": False,
        "is_aggregated": False
    }

def populate_stats(info, agent, ego_state, npc_state, reward, episode_statistics: dict, is_ego: bool = True, scenario = None, policy_name = None, specific_policy = None):

    episode_statistics['spawn_config'] = info['spawn_config']
    episode_statistics['scenario'] = scenario
    if policy_name:
        episode_statistics['policy_name'] = policy_name
    if specific_policy:
        episode_statistics['specific_policy_name'] = specific_policy
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

def reset_stats(stats: dict, preserve_episode_num=False):
    # Save counters and flags if we need to preserve them
    counters_to_preserve = {}
    if preserve_episode_num:
        # Save all step counters and metadata
        counters_to_preserve = {
            'episode_num': stats.get('episode_num', 0),
            'training_step': stats.get('training_step', 0),
            'testing_step': stats.get('testing_step', 0),
            'checkpoint_step': stats.get('checkpoint_step', 0),
            'metrics_type': stats.get('metrics_type', 'training'),
            'checkpoint_testing': stats.get('checkpoint_testing', False),
            'policy_name': stats.get('policy_name', None),
            'specific_policy_name': stats.get('specific_policy_name', None),
            'policy_type': stats.get('policy_type', None),
            'scenario': stats.get('scenario', None),
            'spawn_config': stats.get('spawn_config', None)
        }

    stats["episode_rewards"] = 0
    stats["episode_duration"] = 0
    stats["ego_speed"] = 0
    stats["npc_speed"] = 0
    stats["right_lane_reward"] = 0.0
    stats["high_speed_reward"] = 0.0
    stats["collision_reward"] = 0.0
    stats["ttc_x_reward"] = 0.0
    stats["ttc_y_reward"] = 0.0

    # Restore all preserved counters or increment episode_num
    if preserve_episode_num and counters_to_preserve:
        # Restore all counters and metadata
        for key, value in counters_to_preserve.items():
            stats[key] = value
    else:
        # Just increment the episode number for normal progress
        stats['episode_num'] = stats.get('episode_num', 0) + 1

def aggregate_checkpoint_stats(stats_list, checkpoint_step=None):
    """
    Aggregate multiple episode statistics into a single checkpoint summary.

    Args:
        stats_list: List of statistics dictionaries to aggregate
        checkpoint_step: The training step for this checkpoint, used for logging

    Returns:
        A new dictionary with aggregated statistics
    """
    if not stats_list:
        return initialize_stats()

    # Create a new stats dictionary
    agg_stats = initialize_stats()

    # Set the counters for this aggregated stat
    if checkpoint_step is not None:
        agg_stats['episode_num'] = checkpoint_step
        agg_stats['checkpoint_step'] = checkpoint_step
        agg_stats['metrics_type'] = 'checkpoint_summary'

    # Copy specific policy info from the first stat if available
    if stats_list and len(stats_list) > 0:
        first_stat = stats_list[0]
        if 'specific_policy_name' in first_stat:
            agg_stats['specific_policy_name'] = first_stat['specific_policy_name']
        if 'policy_type' in first_stat:
            agg_stats['policy_type'] = first_stat['policy_type']
        if 'scenario' in first_stat:
            agg_stats['scenario'] = first_stat['scenario']

    # Aggregate simple numeric fields
    total_episodes = len(stats_list)
    if total_episodes > 0:
        for stat in stats_list:
            # Merge crash counts by spawn config
            for spawn_config, crashes in stat.get('num_crashes', {}).items():
                if spawn_config not in agg_stats['num_crashes']:
                    agg_stats['num_crashes'][spawn_config] = []
                agg_stats['num_crashes'][spawn_config].extend(crashes)

            # Merge total crashes
            agg_stats['total_crashes'].extend(stat.get('total_crashes', []))

            # Merge reward histories - handle deques carefully
            # Convert to list, extend, then create a new deque
            temp_rewards = list(agg_stats['ep_rew_total'])
            temp_rewards.extend(list(stat.get('ep_rew_total', [])))
            agg_stats['ep_rew_total'] = deque(temp_rewards[-100:], maxlen=100)

            temp_lengths = list(agg_stats['ep_len_total'])
            temp_lengths.extend(list(stat.get('ep_len_total', [])))
            agg_stats['ep_len_total'] = deque(temp_lengths[-100:], maxlen=100)

            temp_ego_speed = list(agg_stats['ego_speed_total'])
            temp_ego_speed.extend(list(stat.get('ego_speed_total', [])))
            agg_stats['ego_speed_total'] = deque(temp_ego_speed[-100:], maxlen=100)

            temp_npc_speed = list(agg_stats['npc_speed_total'])
            temp_npc_speed.extend(list(stat.get('npc_speed_total', [])))
            agg_stats['npc_speed_total'] = deque(temp_npc_speed[-100:], maxlen=100)

            # Calculate episode statistics
            if 'episode_rewards' in stat:
                agg_stats['episode_rewards'] += stat['episode_rewards']
            if 'episode_duration' in stat:
                agg_stats['episode_duration'] += stat['episode_duration']
            if 'ego_speed' in stat:
                agg_stats['ego_speed'] += stat['ego_speed']
            if 'npc_speed' in stat:
                agg_stats['npc_speed'] += stat['npc_speed']
            if 'right_lane_reward' in stat:
                agg_stats['right_lane_reward'] += stat.get('right_lane_reward', 0)
            if 'high_speed_reward' in stat:
                agg_stats['high_speed_reward'] += stat.get('high_speed_reward', 0)
            if 'collision_reward' in stat:
                agg_stats['collision_reward'] += stat.get('collision_reward', 0)
            if 'ttc_x_reward' in stat:
                agg_stats['ttc_x_reward'] += stat.get('ttc_x_reward', 0)
            if 'ttc_y_reward' in stat:
                agg_stats['ttc_y_reward'] += stat.get('ttc_y_reward', 0)

            # Use the last scenario and policy name (should be the same across all episodes)
            if 'scenario' in stat:
                agg_stats['scenario'] = stat['scenario']
            if 'policy_name' in stat:
                agg_stats['policy_name'] = stat['policy_name']
            if 'specific_policy_name' in stat:
                agg_stats['specific_policy_name'] = stat['specific_policy_name']
            if 'policy_type' in stat:
                agg_stats['policy_type'] = stat['policy_type']

            # Set epsilon to the most recent value
            if 'epsilon' in stat:
                agg_stats['epsilon'] = stat['epsilon']

    # Add calculated aggregate metrics
    if total_episodes > 0:
        agg_stats['total_episodes'] = total_episodes
        if len(agg_stats['total_crashes']) > 0:
            agg_stats['success_rate'] = sum(agg_stats['total_crashes']) / len(agg_stats['total_crashes'])

        # Add metadata to indicate these are aggregated stats
        agg_stats['is_aggregated'] = True

        # Compute mean reward and episode length if available
        if agg_stats['ep_rew_total'] and len(agg_stats['ep_rew_total']) > 0:
            agg_stats['mean_reward'] = sum(agg_stats['ep_rew_total']) / len(agg_stats['ep_rew_total'])
        if agg_stats['ep_len_total'] and len(agg_stats['ep_len_total']) > 0:
            agg_stats['mean_episode_length'] = sum(agg_stats['ep_len_total']) / len(agg_stats['ep_len_total'])

        # Add policy-specific tags for better filtering in WandB
        if 'specific_policy_name' in agg_stats and agg_stats['specific_policy_name']:
            agg_stats['policy_tag'] = agg_stats['specific_policy_name']
        elif 'policy_name' in agg_stats and agg_stats['policy_name']:
            agg_stats['policy_tag'] = agg_stats['policy_name']

    return agg_stats

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
