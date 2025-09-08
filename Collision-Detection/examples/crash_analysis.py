import json
import os
import time
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4


def analyze_crash_trajectories(trajectory_path, file_name, framerate=15):
    """
    Analyze crash trajectories to understand collision patterns and directionality.
    
    Args:
        trajectory_path: Path to trajectory files
        file_name: Name of the trajectory file
        framerate: Simulation framerate
    """
    
    with open(os.path.join(trajectory_path, file_name), "r") as f:
        data = json.load(f)

    episode_keys = list(data.keys())
    num_episodes = len(episode_keys)

    # Initialize arrays for analysis
    ttc_x = np.zeros((num_episodes, 600))
    ttc_y = np.zeros((num_episodes, 600))
    dttc_x_dt = np.zeros((num_episodes, 600))
    dttc_y_dt = np.zeros((num_episodes, 600))
    distance = np.zeros((num_episodes, 600))
    d_distance_dt = np.zeros((num_episodes, 600))
    n_frames = np.zeros(num_episodes)
    ego_vy = np.zeros((num_episodes, 600))
    npc_vy = np.zeros((num_episodes, 600))
    reward = np.zeros((num_episodes, 600))
    end_distance = np.zeros(num_episodes)

    # Process each episode
    for i in range(num_episodes):
        episode_data = np.asarray(data[episode_keys[i]])
        episode_data = np.delete(
            episode_data, np.arange(0, episode_data.shape[0], framerate), axis=0
        )

        num_frames = episode_data.shape[0]
        n_frames[i] = num_frames
        ts = np.arange(0, num_frames, 1) / framerate
        
        # Extract vehicle positions and velocities
        ego_x = episode_data[:, 1]
        ego_y = episode_data[:, 2]
        ego_vx = episode_data[:, 3]
        ego_vy[i, :num_frames] = episode_data[:, 4]

        npc_x = episode_data[:, 6] + episode_data[:, 1]
        npc_y = episode_data[:, 7] + episode_data[:, 2]
        npc_vx = -episode_data[:, 8] + episode_data[:, 3]
        npc_vy[i, :num_frames] = -episode_data[:, 9] + episode_data[:, 4]

        ttc_x[i, :num_frames] = episode_data[:, -2]
        ttc_y[i, :num_frames] = episode_data[:, -1]

        distance[i, :num_frames] = np.sqrt((ego_x - npc_x) ** 2 + (ego_y - npc_y) ** 2)
        end_distance[i] = distance[i, num_frames - 1]

        # Calculate derivatives for collision analysis
        dttc_x_dt[i, :num_frames] = np.gradient(ttc_x[i, :num_frames], ts)
        dttc_y_dt[i, :num_frames] = np.gradient(ttc_y[i, :num_frames], ts)
        d_distance_dt[i, :num_frames] = np.gradient(distance[i, :num_frames], ts)

        reward[i, :num_frames] = episode_data[:, -3]

    return {
        'ttc_x': ttc_x,
        'ttc_y': ttc_y,
        'dttc_x_dt': dttc_x_dt,
        'dttc_y_dt': dttc_y_dt,
        'distance': distance,
        'd_distance_dt': d_distance_dt,
        'n_frames': n_frames,
        'ego_vy': ego_vy,
        'npc_vy': npc_vy,
        'reward': reward,
        'end_distance': end_distance,
        'num_episodes': num_episodes
    }


def detect_collision_directionality(ego_pos, ego_vel, npc_pos, npc_vel, ego_heading, npc_heading):
    """
    Detect collision directionality based on vehicle positions, velocities, and headings.
    
    Args:
        ego_pos: [x, y] position of ego vehicle
        ego_vel: [vx, vy] velocity of ego vehicle  
        npc_pos: [x, y] position of NPC vehicle
        npc_vel: [vx, vy] velocity of NPC vehicle
        ego_heading: heading angle of ego vehicle (radians)
        npc_heading: heading angle of NPC vehicle (radians)
        
    Returns:
        dict: Collision analysis including type and relative motion
    """
    
    # Calculate relative position and velocity
    rel_pos = np.array(npc_pos) - np.array(ego_pos)
    rel_vel = np.array(npc_vel) - np.array(ego_vel)
    
    # Calculate relative distance and approach rate
    distance = np.linalg.norm(rel_pos)
    approach_rate = -np.dot(rel_pos, rel_vel) / distance if distance > 0 else 0
    
    # Determine collision angle
    collision_angle = np.arctan2(rel_pos[1], rel_pos[0])
    
    # Classify collision type based on relative positions and headings
    ego_forward = np.array([np.cos(ego_heading), np.sin(ego_heading)])
    npc_forward = np.array([np.cos(npc_heading), np.sin(npc_heading)])
    
    # Calculate angle between vehicle orientations
    heading_diff = abs(ego_heading - npc_heading)
    heading_diff = min(heading_diff, 2*np.pi - heading_diff)  # Normalize to [0, π]
    
    # Determine collision type
    if heading_diff < np.pi/4:  # Similar directions
        if approach_rate > 0:
            collision_type = "rear-end"
        else:
            collision_type = "same-direction"
    elif heading_diff > 3*np.pi/4:  # Opposite directions
        collision_type = "head-on"
    else:  # Perpendicular or angled
        if abs(rel_pos[1]) > abs(rel_pos[0]):
            collision_type = "side-swipe-lateral"
        else:
            collision_type = "side-swipe-longitudinal"
    
    return {
        'type': collision_type,
        'relative_position': rel_pos,
        'relative_velocity': rel_vel,
        'distance': distance,
        'approach_rate': approach_rate,
        'collision_angle': collision_angle,
        'heading_difference': heading_diff
    }


def plot_collision_analysis(analysis_data):
    """
    Plot collision analysis results.
    """
    ttc_x = analysis_data['ttc_x']
    ttc_y = analysis_data['ttc_y']
    distance = analysis_data['distance']
    d_distance_dt = analysis_data['d_distance_dt']
    dttc_x_dt = analysis_data['dttc_x_dt']
    dttc_y_dt = analysis_data['dttc_y_dt']
    n_frames = analysis_data['n_frames']
    num_episodes = analysis_data['num_episodes']
    
    # Plot distance trajectories
    frames = n_frames[n_frames >= 560]
    nm = distance[n_frames >= 560, :]
    plt.figure(figsize=(12, 8))
    
    for i in range(len(frames)):
        plt.plot(nm[i, : int(frames[i])], c="b", alpha=0.1)
    plt.grid()
    plt.ylabel("Distance (m)")
    plt.xlabel("Frame")
    plt.title("Relative Distance Over Time")
    plt.ylim(-10, 100)
    plt.show()

    # Plot derivatives for collision detection
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    
    for i in range(num_episodes):
        if n_frames[i] < 560.0:
            color = "r"
            alpha = 0.01
        else:
            color = "b"
            alpha = 0.01

        # Highlight collision episodes
        if np.any(distance[i, :int(n_frames[i])] < 6.0) and n_frames[i] >= 560:
            color = "g"
            alpha = 1.0

        ax[0].plot(d_distance_dt[i, :], c=color, alpha=alpha)
        ax[1].plot(dttc_x_dt[i, :], c=color, alpha=alpha)
        ax[2].plot(dttc_y_dt[i, :], c=color, alpha=alpha)
        
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[0].set_ylabel("d_distance/dt")
    ax[1].set_ylabel("d_ttc_x/dt")
    ax[2].set_ylabel("d_ttc_y/dt")
    fig.legend()
    fig.suptitle("Collision Indicators: Derivatives of Distance and TTC")
    fig.supxlabel("Frame")
    plt.show()


if __name__ == "__main__":
    # Example usage
    trajectory_path = "collision_data/episodes/test"
    file_name = "episode_data.json"
    
    if os.path.exists(os.path.join(trajectory_path, file_name)):
        print("Analyzing crash trajectories...")
        analysis_data = analyze_crash_trajectories(trajectory_path, file_name)
        plot_collision_analysis(analysis_data)
        
        print(f"Analyzed {analysis_data['num_episodes']} episodes")
        print(f"Episodes with collisions: {np.sum(analysis_data['end_distance'] < 6.0)}")
    else:
        print(f"Trajectory file not found: {os.path.join(trajectory_path, file_name)}")
        print("Run some episodes first to generate trajectory data.")
