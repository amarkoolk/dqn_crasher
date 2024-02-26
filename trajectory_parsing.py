import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import json
import time


def plot_trajectories(episode_array, dt, axes, time_buffer, episode_num):

    for i in range(episode_array.shape[0]):
        start_time = time.monotonic_ns()

        ego_x = episode_array[i,1]
        ego_y = episode_array[i,2]
        ego_vx = episode_array[i,3]
        ego_vy = episode_array[i,4]

        npc_x = episode_array[i,6] + episode_array[i,1]
        npc_y = episode_array[i,7] + episode_array[i,2]
        npc_vx = episode_array[i,8] + episode_array[i,3]
        npc_vy = episode_array[i,9] + episode_array[i,4]

        # Rectangles
        ego_centroid = (ego_x, ego_y)
        ego_heading = np.arctan2(ego_vy, ego_vx)
        ego_anchor = (ego_centroid[0] - 2.5*np.cos(ego_heading), ego_centroid[1] - 1.0*np.cos(ego_heading))
        ego_bbox = Rectangle(xy=ego_anchor, width=5, height=2, angle=np.rad2deg(ego_heading), fill=False, edgecolor='r')

        npc_centroid = (npc_x,npc_y)
        npc_heading = np.arctan2(npc_vy,npc_vx)
        npc_anchor = (npc_centroid[0] - 2.5*np.cos(npc_heading), npc_centroid[1] - 1.0*np.cos(npc_heading))
        npc_bbox = Rectangle(xy=npc_anchor, width=5, height=2, angle=np.rad2deg(npc_heading), fill=False, edgecolor='b')

        axes.add_patch(ego_bbox)
        axes.add_patch(npc_bbox)
        axes.annotate(f'Ego: {ego_vx:.2f}', xy=ego_anchor, xytext=(ego_anchor[0]+1, ego_anchor[1]+1))
        axes.annotate(f'NPC: {npc_vx:.2f}', xy=npc_anchor, xytext=(npc_anchor[0]+1, npc_anchor[1]+1))
        axes.text(0.1, 0.99, f'Episode: {episode_num}', fontsize=12, ha='left', va='top', transform=axes.transAxes)
        min_y = -10
        max_y = 14
        xy_scalar = (max_y-min_y)/0.6
        min_x = ego_anchor[0]-xy_scalar
        max_x = ego_anchor[0]+xy_scalar
        axes.set_xlim(min_x, max_x)
        axes.set_ylim(min_y, max_y)
        axes.set_ylim(axes.get_ylim()[::-1])
        # Draw the Road
        axes.hlines(-2, min_x - 5, max_x + 5, colors='k')
        axes.hlines(2, min_x - 5, max_x + 5, colors='k', linestyles='dashed')
        axes.hlines(6, min_x - 5, max_x + 5, colors='k')
        if len(time_buffer) == 0:
            time_to_sleep = 0.01
        else:
            time_to_sleep = min(dt - np.mean(time_buffer),0.01)
        plt.pause(time_to_sleep)
        axes.clear()
        end_time = time.monotonic_ns()
        time_buffer.append((end_time - start_time)/1e9)


trajectory_path = 'trajectories'
file_name = 'trajectories_ego_0.json'

with open(os.path.join(trajectory_path, file_name), 'r') as f:
    data = json.load(f)

episode_keys = list(data.keys())
num_episodes = len(episode_keys)

dt = 1/15

fig, axes = plt.subplots(1, 1, figsize=(15, 6))
time_buffer = []

for i in np.arange(0, num_episodes, 100):
    episode_array = np.asarray(data[episode_keys[i]])
    plot_trajectories(episode_array, dt, axes, time_buffer, episode_keys[i])