import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import json
import time
from enum import Enum

class Action(Enum):
  LANE_LEFT = 0
  IDLE = 1
  LANE_RIGHT = 2
  FASTER = 3
  SLOWER = 4

def plot_trajectories(episode_array, frame_rate, axes, time_buffer, episode_num):

    action_string = ''

    # fig = plt.figure()
    # plt.plot(episode_array[:,3])
    # plt.show()

    print(f'Episode Array Shape: {episode_array.shape}')
    for i in range(episode_array.shape[0]):
        if i%frame_rate == 0:
            action_value = episode_array[i,-4]
            action_type = Action(action_value).name
            action_string = f'{action_type}'
            print(f'Episode: {episode_num}, Frame: {i}, Action: {episode_array[i,-4]}')
            continue

        ego_x = episode_array[i,1]
        ego_y = episode_array[i,2]
        ego_vx = episode_array[i,3]
        ego_vy = episode_array[i,4]

        npc_x = episode_array[i,6] + episode_array[i,1]
        npc_y = episode_array[i,7] + episode_array[i,2]
        npc_vx = episode_array[i,8] + episode_array[i,3]
        npc_vy = episode_array[i,9] + episode_array[i,4]

        print(f'Ego: {ego_vx:.2f}, {ego_vy:.2f}')
        print(f'NPC: {npc_vx:.2f}, {npc_vy:.2f}')

        ttcx = episode_array[i,-2]
        ttcy = episode_array[i,-1]

        ttcx_string = f'{ttcx:.2f}'
        ttcy_string = f'{ttcy:.2f}'
        if abs(ttcx) > 1000:
            ttcx_string = 'INF'
        if abs(ttcy) > 1000:
            ttcy_string = 'INF'

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
        # axes.text(0.1, 0.99, f'Episode: {episode_num}, TTCX: {ttcx:.2f}, TTCY: {ttcy:.2f}', fontsize=12, ha='left', va='top', transform=axes.transAxes)

        # Modified lines with individual colors for TTCX and TTCY
        axes.text(0.1, 0.99, f'Episode: {episode_num}, ', fontsize=12, ha='left', va='top', transform=axes.transAxes)
        axes.text(0.3, 0.99, f'TTCX: {ttcx_string}', fontsize=12, ha='left', va='top', color='green', transform=axes.transAxes)
        axes.text(0.5, 0.99, f'TTCY: {ttcy_string}', fontsize=12, ha='left', va='top', color='purple', transform=axes.transAxes)
        axes.text(0.7, 0.99, f'Ego Action: {action_string}', fontsize=12, ha='left', va='top', color='purple', transform=axes.transAxes)

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
        time_to_sleep = 0.01
        if i == 1:
            plt.waitforbuttonpress()
        else:
            plt.pause(time_to_sleep)
        # plt.pause(time_to_sleep)
        axes.clear()
        plt.draw()


trajectory_path = 'complete_cycle_1.0_bl_two_model/E1_V0_TrainEgo_True/NPC'
file_name = '3.json'

with open(os.path.join(trajectory_path, file_name), 'r') as f:
    data = json.load(f)

episode_keys = list(data.keys())
num_episodes = len(episode_keys)

framerate = 15

fig, axes = plt.subplots(1, 1, figsize=(15, 6))
time_buffer = []


# # Play Specific Episode
# episode_num = 300
# episode_array = np.asarray(data[episode_keys[episode_num]])
# plot_trajectories(episode_array, framerate, axes, time_buffer, episode_keys[episode_num])

# Play all episodes
for i in np.arange(0, num_episodes, 10):
    episode_array = np.asarray(data[episode_keys[i]])
    plot_trajectories(episode_array, framerate, axes, time_buffer, episode_keys[i])