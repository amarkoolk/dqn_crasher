import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FFMpegWriter


import json
import time
from enum import Enum

figure_dir = 'figures'
fig_dir = os.path.join(os.getcwd(), figure_dir)
results_dir = 'results'
res_dir = os.path.join(os.getcwd(), results_dir)

tick_fontsize = 14
label_fontsize = 14
legend_fontsize = 14
title_fontsize = 16

class Action(Enum):
  LANE_LEFT = 0
  IDLE = 1
  LANE_RIGHT = 2
  FASTER = 3
  SLOWER = 4

# @profile
def plot_trajectories(episode_array, frame_rate, fig, axes, time_buffer, episode_num, video_path):

    action_string = ''

    # fig = plt.figure()
    # plt.plot(episode_array[:,3])
    # plt.show()

    # print(f'Episode Array Shape: {episode_array.shape}')


    axes[1].set_xlim(0, episode_array.shape[0])
    axes[1].set_ylim(min(episode_array[:,3].min(), episode_array[:,3].min() + episode_array[:,8].min()), max(episode_array[:,3].max(), episode_array[:,3].max() + episode_array[:,8].max()))
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Velocity vs Time')
    axes[1].grid()
    

    # Get Starting Config (behind, adjacent, forward)
    ego_x = episode_array[1,1]
    ego_y = episode_array[1,2]
    npc_x = episode_array[1,6] + episode_array[1,1]
    npc_y = episode_array[1,7] + episode_array[1,2]

    config_string = ''

    threshold = 5
    dist = ego_x - npc_x
    if dist > threshold:
        config_string = 'behind'
    elif dist < -threshold:
        config_string = 'forward'
    else:
        config_string = 'adjacent'

    print(f'Config: {config_string}')

    writer = FFMpegWriter(fps=15)

    with writer.saving(fig, os.path.join(video_path,config_string,f"{episode_num}.mp4"), dpi=100):
        for i in range(episode_array.shape[0]):
            if i%frame_rate == 0:
                action_value = episode_array[i,-4]
                action_type = Action(action_value).name
                action_string = f'{action_type}'
                # print(f'Episode: {episode_num}, Frame: {i}, Action: {episode_array[i,-4]}')
                continue


            axes[0].clear()
            axes[1].clear()

            ego_x = episode_array[i,1]
            ego_y = episode_array[i,2]
            ego_vx = episode_array[i,3]
            ego_vy = episode_array[i,4]

            npc_x = episode_array[i,6] + episode_array[i,1]
            npc_y = episode_array[i,7] + episode_array[i,2]
            npc_vx = episode_array[i,8] + episode_array[i,3]
            npc_vy = episode_array[i,9] + episode_array[i,4]

            # print(f'NPC: {ego_vx:.2f}, {ego_vy:.2f}')
            # print(f'Ego: {npc_vx:.2f}, {npc_vy:.2f}')

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
            ego_bbox = patches.Rectangle(xy=ego_anchor, width=5, height=2, angle=np.rad2deg(ego_heading), fill=False, edgecolor='b')

            npc_centroid = (npc_x,npc_y)
            npc_heading = np.arctan2(npc_vy,npc_vx)
            npc_anchor = (npc_centroid[0] - 2.5*np.cos(npc_heading), npc_centroid[1] - 1.0*np.cos(npc_heading))
            npc_bbox = patches.Rectangle(xy=npc_anchor, width=5, height=2, angle=np.rad2deg(npc_heading), fill=False, edgecolor='r')

            axes[0].add_patch(ego_bbox)
            axes[0].add_patch(npc_bbox)
            # axes[0].annotate(f'NPC: {ego_vx:.2f}', xy=ego_anchor, xytext=(ego_anchor[0]+1, ego_anchor[1]+1))
            # axes[0].annotate(f'Ego: {npc_vx:.2f}', xy=npc_anchor, xytext=(npc_anchor[0]+1, npc_anchor[1]+1))
            axes[0].annotate(f'EGO: {ego_vx:.2f}', xy=ego_anchor, xytext=(ego_anchor[0]+1, ego_anchor[1]+1))
            axes[0].annotate(f'NPC: {npc_vx:.2f}', xy=npc_anchor, xytext=(npc_anchor[0]+1, npc_anchor[1]+1))
            # axes.text(0.1, 0.99, f'Episode: {episode_num}, TTCX: {ttcx:.2f}, TTCY: {ttcy:.2f}', fontsize=12, ha='left', va='top', transform=axes.transAxes)

            # Modified lines with individual colors for TTCX and TTCY
            axes[0].text(0.1, 0.99, f'Episode: {episode_num}, ', fontsize=12, ha='left', va='top', transform=axes[0].transAxes)
            axes[0].text(0.3, 0.99, f'TTCX: {ttcx_string}', fontsize=12, ha='left', va='top', color='green', transform=axes[0].transAxes)
            axes[0].text(0.5, 0.99, f'TTCY: {ttcy_string}', fontsize=12, ha='left', va='top', color='purple', transform=axes[0].transAxes)
            axes[0].text(0.7, 0.99, f'Ego Action: {action_string}', fontsize=12, ha='left', va='top', color='purple', transform=axes[0].transAxes)

            min_y = -15
            max_y = 19
            xy_scalar = (max_y-min_y)/1.2
            min_x = npc_anchor[0]-xy_scalar
            max_x = npc_anchor[0]+xy_scalar
            axes[0].set_xlim(min_x, max_x)
            axes[0].set_ylim(min_y, max_y)
            axes[0].set_ylim(axes[0].get_ylim()[::-1])
            # Draw the Road
            axes[0].hlines(-2, min_x - 5, max_x + 5, colors='k')
            axes[0].hlines(2, min_x - 5, max_x + 5, colors='k', linestyles='dashed')
            axes[0].hlines(6, min_x - 5, max_x + 5, colors='k')
            time_to_sleep = 0.001
            # if i == 1:
            #     plt.waitforbuttonpress()
            # else:
            #     plt.pause(time_to_sleep)
            plt.pause(time_to_sleep)

            indices_to_plot = np.arange(0, episode_array.shape[0], 1)
            indices_to_plot = indices_to_plot[indices_to_plot % frame_rate != 0]
            vel_plot_data = episode_array[indices_to_plot, :]

            if i < vel_plot_data.shape[0]-1:
                axes[1].scatter(i, vel_plot_data[i,3], label='Ego', color = 'blue')
                axes[1].scatter(i, vel_plot_data[i,3] + vel_plot_data[i,8], label='NPC', color = 'red')


            # axes[1].plot(vel_plot_data[:i,3], label='NPC', color = 'blue')
            # axes[1].plot(vel_plot_data[:i,3] + vel_plot_data[:i,8], label='Ego', color = 'red')
            axes[1].legend(['EGO', 'NPC'])
            plt.draw()
            writer.grab_frame()
            
    axes[0].clear()
    axes[1].clear()

def plot_traj_history(episode_array, framerate, 
                      min_idx = 0, max_idx = -1,
                      ego_size=(2, 5), npc_size=(2, 5), padding = 5, annotate = True, annotation_offset=(0, 1.0)):
    """
    Plots the trajectory history of ego and NPC vehicles using bounding boxes.

    Parameters:
    - episode_array: numpy.ndarray
        Array containing trajectory data. Expected shape: (time_steps, features).
        Columns used:
            Ego:
                x position: column 1
                y position: column 2
                velocity x: column 3
                velocity y: column 4
            NPC:
                relative x position: column 6
                relative y position: column 7
                relative velocity x: column 8
                relative velocity y: column 9
    - framerate: int
        Determines the frame interval for plotting.
    - ego_size: tuple, optional
        (width, height) of the ego vehicle bounding box. Default is (2, 4).
    - npc_size: tuple, optional
        (width, height) of the NPC vehicle bounding box. Default is (2, 4).
    """

    episode_array = episode_array[min_idx:max_idx,:]
    
    # Create an action mask based on framerate
    action_mask = np.mod(np.arange(episode_array.shape[0]), framerate)
    action_mask = np.where(action_mask == 0, 0, 1)
    print("Action Mask:", action_mask)

    # Apply the mask to reduce the number of frames plotted
    episode_array = episode_array[action_mask == 1, :]

    # Extract ego vehicle data
    ego_x = episode_array[:, 1]
    ego_y = episode_array[:, 2]
    ego_vx = episode_array[:, 3]
    ego_vy = episode_array[:, 4]
    ego_heading = np.arctan2(ego_vy, ego_vx) * (180 / np.pi)  # Convert to degrees

    # Extract NPC vehicle data (assuming relative positions and velocities)
    npc_x = episode_array[:, 6] + episode_array[:, 1]
    npc_y = episode_array[:, 7] + episode_array[:, 2]
    npc_vx = episode_array[:, 8] + episode_array[:, 3]
    npc_vy = episode_array[:, 9] + episode_array[:, 4]
    npc_heading = np.arctan2(npc_vy, npc_vx) * (180 / np.pi)  # Convert to degrees

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Function to add bounding boxes
    def add_bounding_boxes(x, y, heading, vx, size, edgecolor, label, annotation_text):
        max_idx = x.shape[0]
        # max_idx = 100
        min_idx = 0
        for idx, (xi, yi, hi, vi) in enumerate(zip(x, y, heading, vx)):
            if idx % 10 != 0:
                continue
            if idx < min_idx:
                continue
            if idx > max_idx:
                break
            # Create a rectangle centered at (xi, yi)
            rect_alpha = max((idx - min_idx) / (max_idx - min_idx), 0.3)
            rect = patches.Rectangle(
                (xi - size[0]/2, yi - size[1]/2),  # Lower-left corner
                size[0],  # Width
                size[1],  # Height
                angle=hi,  # Rotation angle
                linewidth=1,
                edgecolor=edgecolor,
                facecolor=edgecolor,
                alpha=rect_alpha,
                label=label if idx == 0 else ""  # Label only once for legend
            )
            ax.add_patch(rect)
            if annotate:
                annotation_text = f't = {idx}'
                text_x = xi
                text_y = yi - annotation_offset[1]
                ax.text(
                    text_x, text_y, annotation_text,
                    fontsize=8,
                    color='black',
                    ha='left',
                    va='top'
                )

    # Add ego vehicle bounding boxes
    add_bounding_boxes(ego_x, ego_y, ego_heading-90, ego_vx, ego_size, 'green', 'Ego Vehicle', 'Ego')

    # Add NPC vehicle bounding boxes
    add_bounding_boxes(npc_x, npc_y, npc_heading-90, npc_vx, npc_size, 'blue', 'NPC Vehicle', 'NPC')

    # Calculate plot limits considering bounding box sizes
    all_x = np.concatenate([ego_x, npc_x])
    all_y = np.concatenate([ego_y, npc_y])
    max_width = max(ego_size[0], npc_size[0])
    max_height = max(ego_size[1], npc_size[1])

    x_min = np.min(all_x) - max_width / 2 - padding
    x_max = np.max(all_x) + max_width / 2 + padding
    y_min = np.min(all_y) - max_height / 2 - padding
    y_max = np.max(all_y) + max_height / 2 + padding

    y_min = -8
    y_max = 6

    # Max of x and y limits for Bounding Box
    # x_max = 160

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)


    # Draw the Road
    ax.hlines(4, x_min - 5, x_max + 5, colors='k')
    ax.hlines(-1, x_min - 5, x_max + 5, colors='k', linestyles='dashed')
    ax.hlines(-6, x_min - 5, x_max + 5, colors='k')

    # Set plot attributes
    ax.set_aspect('equal')
    ax.set_xlabel('X Position', fontsize=label_fontsize)
    ax.set_ylabel('Y Position', fontsize=label_fontsize)
    # ax.set_title('Trajectory History with Bounding Boxes', fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # ax.grid(True)

    # Create a custom legend to avoid duplicate labels
    # Define custom legend handles
    legend_elements = [
        patches.Patch(edgecolor='green', facecolor='green', label='Ego Vehicle'),
        patches.Patch(edgecolor='blue', facecolor='blue', label='NPC Vehicle')
    ]

    # Add the custom legend to the plot
    ax.legend(handles=legend_elements)

    # Display the plot
    plt.savefig(os.path.join(fig_dir, 'forward_left_no_collision.pdf'))
    plt.show()



trajectory_path = 'sanity-check-bl/trajectories_e1v1/EGO'
file_name = '0.json'

with open(os.path.join(trajectory_path, file_name), 'r') as f:
    data = json.load(f)

episode_keys = list(data.keys())
num_episodes = len(episode_keys)

framerate = 15

fig, axes = plt.subplots(1, 2, figsize=(20, 6))
# fig, axes = plt.subplots(1, 2, figsize=(20, 6))
time_buffer = []
# episode_array = np.asarray(data[episode_keys[0]])
# plot_traj_history(episode_array, framerate, annotate=False)


# Play Specific Episode
# episode_num = 1
# episode_array = np.asarray(data[episode_keys[episode_num]])
# plot_trajectories(episode_array, framerate, fig, axes, time_buffer, episode_keys[episode_num], video_path)

video_path = os.path.join(trajectory_path, 'videos')
behind_path = os.path.join(video_path, 'behind')
forward_path = os.path.join(video_path, 'forward')
adjacent_path = os.path.join(video_path, 'adjacent')
if not os.path.exists(behind_path):
    os.makedirs(behind_path)
if not os.path.exists(forward_path):
    os.makedirs(forward_path)
if not os.path.exists(adjacent_path):    
    os.makedirs(adjacent_path)

# Play all episodes
for i in np.arange(1,num_episodes, 1):
    episode_num = i
    episode_array = np.asarray(data[episode_keys[episode_num]])
    plot_trajectories(episode_array, framerate, fig, axes, time_buffer, episode_keys[episode_num], video_path)