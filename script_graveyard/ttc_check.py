import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import json
import time

trajectory_path = "trajectories"
file_name = "trajectories_ego_0.json"

with open(os.path.join(trajectory_path, file_name), "r") as f:
    data = json.load(f)

episode_keys = list(data.keys())
episode_num = episode_keys[-2]
episode_array = np.asarray(data[episode_num])
dt = 1 / 15

num_episodes = len(episode_keys)
print(f"Number of Episodes: {num_episodes}")
episode_avg_ttc = np.zeros((num_episodes, 2))

for i, episode in enumerate(episode_keys):
    episode_array = np.asarray(data[episode])
    ttc_x = np.clip(episode_array[:, -2], -100, 100)
    ttc_y = np.clip(episode_array[:, -1], -100, 100)
    episode_avg_ttc[i, 0] = np.mean(ttc_x)
    episode_avg_ttc[i, 1] = np.mean(ttc_y)
    plt.plot(ttc_x)
    plt.plot(ttc_y)

plt.gca().set_ylim(-100, 100)
plt.show()

plt.hist(episode_avg_ttc[:, 0], bins=50)
plt.show()
plt.hist(episode_avg_ttc[:, 1], bins=50)
plt.show()
