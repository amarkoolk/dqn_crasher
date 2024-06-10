import numpy as np
import json
import os
from tqdm import tqdm

training_type = "PFSP"
cur_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(cur_path, training_type)
directories = os.listdir(dir_path)

mean_lateral_velocity_ego = []
mean_lateral_velocity_npc = []

for training_run in tqdm(directories):
    training_run_path = os.path.join(dir_path, training_run)

    trajectories = {}

    # Count Files in Training Run
    files = os.listdir(training_run_path)
    n_files = len(files)
    for i in tqdm(range(n_files), leave=False):
        file = os.path.join(training_run_path, f"{i}.json")
        with open(file, "r") as f:
            data = json.load(f)
            x = len(data.keys())
            eps = list(data.keys())
            for ep in eps:
                trajectories[ep] = np.asarray(data[ep])
                mean_lateral_velocity_ego.append(np.mean(trajectories[ep][:, 4]))
                mean_lateral_velocity_npc.append(np.mean(trajectories[ep][:, 9]))

mean_lateral_velocity_ego = np.asarray(mean_lateral_velocity_ego)
mean_lateral_velocity_npc = np.asarray(mean_lateral_velocity_npc)

import matplotlib.pyplot as plt

plt.hist(mean_lateral_velocity_ego, bins=100, alpha=0.5, label='ego')
plt.hist(mean_lateral_velocity_npc, bins=100, alpha=0.5, label='npc')
plt.legend(loc='upper right')
plt.show()



