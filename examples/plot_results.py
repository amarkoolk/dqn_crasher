import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from wandb.apis.public import Runs

api = wandb.Api()
training_runs: Runs = api.runs("amar-research/safetyh", filters={"tags": "bl_fl"})

eval_runs = {}
for run in training_runs:
    config = run.config
    history = run.scan_history()
    ego_version = config["ego_version"]
    npc_version = config["npc_version"]
    # ego_version = 0
    # npc_version = 0
    if ego_version not in eval_runs:
        eval_runs[ego_version] = {}

    eval_runs[ego_version][npc_version] = run.history(
        samples=history.max_step, x_axis="_step", pandas=(True), stream="default"
    )

# SCENARIO
scenarios = [
    "behind_left",
    "behind_right",
    "behind_center",
    "forward_left",
    "forward_right",
    "forward_center",
    "adjacent_left",
    "adjacent_right",
]
scenarios = ["behind_left", "forward_left"]

ego_version = 0
npc_version = 1

run = eval_runs[ego_version][npc_version]

fig, ax = plt.subplots(3, 1)
sr_name = "rollout/sr100"
ego_speed_name = "rollout/ego_speed_mean"
npc_speed_name = "rollout/npc_speed_mean"

# Replace all NaNs with 0
run[sr_name] = run[sr_name].fillna(0)
run[ego_speed_name] = run[ego_speed_name].fillna(0)
run[npc_speed_name] = run[npc_speed_name].fillna(0)

# Smooth the data using Exponential Moving Average
run[sr_name] = run[sr_name].ewm(span=500).mean()
run[ego_speed_name] = run[ego_speed_name].ewm(span=500).mean()
run[npc_speed_name] = run[npc_speed_name].ewm(span=500).mean()

# Plot
ax[0].plot(run[sr_name], label=f"{ego_version}-{npc_version}")
ax[1].plot(run[ego_speed_name], label=f"{ego_version}-{npc_version}")
ax[2].plot(run[npc_speed_name], label=f"{ego_version}-{npc_version}")

ax[0].set_title("Success Rate")
ax[0].set_ylabel("Success Rate")
ax[0].grid(axis="y")
ax[0].set_ylim(-0.05, 1.05)

ax[1].set_title("Ego Speed")
ax[1].set_ylabel("Speed (m/s)")
ax[1].grid(axis="y")

ax[2].set_title("NPC Speed")
ax[2].set_ylabel("Speed (m/s)")
ax[2].grid(axis="y")

plt.xlabel("Episode")
plt.legend()
plt.show()


fig, ax = plt.subplots(3, 1)
for scenario in scenarios:
    sr_name = "rollout/" + scenario + "/sr100"
    ego_speed_name = "rollout/" + scenario + "/ego_speed_mean"
    npc_speed_name = "rollout/" + scenario + "/npc_speed_mean"

    # Remove NaNs from the data frame
    sr_nans = run[sr_name].isna()
    ego_speed_nans = run[ego_speed_name].isna()
    npc_speed_nans = run[npc_speed_name].isna()

    sr_step = run["_step"][~sr_nans]
    sr_data = run[sr_name][~sr_nans]

    ego_speed_step = run["_step"][~ego_speed_nans]
    ego_speed_data = run[ego_speed_name][~ego_speed_nans]

    npc_speed_step = run["_step"][~npc_speed_nans]
    npc_speed_data = run[npc_speed_name][~npc_speed_nans]

    # Smooth the data using Exponential Moving Average
    sr_data = sr_data.ewm(span=200).mean()
    ego_speed_data = ego_speed_data.ewm(span=200).mean()
    npc_speed_data = npc_speed_data.ewm(span=200).mean()

    # Plot
    ax[0].plot(sr_step, sr_data, label=scenario)
    ax[1].plot(ego_speed_step, ego_speed_data, label=scenario)
    ax[2].plot(npc_speed_step, npc_speed_data, label=scenario)

ax[0].set_title("Success Rate")
ax[0].set_ylabel("Success Rate")
ax[0].grid(axis="y")
ax[0].set_ylim(-0.05, 1.05)

ax[1].set_title("Ego Speed")
ax[1].set_ylabel("Speed (m/s)")
ax[1].grid(axis="y")

ax[2].set_title("NPC Speed")
ax[2].set_ylabel("Speed (m/s)")
ax[2].grid(axis="y")

plt.xlabel("Episode")
plt.legend()
plt.show()
