import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
training_runs = api.runs(
    "amar-research/safetyh",
    filters={
        "config.model_sampling": "prioritized",
        "config.eval": False,
        "tags": "sfsp-continuouselo",
    },
)

episode_dict = {}
episode_dict["train_ego"] = {}
episode_dict["train_npc"] = {}
episode_dict["train_ego"]["cum_ep"] = {}
episode_dict["train_npc"]["cum_ep"] = {}

history_dict = {}
history_dict["train_ego"] = {}
history_dict["train_npc"] = {}

max_ego_version = 0
max_npc_version = 0

for run in tqdm(training_runs, desc="Processing Training Runs"):
    train_ego = run.config["train_ego"]
    ego_version = run.config["ego_version"]
    npc_version = run.config["npc_version"]
    history = run.scan_history()
    episodes = history.max_step

    if ego_version > max_ego_version:
        max_ego_version = ego_version
    if npc_version > max_npc_version:
        max_npc_version = npc_version

    if train_ego:
        episode_dict["train_ego"][ego_version] = episodes
        history_dict["train_ego"][ego_version] = history
    else:
        episode_dict["train_npc"][npc_version] = episodes
        history_dict["train_npc"][npc_version] = history

cum_eps_ego = [0]
cum_eps_npc = [0]
for i in range(1, max_ego_version + 1):
    cum_eps_ego.append(episode_dict["train_ego"][i] + cum_eps_ego[-1])

for i in range(1, max_npc_version + 1):
    cum_eps_npc.append(episode_dict["train_npc"][i] + cum_eps_npc[-1])

npc_elos = np.zeros((max_ego_version, cum_eps_ego[-1]))

for i in tqdm(range(1, max_ego_version + 1), desc="Concatenating NPC ELOs"):
    history = history_dict["train_ego"][i]
    episodes = episode_dict["train_ego"][i]
    start_idx = cum_eps_ego[i - 1]
    for idx, scan in tqdm(
        enumerate(history),
        leave=False,
        desc=f"Slicing from {start_idx} to {start_idx + episodes}",
        total=episodes,
    ):
        for j in range(1, i + 1):
            npc_elos[j - 1, start_idx + idx] = scan[f"rollout/model_{j - 1}_elo"]


plt.figure()
for i in range(1, max_npc_version):
    plt.plot(npc_elos[i - 1, :], label=f"V{i}")
plt.xlabel("Episodes")
plt.ylabel("ELO")
plt.legend()
plt.grid()
plt.show()
