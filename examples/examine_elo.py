import json
import numpy as np
import matplotlib.pyplot as plt
import os

figure_dir = "figures"
fig_dir = os.path.join(os.getcwd(), figure_dir)
results_dir = "results"
res_dir = os.path.join(os.getcwd(), results_dir)

tick_fontsize = 14
label_fontsize = 14
legend_fontsize = 14
title_fontsize = 16

with open("prioritized_sampling_5/ego_pool_1_prioritized.json") as f:
    ego_pool_data = json.load(f)
with open("prioritized_sampling_5/npc_pool_1_prioritized.json") as f:
    npc_pool_data = json.load(f)


# with open('ego_pool_adj_all_1_two_model.json') as f:
#     ego_pool_data = json.load(f)
# with open('npc_pool_adj_all_1_two_model.json') as f:
#     npc_pool_data = json.load(f)

ego_cycles = list(ego_pool_data.keys())
npc_cycles = list(npc_pool_data.keys())


max_ego_models = len(ego_pool_data[ego_cycles[-1]]["0"]["model_elo"])
max_npc_models = len(npc_pool_data[npc_cycles[-1]]["0"]["model_elo"])

ego_pool_elo = np.zeros((0, 6 + max_ego_models))
npc_pool_elo = np.zeros((0, 6 + max_npc_models))

eval_idx = 0
for cycle in ego_cycles:
    ego_pool_elo = np.append(
        ego_pool_elo, np.zeros((len(ego_pool_data[cycle]), 6 + max_ego_models)), axis=0
    )
    for eval_iter in ego_pool_data[cycle]:
        eval_key = str(eval_iter)
        ego_pool_elo[eval_idx, 0] = cycle
        ego_pool_elo[eval_idx, 1] = ego_pool_data[cycle][eval_key]["Sa"]
        ego_pool_elo[eval_idx, 2] = ego_pool_data[cycle][eval_key]["Sb"]
        ego_pool_elo[eval_idx, 3] = ego_pool_data[cycle][eval_key]["model_idx"]
        ego_pool_elo[eval_idx, 4] = ego_pool_data[cycle][eval_key]["opponent_model"]
        ego_pool_elo[eval_idx, 5] = ego_pool_data[cycle][eval_key]["opponent_elo"]
        ego_pool_elo[eval_idx, 6:] = 1000.0
        for idx, elo in enumerate(ego_pool_data[cycle][eval_key]["model_elo"]):
            ego_pool_elo[eval_idx, 6 + idx] = elo
        eval_idx += 1

eval_idx = 0
for cycle in npc_cycles:
    npc_pool_elo = np.append(
        npc_pool_elo, np.zeros((len(npc_pool_data[cycle]), 6 + max_npc_models)), axis=0
    )
    for eval_iter in npc_pool_data[cycle]:
        eval_key = str(eval_iter)
        npc_pool_elo[eval_idx, 0] = cycle
        npc_pool_elo[eval_idx, 1] = npc_pool_data[cycle][eval_key]["Sa"]
        npc_pool_elo[eval_idx, 2] = npc_pool_data[cycle][eval_key]["Sb"]
        npc_pool_elo[eval_idx, 3] = npc_pool_data[cycle][eval_key]["model_idx"]
        npc_pool_elo[eval_idx, 4] = npc_pool_data[cycle][eval_key]["opponent_model"]
        npc_pool_elo[eval_idx, 5] = npc_pool_data[cycle][eval_key]["opponent_elo"]
        npc_pool_elo[eval_idx, 6:] = 1000.0
        for idx, elo in enumerate(npc_pool_data[cycle][eval_key]["model_elo"]):
            npc_pool_elo[eval_idx, 6 + idx] = elo
        eval_idx += 1


# Define cmap for each cycle in ego_cycles
cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in np.linspace(0, 1, len(ego_cycles))]
# max_ego_models = 3
# max_npc_models = 2
# ego_cycles = [1, 2]
# npc_cycles = [1, 2]
max_idx = 0
fig, ax = plt.subplots(2, 1, figsize=(15, 5), sharex=True)
for ego_version in range(max_ego_models):
    ax[0].plot(ego_pool_elo[:, 6 + ego_version], label=f"E{ego_version}")
for npc_version in range(max_npc_models):
    ax[1].plot(npc_pool_elo[:, 6 + npc_version], label=f"V{npc_version + 1}")
for c in ego_cycles:
    cycle_indices = np.where(ego_pool_elo[:, 0] == float(c))
    # ax[0].axvspan(np.min(cycle_indices), np.max(cycle_indices)+1, alpha=0.2, color=colors[ego_cycles.index(c)])
    if np.max(cycle_indices) > max_idx:
        max_idx = np.max(cycle_indices)
for c in npc_cycles:
    cycle_indices = np.where(npc_pool_elo[:, 0] == float(c))
    # ax[1].axvspan(np.min(cycle_indices), np.max(cycle_indices)+1, alpha=0.2, color=colors[npc_cycles.index(c)])
    if np.max(cycle_indices) > max_idx:
        max_idx = np.max(cycle_indices)
ax[0].axvspan(0, 100, alpha=0.2, color=colors[0])
ax[0].axvspan(100, 200, alpha=0.2, color=colors[1])
ax[0].axvspan(200, 400, alpha=0.2, color=colors[2])
ax[0].axvspan(400, 600, alpha=0.2, color=colors[3])
ax[1].axvspan(0, 100, alpha=0.2, color=colors[0])
ax[1].axvspan(100, 200, alpha=0.2, color=colors[1])
ax[1].axvspan(200, 400, alpha=0.2, color=colors[2])
ax[1].axvspan(400, 600, alpha=0.2, color=colors[3])
ax[0].set_title("Ego Pool Elo", fontsize=title_fontsize)
ax[0].legend(fontsize=legend_fontsize, loc="upper right")
ax[0].grid()
ax[1].set_title("NPC Pool Elo", fontsize=title_fontsize)
ax[1].legend(fontsize=legend_fontsize, loc="upper right")
ax[1].grid()
ax[0].set_xlim([0, max_idx])
ax[1].set_xlim([0, max_idx])
ax[0].set_ylim([500, 1300])
ax[1].set_ylim([500, 1300])
ax[0].set_ylabel("Elo", fontsize=label_fontsize)
ax[1].set_ylabel("Elo", fontsize=label_fontsize)
ax[1].set_xlabel("Episode", fontsize=label_fontsize)
plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'elo_for_2_cycle.pdf'))
plt.show()
