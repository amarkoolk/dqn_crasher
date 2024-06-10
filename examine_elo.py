import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

with open('ego_pool_log_7.json') as f:
    ego_pool_data = json.load(f)
with open('npc_pool_log_7.json') as f:
    npc_pool_data = json.load(f)

ego_cycles = list(ego_pool_data.keys())
npc_cycles = list(npc_pool_data.keys())

print(ego_cycles)

max_ego_models = len(ego_pool_data[ego_cycles[-1]]['0']['model_elo'])
max_npc_models = len(npc_pool_data[npc_cycles[-1]]['0']['model_elo'])

ego_pool_elo = np.zeros((0,6+max_ego_models))
npc_pool_elo = np.zeros((0,6+max_npc_models))

eval_idx = 0
for cycle in ego_cycles:
    ego_pool_elo = np.append(ego_pool_elo, np.zeros((len(ego_pool_data[cycle]), 6+max_ego_models)), axis=0)
    for eval_iter in ego_pool_data[cycle]:
        eval_key = str(eval_iter)
        ego_pool_elo[eval_idx, 0] = cycle
        ego_pool_elo[eval_idx, 1] = ego_pool_data[cycle][eval_key]['Sa']
        ego_pool_elo[eval_idx, 2] = ego_pool_data[cycle][eval_key]['Sb']
        ego_pool_elo[eval_idx, 3] = ego_pool_data[cycle][eval_key]['model_idx']
        ego_pool_elo[eval_idx, 4] = ego_pool_data[cycle][eval_key]['opponent_model']
        ego_pool_elo[eval_idx, 5] = ego_pool_data[cycle][eval_key]['opponent_elo']
        ego_pool_elo[eval_idx, 6:] = 1000.0
        for idx, elo in enumerate(ego_pool_data[cycle][eval_key]['model_elo']):
            ego_pool_elo[eval_idx, 6+idx] = elo
        print(ego_pool_data[cycle][eval_key]['model_probability'])
        eval_idx += 1

eval_idx = 0
for cycle in npc_cycles:
    npc_pool_elo = np.append(npc_pool_elo, np.zeros((len(npc_pool_data[cycle]), 6+max_npc_models)), axis=0)
    for eval_iter in npc_pool_data[cycle]:
        eval_key = str(eval_iter)
        npc_pool_elo[eval_idx, 0] = cycle
        npc_pool_elo[eval_idx, 1] = npc_pool_data[cycle][eval_key]['Sa']
        npc_pool_elo[eval_idx, 2] = npc_pool_data[cycle][eval_key]['Sb']
        npc_pool_elo[eval_idx, 3] = npc_pool_data[cycle][eval_key]['model_idx']
        npc_pool_elo[eval_idx, 4] = npc_pool_data[cycle][eval_key]['opponent_model']
        npc_pool_elo[eval_idx, 5] = npc_pool_data[cycle][eval_key]['opponent_elo']
        npc_pool_elo[eval_idx, 6:] = 1000.0
        for idx, elo in enumerate(npc_pool_data[cycle][eval_key]['model_elo']):
            npc_pool_elo[eval_idx, 6+idx] = elo
        eval_idx += 1


# Define cmap for each cycle in ego_cycles
cmap = plt.get_cmap('tab20')
colors = [cmap(i) for i in np.linspace(0, 1, len(ego_cycles))]

fig, ax = plt.subplots(2,1, sharex=True)
for ego_version in range(max_ego_models):
    ax[0].plot(ego_pool_elo[:,6+ego_version], label=f"E{ego_version}")
for npc_version in range(max_npc_models):
    ax[1].plot(npc_pool_elo[:,6+npc_version], label=f"V{npc_version+1}")
for c in ego_cycles:
    cycle_indices = np.where(ego_pool_elo[:,0] == float(c))
    ax[0].axvspan(np.min(cycle_indices), np.max(cycle_indices)+1, alpha=0.2, color=colors[ego_cycles.index(c)])
for c in npc_cycles:
    cycle_indices = np.where(npc_pool_elo[:,0] == float(c))
    ax[1].axvspan(np.min(cycle_indices), np.max(cycle_indices)+1, alpha=0.2, color=colors[npc_cycles.index(c)])
ax[0].set_title('Ego Pool Elo')
ax[0].legend()
ax[0].grid()
ax[1].set_title('NPC Pool Elo')
ax[1].legend()
ax[1].grid()
plt.show()

