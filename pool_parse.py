import json

npc_file = open('npc_pool_all.json')
ego_file = open('ego_pool_all.json')
npc_data = json.load(npc_file)
ego_data = json.load(ego_file)

cycles = list(npc_data.keys())
npc_elo = []
ego_elo = []

n_cycles = int(len(cycles)/2)

for cycle in cycles:
    episodes = list(npc_data[cycle].keys())
    for episode in episodes:
        npc_elo.append(npc_data[cycle][episode]['model_elo'])

for cycle in cycles:
    episodes = list(ego_data[cycle].keys())
    for episode in episodes:
        ego_elo.append(ego_data[cycle][episode]['model_elo'])

import matplotlib.pyplot as plt
import numpy as np

print(ego_elo)

npc_elo_arr = np.ones((len(npc_elo),n_cycles+1))*1000.0
ego_elo_arr = np.ones((len(ego_elo),n_cycles+1))*1000.0

npc_elo_agent_freq = np.zeros(n_cycles+1)
ego_elo_agent_freq = np.zeros(n_cycles+1)

for i in range(len(npc_elo)):
    npc_elo_agent_freq[int(len(npc_elo[i]))-1] += 1
    for j in range(len(npc_elo[i])):
        npc_elo_arr[i,j] = npc_elo[i][j]

for i in range(len(ego_elo)):
    ego_elo_agent_freq[int(len(ego_elo[i]))-1] += 1
    for j in range(len(ego_elo[i])):
        ego_elo_arr[i,j] = ego_elo[i][j]

npc_elo_agent_freq = np.cumsum(npc_elo_agent_freq)
ego_elo_agent_freq = np.cumsum(ego_elo_agent_freq)

for i in range(n_cycles+1):
    plt.plot(npc_elo_arr[:,i], label=f'NPC_{i}', linestyle='solid')
    if(i == 0):
        plt.plot(ego_elo_arr[:,i], label=f'Ego_{i} - MOBIL', linestyle='solid')
    else:
        plt.plot(ego_elo_arr[:,i], label=f'Ego_{i}', linestyle='solid')
    plt.vlines(x=ego_elo_agent_freq[i], ymin=0, ymax=2000, color='r', linestyle='--')
    plt.vlines(x=npc_elo_agent_freq[i], ymin=0, ymax=2000, color='b', linestyle='--')

plt.legend()
plt.grid()
plt.show()