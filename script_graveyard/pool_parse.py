import json
import matplotlib.pyplot as plt
import numpy as np

def parse_pool(npc_file, ego_file):

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


    npc_elo_arr = np.ones((len(npc_elo),n_cycles+2))*1000.0
    ego_elo_arr = np.ones((len(ego_elo),n_cycles+1))*1000.0

    npc_elo_agent_freq = np.zeros(n_cycles+2)
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

    return npc_elo_arr, ego_elo_arr, npc_elo_agent_freq, ego_elo_agent_freq


npc_files = ['npc_pool_adj_all_1_two_model.json']
ego_files = ['ego_pool_adj_all_1_two_model.json']
eps = [1.0]
npc_elo_arr = []
ego_elo_arr = []
npc_elo_agent_freq = []
ego_elo_agent_freq = []

for i in range(len(npc_files)):
    npc_file = open(npc_files[i])
    ego_file = open(ego_files[i])

    npc_elo, ego_elo, npc_freq, ego_freq = parse_pool(npc_file, ego_file)

    npc_elo_arr.append(npc_elo)
    ego_elo_arr.append(ego_elo)
    npc_elo_agent_freq.append(npc_freq)
    ego_elo_agent_freq.append(ego_freq)


# for v in range(4):
#     for i in range(len(eps)):
#         plt.plot(npc_elo_arr[i][:,v], label=f'NPC_{v} - {eps[i]}', linestyle='solid')
#         if(v == 0):
#             plt.plot(ego_elo_arr[i][:,v], label=f'Ego_{v} - MOBIL - {eps[i]}', linestyle='solid')
#         else:
#             plt.plot(ego_elo_arr[i][:,v], label=f'Ego_{v} - {eps[i]}', linestyle='solid')
#         plt.vlines(x=ego_elo_agent_freq[i][:], ymin=0, ymax=2000, color='r', linestyle='--')
#         plt.vlines(x=npc_elo_agent_freq[i][:], ymin=0, ymax=2000, color='b', linestyle='--')

#     plt.legend()
#     plt.grid()
#     plt.title(f'Agent {v} ELO')
#     plt.xlabel('Episode')
#     plt.ylabel('ELO')
#     plt.show()


for i in range(len(eps)):
    for v in range(2):
        if(v == 0):
            plt.plot(npc_elo_arr[i][:,v], label=f'NPC - MOBIL', linestyle='solid')
            plt.plot(ego_elo_arr[i][:,v], label=f'Ego_{v} - MOBIL', linestyle='solid')
        else:
            plt.plot(npc_elo_arr[i][:,v], label=f'NPC_{v-1}', linestyle='solid')
            plt.plot(ego_elo_arr[i][:,v], label=f'Ego_{v}', linestyle='solid')
        plt.vlines(x=ego_elo_agent_freq[i][:], ymin=0, ymax=2000, color='r', linestyle='--')
        plt.vlines(x=npc_elo_agent_freq[i][:], ymin=0, ymax=2000, color='b', linestyle='--')
    

    plt.plot(npc_elo_arr[i][:,2], label=f'NPC_{4-1}', linestyle='solid')

    plt.legend()
    plt.grid()
    plt.title(f'ELO Progression - eps_start {eps[i]}')
    plt.xlabel('Episode')
    plt.ylabel('ELO')
    plt.show()