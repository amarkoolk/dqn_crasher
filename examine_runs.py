import pandas as pd
import numpy as np
import json
import ast

eval_runs_reset = []
eval_runs_continuous = []
eval_runs_uniform = []
eval_runs_sequential = []
max_ego_version = 0
max_npc_version = 0

df = pd.read_csv('project.csv')
timestamps = []
for ind in df.index:
    summary_entry_dict = None
    config_entry_dict = None
    attr_entry_dict = None
    tag_entry_dict = None
    summary_entry = df['summary'][ind]
    if isinstance(summary_entry, str):
        summary_entry = summary_entry.replace("'", "\"")
        summary_entry = summary_entry.replace("True", "true")
        summary_entry = summary_entry.replace("False", "false")
        summary_entry_dict = json.loads(summary_entry)
    config_entry = df['config'][ind]
    if isinstance(config_entry, str):
        config_entry = config_entry.replace("'", "\"")
        config_entry = config_entry.replace("True", "true")
        config_entry = config_entry.replace("False", "false")
        config_entry_dict = json.loads(config_entry)

    tag_entry_list = df['tags'][ind]
    
    name = df['name'][ind]
    # Safely evaluate the string to a Python object if it's not already one
    # if isinstance(attr_entry, str):
    #     # Attempt to directly parse as JSON first
    #     try:
    #         attr_entry_dict = json.loads(attr_entry)
    #     except json.JSONDecodeError:
    #         # If direct JSON parsing fails, try fixing Python-specific notations
    #         try:
    #             # Use ast.literal_eval for safe evaluation of Python literals
    #             attr_entry_python = ast.literal_eval(attr_entry)
    #             # Convert back to a string using json.dumps to ensure JSON compatibility
    #             attr_entry = json.dumps(attr_entry_python)
    #             attr_entry_dict = json.loads(attr_entry)
    #         except (ValueError, SyntaxError):
    #             print("Error evaluating or converting the attribute entry.")
    # else:
    #     attr_entry_dict = attr_entry  # If it's already a dict, no need to parse


    if isinstance(summary_entry_dict, dict) and isinstance(config_entry_dict, dict):
        config_keys = config_entry_dict.keys()
        if 'eval' in config_keys and 'model_sampling' in config_keys and 'adjustable_k' in config_keys:
            # if config_entry_dict['eval'] and config_entry_dict['model_sampling'] == 'prioritized' and ('sfsp-resetelo' in attr_entry_dict['tags']):
            #     timestamps.append(summary_entry_dict['_timestamp'])
            #     if summary_entry_dict['_runtime'] < 1000.0:
            #         ego_v = config_entry_dict['ego_version']
            #         npc_v = config_entry_dict['npc_version']
            #         sr = summary_entry_dict['rollout/sr100']
            #         eval_runs_reset.append([ego_v, npc_v, sr])

            #         if ego_v > max_ego_version:
            #             max_ego_version = ego_v
            #         if npc_v > max_npc_version:
            #             max_npc_version = npc_v
            if config_entry_dict['eval'] and ('prio_all_5_cycle_eval' in tag_entry_list):
                timestamps.append(summary_entry_dict['_timestamp'])
                ego_v = config_entry_dict['ego_version']
                npc_v = config_entry_dict['npc_version']
                sr = summary_entry_dict['rollout/sr100']
                eval_runs_continuous.append([ego_v, npc_v, sr])

                if ego_v > max_ego_version:
                    max_ego_version = ego_v
                if npc_v > max_npc_version:
                    max_npc_version = npc_v

            if config_entry_dict['eval'] and config_entry_dict['model_sampling'] == 'uniform' and ('uniform_all_5_cycle_eval' in tag_entry_list):
                timestamps.append(summary_entry_dict['_timestamp'])
                ego_v = config_entry_dict['ego_version']
                npc_v = config_entry_dict['npc_version']
                sr = summary_entry_dict['rollout/sr100']
                eval_runs_uniform.append([ego_v, npc_v, sr])

                if ego_v > max_ego_version:
                    max_ego_version = ego_v
                if npc_v > max_npc_version:
                    max_npc_version = npc_v

            if config_entry_dict['eval']  and ('local_all_5_cycle_eval' in tag_entry_list):
                timestamps.append(summary_entry_dict['_timestamp'])
                ego_v = config_entry_dict['ego_version']
                npc_v = config_entry_dict['npc_version']
                sr = summary_entry_dict['rollout/sr100']
                eval_runs_sequential.append([ego_v, npc_v, sr])

                if ego_v > max_ego_version:
                    max_ego_version = ego_v
                if npc_v > max_npc_version:
                    max_npc_version = npc_v


eval_table_sequential = np.zeros((max_ego_version+1, max_npc_version+1))
eval_table_continuous = np.zeros((max_ego_version+1, max_npc_version+1))
eval_table_uniform = np.zeros((max_ego_version+1, max_npc_version+1))

for run in eval_runs_sequential:
    ego_v = run[0]
    npc_v = run[1]
    sr = run[2]
    eval_table_sequential[int(ego_v), int(npc_v)] = sr

for run in eval_runs_continuous:
    ego_v = run[0]
    npc_v = run[1]
    sr = run[2]
    eval_table_continuous[int(ego_v), int(npc_v)] = sr

for run in eval_runs_uniform:
    ego_v = run[0]
    npc_v = run[1]
    sr = run[2]
    eval_table_uniform[int(ego_v), int(npc_v)] = sr

import matplotlib.pyplot as plt
for i in range(0, max_npc_version+1):
    plt.plot(range(0, max_ego_version+1),eval_table_sequential[:, i], label=f'V{i+1}',linestyle='--', marker='o')
plt.xlabel('ego_version')
plt.ylabel('crash rate')
plt.xticks(range(0, max_ego_version+1))
plt.title('Sequential')
plt.legend()
plt.grid()
plt.show()
for i in range(0, max_ego_version+1):
    plt.plot(range(1, max_npc_version+2),eval_table_sequential[i, :], label=f'E{i}',linestyle='--', marker='o')
plt.xlabel('npc_version')
plt.ylabel('crash rate')
plt.xticks(range(1, max_npc_version+2))
plt.title('Sequential')
plt.legend()
plt.grid()
plt.show()

for i in range(0, max_npc_version+1):
    plt.plot(range(0, max_ego_version+1),eval_table_continuous[:, i], label=f'V{i+1}',linestyle='--', marker='o')
plt.xlabel('ego_version')
plt.ylabel('crash rate')
plt.xticks(range(0, max_ego_version+1))
plt.title('Prioritized Sampling')
plt.legend()
plt.grid()
plt.show()
for i in range(0, max_ego_version+1):
    plt.plot(range(1, max_npc_version+2),eval_table_continuous[i, :], label=f'E{i}',linestyle='--', marker='o')
plt.xlabel('npc_version')
plt.ylabel('crash rate')
plt.xticks(range(1, max_npc_version+2))
plt.title('Prioritized Sampling')
plt.legend()
plt.grid()
plt.show()

for i in range(0, max_npc_version+1):
    plt.plot(range(0, max_ego_version+1),eval_table_uniform[:, i], label=f'V{i+1}',linestyle='--', marker='o')
plt.xlabel('ego_version')
plt.ylabel('crash rate')
plt.xticks(range(0, max_ego_version+1))
plt.title('Uniform Sampling')
plt.legend()
plt.grid()
plt.show()
for i in range(0, max_ego_version+1):
    plt.plot(range(1, max_npc_version+2),eval_table_uniform[i, :], label=f'E{i}',linestyle='--', marker='o')
plt.xlabel('npc_version')
plt.ylabel('crash rate')
plt.xticks(range(1, max_npc_version+2))
plt.title('Uniform Sampling')
plt.legend()
plt.grid()
plt.show()

# Plot Average Crash Rate per Ego/NPC Version
ego_avg_crash_rate = np.mean(eval_table_sequential, axis=1)
npc_avg_crash_rate = np.mean(eval_table_sequential, axis=0)

ego_avg_crash_rate_celo = np.mean(eval_table_continuous, axis=1)
npc_avg_crash_rate_celo = np.mean(eval_table_continuous, axis=0)

ego_avg_crash_rate_uniform = np.mean(eval_table_uniform, axis=1)
npc_avg_crash_rate_uniform = np.mean(eval_table_uniform, axis=0)

plt.plot(range(0, max_ego_version+1), ego_avg_crash_rate, label='Ego Version', linestyle='--', marker='o')
plt.plot(range(0, max_ego_version+1), ego_avg_crash_rate_celo, label='Ego Version Prioritized', linestyle='--', marker='o')
plt.plot(range(0, max_ego_version+1), ego_avg_crash_rate_uniform, label='Ego Version Uniform', linestyle='--', marker='o')
plt.xlabel('Version')
plt.ylabel('Average Crash Rate')
plt.xticks(range(0, max_ego_version+1))
plt.legend()
plt.grid()
plt.show()

plt.plot(range(1, max_npc_version+2), npc_avg_crash_rate, label='NPC Version', linestyle='--', marker='o')
plt.plot(range(1, max_npc_version+2), npc_avg_crash_rate_celo, label='NPC Version Prioritized', linestyle='--', marker='o')
plt.plot(range(1, max_npc_version+2), npc_avg_crash_rate_uniform, label='NPC Version Uniform', linestyle='--', marker='o')
plt.xlabel('Version')
plt.ylabel('Average Crash Rate')
plt.xticks(range(0, max_ego_version+1))
plt.legend()
plt.grid()
plt.show()


# Append Means of Each Axis to End of Each Axis
eval_table_sequential = np.append(eval_table_sequential, np.mean(eval_table_sequential, axis=0).reshape(1, -1), axis=0)
eval_table_sequential = np.append(eval_table_sequential, np.mean(eval_table_sequential, axis=1).reshape(-1, 1), axis=1)

eval_table_continuous = np.append(eval_table_continuous, np.mean(eval_table_continuous, axis=0).reshape(1, -1), axis=0)
eval_table_continuous = np.append(eval_table_continuous, np.mean(eval_table_continuous, axis=1).reshape(-1, 1), axis=1)

eval_table_uniform = np.append(eval_table_uniform, np.mean(eval_table_uniform, axis=0).reshape(1, -1), axis=0)
eval_table_uniform = np.append(eval_table_uniform, np.mean(eval_table_uniform, axis=1).reshape(-1, 1), axis=1)

# Plot and Annotate Heatmap with Means
fig, ax = plt.subplots()
cax = ax.matshow(eval_table_sequential)
fig.colorbar(cax)
for i in range(eval_table_sequential.shape[0]):
    for j in range(eval_table_sequential.shape[1]):
        text = ax.text(j, i, round(eval_table_sequential[i, j], 2), ha='center', va='center', color='w')
# Color the last row and column
ax.axhline(y=eval_table_sequential.shape[0]-1, color='r', linewidth=30, alpha = 0.5)
ax.axvline(x=eval_table_sequential.shape[1]-1, color='r', linewidth=30, alpha = 0.5)
# Add label for mean
ax.text(eval_table_sequential.shape[1]-1, eval_table_sequential.shape[0]-1, 'Mean', ha='center', va='center', color='r', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))
plt.xlabel('NPC Version')
plt.ylabel('Ego Version')
plt.title('Sequential')
plt.show()

fig, ax = plt.subplots()
cax = ax.matshow(eval_table_continuous)
fig.colorbar(cax)
for i in range(eval_table_continuous.shape[0]):
    for j in range(eval_table_continuous.shape[1]):
        text = ax.text(j, i, round(eval_table_continuous[i, j], 2), ha='center', va='center', color='w')
# Color the last row and column
ax.axhline(y=eval_table_continuous.shape[0]-1, color='r', linewidth=30, alpha = 0.5)
ax.axvline(x=eval_table_continuous.shape[1]-1, color='r', linewidth=30, alpha = 0.5)
# Add label for mean
ax.text(eval_table_continuous.shape[1]-1, eval_table_continuous.shape[0]-1, 'Mean', ha='center', va='center', color='r', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))
plt.xlabel('NPC Version')
plt.ylabel('Ego Version')
plt.title('Prioritized Sampling')
plt.show()

fig, ax = plt.subplots()
cax = ax.matshow(eval_table_uniform)
fig.colorbar(cax)
for i in range(eval_table_uniform.shape[0]):
    for j in range(eval_table_uniform.shape[1]):
        text = ax.text(j, i, round(eval_table_uniform[i, j], 2), ha='center', va='center', color='w')
# Color the last row and column
ax.axhline(y=eval_table_uniform.shape[0]-1, color='r', linewidth=30, alpha = 0.5)
ax.axvline(x=eval_table_uniform.shape[1]-1, color='r', linewidth=30, alpha = 0.5)
# Add label for mean
ax.text(eval_table_uniform.shape[1]-1, eval_table_uniform.shape[0]-1, 'Mean', ha='center', va='center', color='r', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round'))
plt.xlabel('NPC Version')
plt.ylabel('Ego Version')
plt.title('Uniform Sampling')
plt.show()