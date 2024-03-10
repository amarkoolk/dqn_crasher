import pandas as pd
import numpy as np
import json

eval_runs = []
max_ego_version = 0
max_npc_version = 0

df = pd.read_csv('project.csv')
for ind in df.index:
    summary_entry_dict = None
    config_entry_dict = None
    summary_entry = df['summary'][ind]
    if isinstance(summary_entry, str):
        summary_entry = summary_entry.replace("'", "\"")
        summary_entry_dict = json.loads(summary_entry)
    
    config_entry = df['config'][ind]
    if isinstance(config_entry, str):
        config_entry = config_entry.replace("'", "\"")
        config_entry = config_entry.replace("True", "true")
        config_entry = config_entry.replace("False", "false")
        config_entry_dict = json.loads(config_entry)
        
    if isinstance(summary_entry_dict, dict) and isinstance(config_entry_dict, dict):
        config_keys = config_entry_dict.keys()
        if 'eval' in config_keys:
            if not config_entry_dict['eval']:
                if summary_entry_dict['_runtime'] < 1000.0:
                    ego_v = config_entry_dict['ego_version']
                    npc_v = config_entry_dict['npc_version']
                    sr = summary_entry_dict['rollout/sr100']
                    eval_runs.append([ego_v, npc_v, sr])

                    if ego_v > max_ego_version:
                        max_ego_version = ego_v
                    if npc_v > max_npc_version:
                        max_npc_version = npc_v

eval_table = np.zeros((max_ego_version+1, max_npc_version+1))

for run in eval_runs:
    ego_v = run[0]
    npc_v = run[1]
    sr = run[2]
    eval_table[int(ego_v), int(npc_v)] = sr

print(eval_table)