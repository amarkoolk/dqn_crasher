import numpy as np
import itertools as it
from tqdm import tqdm
filename = '/home/amar/rl_crash_course/src/dqn_crasher/wandb/latest-run/files/episode_rollouts.log'

lines = []
temp = []
i = 0
with open(filename, 'r') as f:
    for key,group in tqdm(it.groupby(f, lambda line: line.startswith("INFO"))):
        if key:
            i+=1
            lines.append(list(map(''.join, temp)))
            temp = list(map(''.join, group))
        if not key:
            temp.append(list(map(''.join, group)))

print(i)
print(len(lines))
episodes = {}

for line in tqdm(lines[1:]):
    try:
        split_line = (line[0] + line[1]).split(',')
        for i,split in enumerate(split_line):
            if 'dtype' in split:
                split_line.pop(i)
        episode = split_line[0].split(':')[-1]
        if episode not in episodes.keys():
            episodes[episode] = {}
            episodes[episode]['obs'] = []
            episodes[episode]['action'] = []
            episodes[episode]['reward'] = []
            episodes[episode]['done'] = []
            episodes[episode]['info'] = []
        step = split_line[1].split(':')[-1]
        episodes[episode]['obs'].append(np.reshape(np.asarray([x for x in split_line[2].split(':')[-1].replace('\n','').replace(']','').replace('[','').split(' ') if x != '']), (5, 5)))
        episodes[episode]['action'].append(split_line[3].split(':')[-1].replace('tensor','').replace(']','').replace('[','').replace('(','').replace(')',''))
        episodes[episode]['reward'].append(split_line[4].split(':')[-1].replace('tensor','').replace(']','').replace('[','').replace('(','').replace(')',''))
        episodes[episode]['done'].append(split_line[5].split(':')[-1] == 'True')
        episodes[episode]['info'].append(eval(','.join(split_line[6:])[5:].replace('inf','0.0').replace('nan','0.0')))
    except:
        print('Error in line:')
        for i,split in enumerate(split_line):
            print('{} {}'.format(i,split))
        break

