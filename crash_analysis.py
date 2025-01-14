import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import json
import time
from enum import Enum

class Action(Enum):
  LANE_LEFT = 0
  IDLE = 1
  LANE_RIGHT = 2
  FASTER = 3
  SLOWER = 4

trajectory_path = 'local_falsification_hardening_traj_all_5/E1_V1_TrainEgo_False/NPC'
file_name = '0.json'
framerate = 15

with open(os.path.join(trajectory_path, file_name), 'r') as f:
    data = json.load(f)

episode_keys = list(data.keys())
num_episodes = len(episode_keys)

ttc_x = np.zeros((num_episodes, 600))
ttc_y = np.zeros((num_episodes, 600))
dttc_x_dt = np.zeros((num_episodes, 600))
dttc_y_dt = np.zeros((num_episodes, 600))
distance = np.zeros((num_episodes, 600))
d_distance_dt = np.zeros((num_episodes, 600))
n_frames = np.zeros(num_episodes)
ego_vy = np.zeros((num_episodes, 600))
npc_vy = np.zeros((num_episodes, 600))
reward = np.zeros((num_episodes, 600))
end_distance = np.zeros(num_episodes)

for i in range(num_episodes):
    episode_data = np.asarray(data[episode_keys[i]])
    episode_data = np.delete(episode_data, np.arange(0, episode_data.shape[0], framerate), axis=0)

    num_frames = episode_data.shape[0]
    n_frames[i] = num_frames
    ts = np.arange(0, num_frames, 1)/framerate
    ego_x = episode_data[:,1]
    ego_y = episode_data[:,2]
    ego_vx = episode_data[:,3]
    ego_vy[i,:num_frames] = episode_data[:,4]

    npc_x = episode_data[:,6] + episode_data[:,1]
    npc_y = episode_data[:,7] + episode_data[:,2]
    npc_vx = -episode_data[:,8] + episode_data[:,3]
    npc_vy[i,:num_frames] = -episode_data[:,9] + episode_data[:,4]

    ttc_x[i,:num_frames] = episode_data[:,-2]
    ttc_y[i,:num_frames] = episode_data[:,-1]

    distance[i,:num_frames] = np.sqrt((ego_x - npc_x)**2 + (ego_y - npc_y)**2)
    end_distance[i] = distance[i,num_frames-1]

    # Derivative of TTC
    dttc_x_dt[i,:num_frames] = np.gradient(ttc_x[i,:num_frames],ts)
    dttc_y_dt[i,:num_frames] = np.gradient(ttc_y[i,:num_frames],ts)

    #Derivative of Relative Distance
    d_distance_dt[i,:num_frames] = np.gradient(distance[i,:num_frames],ts)

    reward[i,:num_frames] = episode_data[:,-3]

frames = n_frames[n_frames >= 560]
nm = distance[n_frames >= 560,:]
for i in range(len(frames)):
    plt.plot(nm[i,:int(frames[i])], c = 'b', alpha = 0.1)
plt.grid()
plt.ylabel('distance')
plt.xlabel('Frame')
plt.title('Relative Distance')
plt.ylim(-10, 100)
plt.show()

# Determine which d_distance_dt rows pass from negative to positive
crossover_eps_positive = []
crossover_eps_negative = []
for i in range(num_episodes):
    if n_frames[i] < 560.0:
        continue
    for j in range(1, int(n_frames[i])):
        if d_distance_dt[i,j] > 0 and d_distance_dt[i,j-1] < 0:
            crossover_eps_positive.append(i)
        if d_distance_dt[i,j] < 0 and d_distance_dt[i,j-1] > 0:
            crossover_eps_negative.append(i)

fig,ax = plt.subplots(3,1)
for i in range(num_episodes):
    if n_frames[i] < 560.0:
        color = 'r'
        alpha = 0.01
    else:
        color = 'b'
        alpha = 0.01

    if np.any(distance[i,:num_frames] < 6.0) and n_frames[i] >= 560:
        color = 'g'
        alpha = 1.0
    # else:
    #     continue

    ax[0].plot(d_distance_dt[i,:], c = color, alpha = alpha)
    ax[1].plot(dttc_x_dt[i,:], c = color, alpha = alpha)
    ax[2].plot(dttc_y_dt[i,:], c = color, alpha = alpha)
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[0].set_ylabel('d_distance_dt')
ax[1].set_ylabel('dttc_x_dt')
ax[2].set_ylabel('dttc_y_dt')
fig.legend()
fig.suptitle('Derivatives TTC and Relative Distance')
fig.supxlabel('Frame')
plt.show()

fig,ax = plt.subplots(3,1)
ep = 300
ax[0].plot(distance[ep,:int(n_frames[ep])])
ax[1].plot(ttc_x[ep,:int(n_frames[ep])])
ax[2].plot(ttc_y[ep,:int(n_frames[ep])])
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[0].set_ylabel('distance')
ax[1].set_ylabel('ttcx')
ax[2].set_ylabel('ttcy')
fig.suptitle('TTC and Relative Distance - Episode {}'.format(ep))
fig.supxlabel('Frame')
plt.show()

# FFT of Derivative of Relative Distance
fig,ax = plt.subplots(1,1)
for ep in range(num_episodes):
    if n_frames[i] < 560.0:
        color = 'r'
        alpha = 0.01
    else:
        color = 'b'
        alpha = 0.01

    if np.any(distance[ep,:num_frames] < 6.0) and n_frames[ep] >= 560:
        color = 'g'
        alpha = 1.0

    Fs = 100
    N = int(n_frames[ep])
    f = np.linspace(0, Fs, N)
    Y = np.fft.fft(d_distance_dt[ep,:N])
    freq = np.fft.fftfreq(N, 1/Fs)
    ax.plot(freq, Y.real, c = color, alpha = alpha)
    ax.grid()
    ax.set_ylabel('Magnitude')
    ax.set_xlabel('Frequency [Hz]')
fig.suptitle('FFT of d_distance_dt')
plt.show()