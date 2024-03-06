import gymnasium as gym
import math
import random
import numpy as np
from tqdm import tqdm

from config import load_config

from enum import Enum

class Action(Enum):
  LANE_LEFT = 0
  IDLE = 1
  LANE_RIGHT = 2
  FASTER = 3
  SLOWER = 4

class CutInState(Enum):
  EVADE = 0
  ACCELERATE = 1
  CUTIN = 2
  BRAKE = 3
  END = 4


na_env_cfg = load_config("env_configs/cut_in.yaml")

env = gym.make('crash-v0', config=na_env_cfg, render_mode='rgb_array')


while True:
  done = truncated = False
  obs, info = env.reset()
  ego_x = obs[0,1]
  ego_y = obs[0,2]
  ego_vx = obs[0,3]
  ego_vy = obs[0,4]

  npc_x = obs[1,1]
  npc_y = obs[1,2]
  npc_vx = obs[1,3]
  npc_vy = obs[1,4]
  maneuver  = CutInState.EVADE
  action = Action.IDLE.value
  end_frames = 0
  while not (done or truncated or end_frames > 2):
    
    if 'int_frames' in info.keys():
      state = info['int_frames'][-1,:]
      ego_x = state[1]
      ego_y = state[2]
      ego_vx = state[3]
      ego_vy = state[4]
      npc_x = state[6]
      npc_y = state[7]
      npc_vx = state[8]
      npc_vy = state[9]
    
    print(maneuver, npc_x, npc_y)
    if maneuver == CutInState.EVADE:
      if abs(npc_y) < 1:
        print("Cutting Out")
        if abs(0-ego_y) < 0.2:
          action = Action.LANE_RIGHT.value
        elif abs(4-ego_y) < 0.2:
          action = Action.LANE_LEFT.value
        maneuver = CutInState.ACCELERATE
      else:
        maneuver = CutInState.ACCELERATE
    elif maneuver == CutInState.ACCELERATE:
      action = Action.FASTER.value
      if npc_x <= -3.0:
        maneuver = CutInState.CUTIN
        if npc_y <= 0:
          action = Action.LANE_LEFT.value
        else:
          action = Action.LANE_RIGHT.value
    elif maneuver == CutInState.CUTIN:
      maneuver = CutInState.END
    else:
      end_frames += 1
      action = Action.IDLE.value


    obs, reward, done, truncated, info = env.step(action)


    env.render()