from enum import Enum
import numpy as np

class Action(Enum):
  LANE_LEFT = 0
  IDLE = 1
  LANE_RIGHT = 2
  FASTER = 3
  SLOWER = 4



class Scenario:

  def set_state(self, state):
    raise NotImplementedError
  
  def get_action(self):
    raise NotImplementedError
  
class CutIn(Scenario):

  class CutInManeuver(Enum):
    EVADE = 0
    ACCELERATE = 1
    CUTIN = 2
    BRAKE = 3
    END = 4

  def __init__(self):
    self.current_maneuver = self.CutInManeuver.EVADE
    self.state = None
    self.end_frames = 0
    self.prev_action = Action.IDLE.value

  def reset(self, obs : np.ndarray, info : dict):
    self.current_maneuver = self.CutInManeuver.EVADE
    self.state = None
    self.end_frames = 0
    self.prev_action = Action.IDLE.value

    ego_x = obs[0][0,0,1]
    ego_y = obs[0][0,0,2]
    npc_x = obs[1][0,0,1]
    npc_y = obs[1][0,0,2]

    self.set_state(ego_x, ego_y, npc_x, npc_y)

  def set_state(self, ego_x : float, ego_y: float, npc_x : float, npc_y : float):
    self.ego_x = ego_x
    self.ego_y = ego_y
    self.npc_x = npc_x
    self.npc_y = npc_y

  def get_action(self):
    action = self.prev_action
    if self.current_maneuver == self.CutInManeuver.EVADE:
      if abs(self.ego_y - self.npc_y) < 0.2:
        if abs(self.npc_y) < 0.2:
          action = Action.LANE_RIGHT.value
        elif abs(4-self.npc_y) < 0.2:
          action = Action.LANE_LEFT.value
        self.current_maneuver = self.CutInManeuver.ACCELERATE
      else:
        self.current_maneuver = self.CutInManeuver.ACCELERATE
    elif self.current_maneuver == self.CutInManeuver.ACCELERATE:
      action = Action.FASTER.value
      if abs(self.ego_y - self.npc_y) < 0.2:
        if abs(self.npc_y) < 0.2:
          action = Action.LANE_RIGHT.value
        elif abs(4-self.npc_y) < 0.2:
          action = Action.LANE_LEFT.value
      elif self.ego_x - self.npc_x <= 0.0:
        self.current_maneuver = self.CutInManeuver.CUTIN
        if abs(self.npc_y) < 0.2:
          action = Action.LANE_RIGHT.value
        elif abs(4-self.npc_y) < 0.2:
          action = Action.LANE_LEFT.value
      
    elif self.current_maneuver == self.CutInManeuver.CUTIN:
      self.current_maneuver = self.CutInManeuver.END
    elif self.current_maneuver == self.CutInManeuver.END:
      self.end_frames += 1
      action = Action.IDLE.value
    self.prev_action = action
    return action