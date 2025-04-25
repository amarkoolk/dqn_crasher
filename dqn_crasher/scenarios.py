from enum import Enum
import numpy as np

class Action(Enum):
    LANE_LEFT = 0
    IDLE = 1
    LANE_RIGHT = 2
    FASTER = 3
    SLOWER = 4

class Scenario:
    def __init__(self):
        self.state = None
        self.end_frames = 0
        self.prev_action = Action.IDLE.value
        self.spawn_configs = []

    def set_state(self, ego_state: np.ndarray, npc_state: np.ndarray):

        self.ego_x = ego_state[0, 1]
        self.ego_y = ego_state[0, 2]
        self.npc_x = npc_state[0, 1]
        self.npc_y = npc_state[0, 2]

    def reset(self):
        """
        Common reset logic across all scenarios.
        """
        self.state = None
        self.end_frames = 0
        self.prev_action = Action.IDLE.value

    def get_action(self):
        """
        Should be overridden by each scenario with scenario-specific logic.
        """
        raise NotImplementedError

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        raise NotImplementedError

#
# Child classes
#

class IdleSlower(Scenario):
    def __init__(self):
        super().__init__()

    def get_action(self):
        return Action.IDLE.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = -5.0
        config['spawn_configs'] = ['forward_left']

class IdleFaster(Scenario):
    def __init__(self):
        super().__init__()

    def get_action(self):
        return Action.IDLE.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = 5.0
        config['spawn_configs'] = ['behind_right']

class Slowdown(Scenario):
    def __init__(self):
        super().__init__()

    def get_action(self):
        return Action.SLOWER.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = 0.0
        config['spawn_configs'] = ['forward_left']


class SlowdownSameLane(Scenario):
    def __init__(self):
        super().__init__()

    def get_action(self):
        return Action.SLOWER.value

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = 0.0
        config['spawn_configs'] = ['forward_center']


class SpeedUp(Scenario):
    def __init__(self):
        super().__init__()

    def get_action(self):
        return Action.FASTER.value


    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = 0.0
        config['spawn_configs'] = ['behind_left']


class CutIn(Scenario):
    class CutInManeuver(Enum):
        EVADE = 0
        ACCELERATE = 1
        CUTIN = 2
        BRAKE = 3
        END = 4

    def __init__(self):
        super().__init__()
        self.current_maneuver = self.CutInManeuver.EVADE
        self.spawn_configs = ['behind_left']

    def reset(self):
        """
        Override reset if scenario-specific attributes need re-initialization.
        Still call super() for the common steps.
        """
        super().reset()
        self.current_maneuver = self.CutInManeuver.EVADE

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = 0.0
        config['spawn_configs'] = ['behind_left']

    def get_action(self):
        action = self.prev_action

        if self.current_maneuver == self.CutInManeuver.EVADE:
            if abs(self.ego_y - self.npc_y) < 0.2:
                if abs(self.npc_y) < 0.2:
                    action = Action.LANE_RIGHT.value
                elif abs(4 - self.npc_y) < 0.2:
                    action = Action.LANE_LEFT.value
                self.current_maneuver = self.CutInManeuver.ACCELERATE
            else:
                self.current_maneuver = self.CutInManeuver.ACCELERATE

        elif self.current_maneuver == self.CutInManeuver.ACCELERATE:
            action = Action.FASTER.value
            if abs(self.ego_y - self.npc_y) < 0.2:
                if abs(self.npc_y) < 0.2:
                    action = Action.LANE_RIGHT.value
                elif abs(4 - self.npc_y) < 0.2:
                    action = Action.LANE_LEFT.value
            elif (self.ego_x - self.npc_x) > 5.0:
                self.current_maneuver = self.CutInManeuver.CUTIN
                if abs(self.npc_y) > 0.2:
                    action = Action.LANE_RIGHT.value
                elif abs(4 - self.npc_y) > 0.2:
                    action = Action.LANE_LEFT.value

        elif self.current_maneuver == self.CutInManeuver.CUTIN:
            # Once we do the cut-in, move to END
            self.current_maneuver = self.CutInManeuver.END

        elif self.current_maneuver == self.CutInManeuver.END:
            self.end_frames += 1
            action = Action.IDLE.value


        self.prev_action = action
        return action

class CutInSlowDown(Scenario):
    class CutInManeuver(Enum):
        EVADE = 0
        ACCELERATE = 1
        CUTIN = 2
        BRAKE = 3
        END = 4

    def __init__(self):
        super().__init__()
        self.counter = 0
        self.current_maneuver = self.CutInManeuver.EVADE
        self.prev_action = Action.IDLE.value
        self.spawn_configs = ['forward_left', 'forward_right']

    def reset(self, ego_state: np.ndarray, npc_state: np.ndarray):
        """
        Override reset if scenario-specific attributes need re-initialization.
        Still call super() for the common steps.
        """
        super().reset(ego_state, npc_state)
        self.counter = 0
        self.prev_action = Action.IDLE.value
        self.current_maneuver = self.CutInManeuver.EVADE

    def set_config(self, config: dict):
        """
        Set the configuration for the scenario.
        """
        config['adversarial'] = False
        config['use_spawn_distribution'] = False
        config['mean_delta_v'] = 0.0
        config['mean_distance'] = 20.0
        config['spawn_configs'] = ['forward_left']

    def get_action(self):
        action = self.prev_action

        dy = self.ego_y - self.npc_y

        if self.current_maneuver == self.CutInManeuver.EVADE:
            self.counter +=1
            if self.counter > 2:
                self.current_maneuver = self.CutInManeuver.CUTIN
        elif self.current_maneuver == self.CutInManeuver.CUTIN:
            self.current_maneuver = self.CutInManeuver.BRAKE
            if dy < -0.2:
                action = Action.LANE_RIGHT.value
            elif dy > -0.2:
                action = Action.LANE_LEFT.value
            # Once we do the cut-in, move to BRAKE
            self.current_maneuver = self.CutInManeuver.BRAKE

        elif self.current_maneuver == self.CutInManeuver.BRAKE:
            action = Action.SLOWER.value

        self.prev_action = action
        return action
