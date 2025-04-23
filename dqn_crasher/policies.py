import torch
import random
import helpers
from abc import ABC, abstractmethod
import os
from dqn_agent import DQN_Agent
from buffers import Transition
from trajectory_store import TrajectoryStore

class BasePolicy(ABC):
    @abstractmethod
    def reset(self, ego_state, npc_state, info=None, train_mode: bool = True):
        """Called once at the start of every episode."""
        pass

    @abstractmethod
    def select_action(self, own_state, other_state, t_step: int = None):
        """Only called in train-mode for whichever player is learning."""
        pass

    @abstractmethod
    def predict(self, own_state, other_state):
        """Called in test-mode, or for the fixed agent in train-mode."""
        pass

    @abstractmethod
    def update(self, transition: Transition, done):
        """Called only for the training player in train-mode."""
        pass


class DQNPolicy(BasePolicy):
    def __init__(self, agent, trajectory_store_dir):
        self.agent : DQN_Agent = agent
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir)

    def reset(self, ego_state, npc_state, info=None, train_mode=True):
        # nothing to do
        pass

    def select_action(self, own_state, other_state, t_step=None):
        return self.agent.select_action(own_state, t_step)

    def predict(self, own_state, other_state):
        return self.agent.predict(own_state)

    def update(self, transition: Transition, done):
        return self.agent.update(transition.state, transition.action, transition.next_state, transition.reward, done)
    
    def save_model(self, file_path='model.pth'):
        dirpath = os.path.dirname(file_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.agent.save_model(file_path)


class ScenarioPolicy(BasePolicy):
    def __init__(self, scenario_classes, gym_cfg, len_obs, trajectory_store_dir):
        self.scenario_classes = scenario_classes
        self.gym_cfg           = gym_cfg
        self.scenario          = None
        self.len_obs           = len_obs
        self.agent             = None
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir)


    def reset(self, ego_state, npc_state, info=None, train_mode=True):
        # sample a fresh scenario each episode
        Cls = random.choice(self.scenario_classes)
        self.scenario = Cls()
        self.scenario.set_config(self.gym_cfg)
        if train_mode:
            self.scenario.set_state(
                ego_state[0,1], ego_state[0,2],
                npc_state[0,1], npc_state[0,2]
            )
        else:
            self.scenario.reset(ego_state, npc_state, info)

    def select_action(self, own_state, other_state, t_step=None):
        action = self.scenario.get_action()
        return torch.squeeze(torch.tensor([action])).view(1, 1)

    def predict(self, own_state, other_state):
        action = self.scenario.get_action()
        return torch.squeeze(torch.tensor([action])).view(1, 1)

    def update(self, transition: Transition, done):

        next_state = transition.next_state

        self.scenario.set_state(
            next_state[0,1], next_state[0,2],
            next_state[0, 6], next_state[0, 7]
        )

class MobilPolicy(BasePolicy):

    def __init__(self, trajectory_store_dir):
        self.agent = None
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir)

    def reset(self, ego_state, npc_state, info=None, train_mode=True):
        pass

    def select_action(self, own_state, other_state, t_step=None):
        return torch.squeeze(torch.tensor([0])).view(1, 1)

    def predict(self, own_state, other_state):
        return torch.squeeze(torch.tensor([0])).view(1, 1)

    def update(self, transition: Transition, done):
        pass