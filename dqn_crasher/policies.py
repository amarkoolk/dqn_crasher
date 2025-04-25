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

    def reset(self):
        """Called once at the start of every episode."""
        pass

    def set_state(self, ego_state, npc_state):
        pass

    def set_config(self, config: dict):
        pass

    @abstractmethod
    def select_action(self, own_state, other_state, t_step: int = None):
        """Only called in train-mode for whichever player is learning."""
        pass

    @abstractmethod
    def update(self, transition: Transition, done):
        """Called only for the training player in train-mode."""
        pass


class DQNPolicy(BasePolicy):
    def __init__(self, agent, trajectory_store_dir, train):
        self.agent : DQN_Agent = agent
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir)
        self.train : bool = train

    def reset(self):
        # nothing to do
        pass

    def set_state(self, ego_state, npc_state):
        pass

    def set_config(self, config: dict):
        pass

    def select_action(self, own_state, other_state, t_step=None):
        if self.train:
            return self.agent.select_action(own_state, t_step)
        else:
            return self.agent.predict(own_state)

    def update(self, transition: Transition, done):
        return self.agent.update(transition.state, transition.action, transition.next_state, transition.reward, done)

    def save_model(self, file_path='model.pth'):
        dirpath = os.path.dirname(file_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.agent.save_model(file_path)


class ScenarioPolicy(BasePolicy):
    def __init__(self, scenario_class, len_obs, trajectory_store_dir):
        self.scenario    = scenario_class()
        self.len_obs           = len_obs
        self.agent             = None
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir)


    def reset(self):
        self.scenario.reset()

    def set_config(self, config: dict):
        self.scenario.set_config(config)

    def set_state(self, ego_state, npc_state):
        self.scenario.set_state(ego_state, npc_state)

    def select_action(self, own_state, other_state, t_step=None):
        action = self.scenario.get_action()
        return torch.squeeze(torch.tensor([action])).view(1, 1)

    def update(self, transition: Transition, done):
        next_state = transition.next_state
        return next_state

class MobilPolicy(BasePolicy):

    def __init__(self, trajectory_store_dir):
        self.agent = None
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir)

    def reset(self):
        pass

    def set_state(self, ego_state, npc_state):
        pass

    def set_config(self, config: dict):
        pass

    def select_action(self, own_state, other_state, t_step=None):
        return torch.squeeze(torch.tensor([0])).view(1, 1)

    def update(self, transition: Transition, done):
        return transition.next_state

class PolicyDistribution(BasePolicy):

    def __init__(self, policies, seed = 0):
        # entries is a list of dicts with keys: from, to, opponents
        self.policies : list = policies
        self.current_policy = self.policies[0]
        self.store = self.current_policy.store
        random.seed(seed)

    def reset(self):
        self.current_policy = random.choice(self.policies)
        self.store = self.current_policy.store
        self.current_policy.reset()

    def set_state(self, ego_state, npc_state):
        self.current_policy.set_state(ego_state, npc_state)

    def set_config(self, config: dict):
        self.current_policy.set_config(config)

    def select_action(self, own_state, other_state, t_step=None):
        return self.current_policy.select_action(own_state, other_state, t_step)

    def update(self, transition: Transition, done):
        return self.current_policy.update(transition, done)
