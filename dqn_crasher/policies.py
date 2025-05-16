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
        class_name = type(self).__module__ + "." + type(self).__name__
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir + f'_{class_name}.jsonl')
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
    def __init__(self, scenario_class, len_obs, config):
        self.scenario          = scenario_class(use_spawn_distribution = config.get('train_ego',False))
        self.len_obs           = len_obs
        self.agent             = None
        class_name = type(self.scenario).__module__ + "." + type(self.scenario).__name__
        trajectory_save_path = os.path.join(config.get('root_directory', './'), config.get('trajectory_path', './trajectories')+ f'_{class_name}.jsonl')
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_save_path)


    def reset(self):
        self.scenario.reset()

    def set_config(self, config: dict):
        self.scenario.set_config(config['gym_config'])

    def set_state(self, ego_state, npc_state):
        self.scenario.set_state(ego_state, npc_state)

    def select_action(self, own_state, other_state, t_step=None):
        action = self.scenario.get_action()
        return torch.squeeze(torch.tensor([action])).view(1, 1)

    def update(self, transition: Transition, done):
        next_state = transition.next_state
        return next_state

class MobilPolicy(BasePolicy):

    def __init__(self, trajectory_store_dir, spawn_configs):
        self.agent = None
        self.spawn_configs = spawn_configs
        class_name = type(self).__module__ + "." + type(self).__name__
        spawn_name = self.spawn_configs[0] if len(self.spawn_configs) == 1 else 'scenario'
        self.store : TrajectoryStore = TrajectoryStore(file_path = trajectory_store_dir + f'_{class_name}.{spawn_name}.jsonl')
        print(f"Mobil Policy: {self.spawn_configs}")

    def reset(self):
        pass

    def set_state(self, ego_state, npc_state):
        pass

    def set_config(self, config: dict):
        config['gym_config']['spawn_configs'] = self.spawn_configs
        config['gym_config']['vs_mobil'] = True
        config['gym_config']['use_mobil'] = True
        config['gym_config']['controlled_vehicles'] = 1
        config['gym_config']['other_vehicles'] = 1

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

    def reset(self, policy_num = None):
        if policy_num == None:
            self.current_policy = random.choice(self.policies)
        else:
            assert isinstance(policy_num, int)
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
