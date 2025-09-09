import os
import random
from abc import ABC, abstractmethod

import numpy as np
import torch

from dqn_crasher.buffers.experience_replay import Transition
from dqn_crasher.utils.trajectory_store import TrajectoryStore
from dqn_crasher.agents.dqn_agent import DQN_Agent


class BasePolicy(ABC):
    @abstractmethod
    def reset(self, test=False):
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

    @abstractmethod
    def save_state(self):
        """Save and return the current state of the policy."""
        pass

    @abstractmethod
    def restore_state(self, state):
        """Restore the policy state from a previously saved state."""
        pass


class DQNPolicy(BasePolicy):
    def __init__(self, agent, trajectory_store_dir, train, init_model=None):
        self.agent: DQN_Agent = agent
        class_name = type(self).__module__ + "." + type(self).__name__
        train_file_path = os.path.join(
            trajectory_store_dir, "train", f"{class_name}.jsonl"
        )
        test_file_path = os.path.join(
            trajectory_store_dir, "test", f"{class_name}.jsonl"
        )
        self.store: TrajectoryStore = TrajectoryStore(file_path=train_file_path)
        self.test_store: TrajectoryStore = TrajectoryStore(file_path=test_file_path)
        self.train: bool = train

        if init_model:
            self.load_model(init_model)

    def set_train(self, train):
        self.train = train

    def reset(self, test=False):
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
        return self.agent.update(
            transition.state,
            transition.action,
            transition.next_state,
            transition.reward,
            done,
        )

    def save_model(self, file_path="model.pth"):
        dirpath = os.path.dirname(file_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        self.agent.save_model(file_path)

    def load_model(self, file_path):
        self.agent.load_model(file_path)

    def save_state(self):
        """Save and return the current state of the policy."""
        if hasattr(self.agent, "policy_net") and hasattr(self.agent, "target_net"):
            return {
                "policy_net_state": self.agent.policy_net.state_dict()
                if self.agent.policy_net
                else None,
                "target_net_state": self.agent.target_net.state_dict()
                if self.agent.target_net
                else None,
                "train": self.train,
                "eps_threshold": self.agent.eps_threshold
                if hasattr(self.agent, "eps_threshold")
                else None,
            }
        return {"train": self.train}

    def restore_state(self, state):
        """Restore the policy state from a previously saved state."""
        if (
            "policy_net_state" in state
            and state["policy_net_state"]
            and hasattr(self.agent, "policy_net")
        ):
            self.agent.policy_net.load_state_dict(state["policy_net_state"])
        if (
            "target_net_state" in state
            and state["target_net_state"]
            and hasattr(self.agent, "target_net")
        ):
            self.agent.target_net.load_state_dict(state["target_net_state"])
        if "train" in state:
            self.train = state["train"]
        if (
            "eps_threshold" in state
            and state["eps_threshold"] is not None
            and hasattr(self.agent, "eps_threshold")
        ):
            self.agent.eps_threshold = state["eps_threshold"]


class ScenarioPolicy(BasePolicy):
    def __init__(self, scenario_class, len_obs, config, action_space=5):
        self.scenario = scenario_class(
            use_spawn_distribution=config.get("train_ego", False)
        )
        self.len_obs = len_obs
        self.agent = None
        self.action_space = action_space
        class_name = type(self.scenario).__module__ + "." + type(self.scenario).__name__
        trajectory_save_path = os.path.join(
            config.get("root_directory", "./"),
            config.get("trajectory_path", "./trajectories"),
            "train",
            f"{class_name}.jsonl",
        )
        test_trajectory_save_path = os.path.join(
            config.get("root_directory", "./"),
            config.get("trajectory_path", "./trajectories"),
            "test",
            f"{class_name}.jsonl",
        )
        self.store: TrajectoryStore = TrajectoryStore(file_path=trajectory_save_path)
        self.test_store: TrajectoryStore = TrajectoryStore(
            file_path=test_trajectory_save_path
        )

    def reset(self, test=False):
        self.scenario.reset(test)

    def set_config(self, config: dict):
        self.scenario.set_config(config.get("gym_config", config))

    def set_state(self, ego_state, npc_state):
        self.scenario.set_state(ego_state, npc_state)

    def select_action(self, own_state, other_state, t_step=None):
        action = self.scenario.get_action()
        sampled_action_tensor = torch.tensor(
            np.zeros((1, self.action_space)), dtype=torch.long
        )
        sampled_action_tensor[0, int(action)] = 1.0
        return sampled_action_tensor

    def update(self, transition: Transition, done):
        next_state = transition.next_state
        return next_state

    def save_state(self):
        """Save and return the current state of the policy."""
        # Save scenario state if the scenario has a save_state method
        scenario_state = None
        if hasattr(self.scenario, "save_state"):
            scenario_state = self.scenario.save_state()
        return {"scenario_state": scenario_state, "len_obs": self.len_obs}

    def restore_state(self, state):
        """Restore the policy state from a previously saved state."""
        if (
            "scenario_state" in state
            and state["scenario_state"]
            and hasattr(self.scenario, "restore_state")
        ):
            self.scenario.restore_state(state["scenario_state"])
        if "len_obs" in state:
            self.len_obs = state["len_obs"]


class MobilPolicy(BasePolicy):
    def __init__(self, trajectory_store_dir, spawn_configs, action_space=5):
        self.agent = None
        self.spawn_configs = spawn_configs
        class_name = type(self).__module__ + "." + type(self).__name__
        spawn_name = (
            self.spawn_configs[0] if len(self.spawn_configs) == 1 else "scenario"
        )
        train_file_path = os.path.join(
            trajectory_store_dir, "train", f"{class_name}.{spawn_name}.jsonl"
        )
        test_file_path = os.path.join(
            trajectory_store_dir, "test", f"{class_name}.{spawn_name}.jsonl"
        )
        self.store: TrajectoryStore = TrajectoryStore(file_path=train_file_path)
        self.test_store: TrajectoryStore = TrajectoryStore(file_path=test_file_path)
        self.action_space = action_space
        self.test = False

    def reset(self, test=False):
        self.test = test

    def set_state(self, ego_state, npc_state):
        pass

    def set_config(self, config: dict):
        config["gym_config"]["spawn_configs"] = self.spawn_configs
        config["gym_config"]["vs_mobil"] = True
        config["gym_config"]["use_mobil"] = True
        config["gym_config"]["controlled_vehicles"] = 1
        config["gym_config"]["other_vehicles"] = 1
        if self.test:
            config["gym_config"]["use_spawn_distribution"] = False
        else:
            config["gym_config"]["use_spawn_distribution"] = True

    def select_action(self, own_state, other_state, t_step=None):
        sampled_action_tensor = torch.tensor(
            np.zeros((1, self.action_space)), dtype=torch.long
        )
        return sampled_action_tensor

    def update(self, transition: Transition, done):
        return transition.next_state

    def save_state(self):
        """Save and return the current state of the policy."""
        return {
            "spawn_configs": self.spawn_configs.copy() if self.spawn_configs else None
        }

    def restore_state(self, state):
        """Restore the policy state from a previously saved state."""
        if "spawn_configs" in state and state["spawn_configs"]:
            self.spawn_configs = state["spawn_configs"]


class PolicyDistribution(BasePolicy):
    def __init__(self, policies, seed=0):
        # entries is a list of dicts with keys: from, to, opponents
        self.policies: list = policies
        self.current_policy = self.policies[0]
        self.store = self.current_policy.store
        self.test_store = self.current_policy.test_store
        self.seed = seed
        random.seed(seed)

    def reset(self, policy_num=None, test=False):
        if policy_num == None:
            self.current_policy = random.choice(self.policies)
        else:
            self.current_policy = self.policies[policy_num]

        self.store = self.current_policy.store
        self.test_store = self.current_policy.test_store
        self.current_policy.reset(test)

    def set_state(self, ego_state, npc_state):
        self.current_policy.set_state(ego_state, npc_state)

    def set_config(self, config: dict):
        self.current_policy.set_config(config)

    def select_action(self, own_state, other_state, t_step=None):
        return self.current_policy.select_action(own_state, other_state, t_step)

    def update(self, transition: Transition, done):
        return self.current_policy.update(transition, done)

    def save_state(self):
        """Save and return the current state of the policy distribution."""
        # Save current policy index and individual policy states
        current_policy_idx = self.policies.index(self.current_policy)
        policy_states = [
            policy.save_state() if hasattr(policy, "save_state") else None
            for policy in self.policies
        ]

        return {
            "current_policy_idx": current_policy_idx,
            "policy_states": policy_states,
            "seed": self.seed,
        }

    def restore_state(self, state):
        """Restore the policy distribution state from a previously saved state."""
        if "seed" in state:
            self.seed = state["seed"]
            random.seed(self.seed)

        if "policy_states" in state and state["policy_states"]:
            for i, policy_state in enumerate(state["policy_states"]):
                if (
                    i < len(self.policies)
                    and policy_state
                    and hasattr(self.policies[i], "restore_state")
                ):
                    self.policies[i].restore_state(policy_state)

        if "current_policy_idx" in state and 0 <= state["current_policy_idx"] < len(
            self.policies
        ):
            self.current_policy = self.policies[state["current_policy_idx"]]
            self.store = self.current_policy.store
