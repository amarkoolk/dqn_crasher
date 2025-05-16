import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from policies    import BasePolicy, DQNPolicy, ScenarioPolicy, MobilPolicy, PolicyDistribution
from runner      import MultiAgentRunner
from dqn_agent   import DQN_Agent
from config      import load_config

import scenarios, helpers, utils
import gymnasium as gym
import highway_env


class DeviceHelper:
    @staticmethod
    def get(config):
        if config["device"] == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config["device"] == "mps":
            return torch.device("mps"  if torch.backends.mps.is_available() else "cpu")
        return torch.device("cpu")

def pick_policy_function(class_str : str, config, device):
    strings = class_str.split(".")
    if "scenario" in class_str:
        class_type = utils.class_from_path(class_str)
        return set_scenario_policy(class_type, config)
    elif "Mobil" in class_str and len(strings) > 2:
        class_type = utils.class_from_path(strings[0] + '.' + strings[1])
        config['spawn_config'] = [strings[2]]
        return set_policy(class_type, config, device)
    else:
        class_type = utils.class_from_path(class_str)
        return set_policy(class_type, config, device)


def set_policy(class_type, config, device):
    trajectory_save_path = os.path.join(config.get('root_directory', './'), config.get('trajectory_path', 'trajectories'))

    if class_type == DQNPolicy:
        gym_config = config['gym_config']
        tmp = gym.make(config['env_name'], config=gym_config)
        act_space = tmp.action_space[0]
        n_act = act_space.n
        n_obs = 10 * config.get('frame_stack', 1)
        tmp.close()

        agent = DQN_Agent(n_obs, n_act, act_space, config, device)
        if config.get('train_ego', False) is False:
            agent.load_model(config.get('ego_model',''))
        dqn_policy = class_type(agent, trajectory_save_path, config.get('train_ego', False))
        return dqn_policy
    elif class_type == MobilPolicy:
        mobil_policy = MobilPolicy(trajectory_save_path, config['spawn_config'])
        return mobil_policy

def set_scenario_policy(scenario_type, config):
    n_obs = 10 * config.get('frame_stack', 1)
    if scenario_type == scenarios.IdleFaster:
        scen = ScenarioPolicy(scenarios.IdleFaster, n_obs, config)
    elif scenario_type == scenarios.IdleSlower:
        scen = ScenarioPolicy(scenarios.IdleSlower, n_obs, config)
    elif scenario_type == scenarios.CutIn:
        scen = ScenarioPolicy(scenarios.CutIn, n_obs, config)
    elif scenario_type == scenarios.CutInSlowDown:
        scen = ScenarioPolicy(scenarios.CutInSlowDown, n_obs, config)
    else:
        raise Exception('No Viable Scenario Chosen')

    return scen


def make_players(config, gym_config, device):
    # build your DQN agents
    tmp = gym.make(config['env_name'], config=gym_config)
    act_space = tmp.action_space[0]
    n_act = act_space.n
    n_obs = 10 * config.get('frame_stack', 1)
    tmp.close()

    # Policy A
    p_A_list = config.get("policy_A")
    class_list_A = []
    if len(p_A_list) == 1:
        p_A = pick_policy_function(p_A_list[0], config, device)
    elif len(p_A_list) > 1:
        for class_str in p_A_list:
            class_list_A.append(pick_policy_function(class_str, config, device))
        p_A = PolicyDistribution(class_list_A)
    else:
        raise Exception("Empty Policy List")

    # Policy B
    p_B_list = config.get("policy_B")
    class_list_B = []
    if len(p_B_list) == 1:
        p_B = pick_policy_function(p_B_list[0], config, device)
    elif len(p_B_list) > 1:
        for class_str in p_B_list:
            class_list_B.append(pick_policy_function(class_str, config, device))
        p_B = PolicyDistribution(class_list_B)
    else:
        raise Exception("Empty Policy List")

    return p_A, p_B

if __name__=="__main__":

    config = load_config("model_configs/dqn_vs_scenarios.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config['adversarial'] = False
    gym_config['normalize_reward'] = True
    gym_config['collision_reward'] = -100
    gym_config['observation']['observation_config']['frame_stack'] = config['frame_stack']
    config['gym_config'] = gym_config

    device = DeviceHelper.get(config)
    p_A, p_B = make_players(config, gym_config, device)

    runner = MultiAgentRunner(config["env_name"], config, gym_config, device, p_A, p_B)

    if config.get('train_ego', False):
        runner.train(train_player="A")
    else:
        runner.test()
