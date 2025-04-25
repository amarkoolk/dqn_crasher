import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
from policies    import DQNPolicy, ScenarioPolicy, MobilPolicy, PolicyDistribution
from runner      import MultiAgentRunner
from dqn_agent   import DQN_Agent
from config      import load_config
import scenarios, helpers
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

def make_players(cfg, gym_config, device):
    # build your DQN agents
    tmp = gym.make(cfg['env_name'], config=gym_config)
    act_space = tmp.action_space[0]
    n_act = act_space.n
    n_obs = 10 * config.get('frame_stack', 1)
    tmp.close()
    ego_agent = DQN_Agent(n_obs, n_act, act_space, cfg, device)
    ego_agent.load_model(cfg.get("ego_model", None))
    npc_agent = DQN_Agent(n_obs, n_act, act_space, cfg, device)
    npc_agent.load_model(cfg.get("npc_model", None))

    # wrap them
    p_ego = DQNPolicy(ego_agent, cfg.get('trajectory_path', 'trajectories'), cfg.get('train_ego', False))
    p_npc = DQNPolicy(npc_agent, cfg.get('trajectory_path', 'trajectories'), not cfg.get('train_npc', False))

    # scenario wrapper
    cutin_scen = ScenarioPolicy(scenarios.CutIn, n_obs, cfg.get('trajectory_path', 'trajectories'))
    slowdown_scen = ScenarioPolicy(scenarios.IdleSlower, n_obs, cfg.get('trajectory_path', 'trajectories'))
    p_scen = PolicyDistribution([cutin_scen, slowdown_scen])

    p_mobil = MobilPolicy(cfg.get('trajectory_path', 'trajectories'))

    return p_ego, p_npc, p_scen, p_mobil

if __name__=="__main__":

    config = load_config("model_configs/test_scenario_config.yaml")
    gym_config = load_config("env_configs/multi_agent.yaml")
    gym_config['vs_mobil'] = True
    gym_config['use_mobil'] = True
    gym_config['controlled_vehicles'] = 1
    gym_config['other_vehicles'] = 1
    gym_config['adversarial'] = False
    gym_config['normalize_reward'] = True
    gym_config['collision_reward'] = -100
    config['gym_config'] = gym_config

    device = DeviceHelper.get(config)
    p_ego, p_npc, p_scen, p_mobil = make_players(config, gym_config, device)
    A, B, train_pl = p_scen, p_mobil, None
    # A, B, train_pl = p_ego,  p_scen, None
    # A, B, train_pl = p_ego,  p_npc, "A"

    runner = MultiAgentRunner(config["env_name"], config, gym_config, device, A, B)

    # runner.train(train_player=train_pl or "A")
    runner.test()
