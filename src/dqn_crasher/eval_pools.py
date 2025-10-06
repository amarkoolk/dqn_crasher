import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

import gymnasium as gym
import highway_env
import torch
from dqn_crasher.utils.config import load_pkg_yaml
from dqn_crasher.utils.helpers import make_players
from dqn_crasher.utils.wandb_logging import initialize_logging

import dqn_crasher.scenarios.policies as policies
import dqn_crasher.scenarios.scenarios as scenarios
import wandb
from dqn_crasher.agents.dqn_agent import DQN_Agent
from dqn_crasher.training.runner import MultiAgentRunner
from dqn_crasher.utils.utils import DeviceHelper

def build_model_pools(config, device):

    model_directory = config.get("model_directory")
    cycles = config.get("cycles")

    ego_policies = []
    npc_policies = []
    ego_models = config.get("models_EGO")
    npc_models = config.get("models_NPC")
    for cycle in range(cycles+1):
        if cycle == 0:
            ego_model = os.path.join(model_directory, f"EGO_{cycle}", "model_1000000.pth")
            ego_agent, ego_policy = build_new_dqn_policy(config, device, cycle, ego_model, "EGO", False)
            ego_policies.append(ego_policy)

            continue

        ego_model = os.path.join(model_directory, f"EGO_{cycle}", f"model_{ego_models[cycle-1]}.pth")
        ego_agent, ego_policy = build_new_dqn_policy(config, device, cycle, ego_model, "EGO", False)
        ego_policies.append(ego_policy)


        npc_model = os.path.join(model_directory, f"NPC_{cycle}", f"model_{npc_models[cycle-1]}.pth")
        npc_agent, npc_policy = build_new_dqn_policy(config, device, cycle, npc_model, "NPC", False)
        npc_policies.append(npc_policy)

    ego_pool = policies.PolicyDistribution(ego_policies, 0)
    npc_pool = policies.PolicyDistribution(npc_policies, 0)

    return  ego_pool, npc_pool

def build_new_dqn_policy(config, device, cycle, init_model, model_str, train : bool = False):
    trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
        f"{model_str}_{cycle}"
    )

    if model_str == "EGO":
        config["gym_config"] = load_pkg_yaml("configs/env/multi_agent.yaml")
    elif model_str == "NPC":
        config["gym_config"] = load_pkg_yaml("configs/env/multi_agent_adversarial.yaml")

    tmp = gym.make(config["env_name"], config=config["gym_config"])
    act_space = tmp.action_space[0]
    n_act = act_space.n
    n_obs = 10 * config.get("frame_stack", 1)
    tmp.close()

    dqn_agent = DQN_Agent(n_obs, n_act, act_space, config, device)
    dqn_policy = policies.DQNPolicy(dqn_agent, trajectory_save_path, train = train, init_model = init_model)

    return dqn_agent, dqn_policy

def run_pool_evaluation(config, device, cycle, policy, pool, model_str):

    if model_str == "EGO":
        gym_config = load_pkg_yaml("configs/env/multi_agent.yaml")
    elif model_str == "NPC":
        gym_config = load_pkg_yaml("configs/env/multi_agent_adversarial.yaml")

    gym_config["observation"]["observation_config"]["frame_stack"] = config.get("frame_stack", 1)
    config["gym_config"] = gym_config
    run = initialize_logging(config, False, eval=True, run_name = f"{model_str}{cycle}_eval_{config.get("model_pool")}_best")
    runner = MultiAgentRunner(config, device, policy, pool)
    runner.test()
        


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("env_config")
    args = parser.parse_args()

    config = load_pkg_yaml(f"configs/model/{args.env_config}.yaml")
    device = DeviceHelper.get(config)

    mp_ego, mp_npc = build_model_pools(config, device)

    for cycle in range(config.get("cycles")+1):

        # Eval E vs NPC Pool
        run_pool_evaluation(config, device, cycle, mp_ego.policies[cycle], mp_npc, "EGO")
        
    


if __name__ == "__main__":
    main()
