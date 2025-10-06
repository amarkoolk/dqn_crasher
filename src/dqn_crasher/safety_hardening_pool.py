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

def build_model_pool(config, device, init_model, model_str):

    if model_str == "EGO":
        adversarial = False
    else:
        adversarial = True

    if init_model:
        agent, policy = build_new_dqn_policy(config, device, 0, init_model, model_str)
        policy_distribution = policies.ModelPoolPolicy(policies = [policy], adversarial = adversarial, sampling = config.get("model_pool", "uniform"))
    else:
        policy_distribution = policies.ModelPoolPolicy(policies = [], adversarial = adversarial, sampling = config.get("model_pool", "uniform"))

    return  policy_distribution

def build_new_dqn_policy(config, device, cycle, init_model, model_str, train : bool = False):
    trajectory_save_path = os.path.join(
        config.get("root_directory", "./"),
        config.get("trajectory_path", "trajectories"),
        f"{model_str}_{cycle}"
    )

    if model_str == "EGO":
        config["gym_config"] = load_pkg_yaml("configs/env/multi_agent_old.yaml")
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

def add_new_model_to_pool(config, device, cycle, pool : policies.ModelPoolPolicy, init_model : str, model_str):
    
    dqn_agent, dqn_policy = build_new_dqn_policy(config, device, cycle, init_model, model_str)
    pool.add_model(dqn_agent, dqn_policy)


def run_pool_training(config, device, cycle, model_dir, init_model, pool, model_str):

    dqn_agent, dqn_policy = build_new_dqn_policy(config, device, cycle, init_model, model_str, train = True)

    if model_str == "EGO":  
        gym_config = load_pkg_yaml("configs/env/multi_agent_old.yaml")
        train_ego = True
    elif model_str == "NPC":
        gym_config = load_pkg_yaml("configs/env/multi_agent_adversarial.yaml")
        train_ego = False

    gym_config["observation"]["observation_config"]["frame_stack"] = config.get("frame_stack", 1)
    config["gym_config"] = gym_config

    run_name = f"{model_str}{cycle}_" + config["run_name"]
    config["model_save_path"] = model_dir + f"/{model_str}_{cycle}"

    run = initialize_logging(config, train_ego, run_name = run_name)
    runner = MultiAgentRunner(config, device, dqn_policy, pool)
    runner.train(train_player="A")

    # Freeze Model and return it
    dqn_policy.set_train(False)

    return dqn_policy

def run_pool_evaluation(config, device, cycle, policy, pool, model_str, update_elo):

    gym_config = load_pkg_yaml("configs/env/multi_agent.yaml")
    gym_config["observation"]["observation_config"]["frame_stack"] = config.get("frame_stack", 1)
    config["gym_config"] = gym_config
    run = initialize_logging(config, False, eval=True, run_name = f"{model_str}{cycle}_eval_{config.get("model_pool")}")

    if config.get("model_pool") == "prioritized":
        pool.init_eval(eval_episodes = config.get("tournament_episodes") * len(pool.policies), latest_model = False)
    runner = MultiAgentRunner(config, device, policy, pool, update_elo)
    runner.test()
        


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("env_config")
    parser.add_argument("cycles")
    args = parser.parse_args()

    config = load_pkg_yaml(f"configs/model/{args.env_config}.yaml")
    device = DeviceHelper.get(config)

    ego_model = config.get("initial_model_A")
    npc_model = ""


    mp_ego : policies.ModelPoolPolicy = build_model_pool(config, device, ego_model, "EGO")
    mp_npc : policies.ModelPoolPolicy = build_model_pool(config, device, None, "NPC")

    model_dir = os.path.join(
                    config.get("root_directory", ""),
                    config.get("model_save_path", "models/model")
                )
    
    ego_elo = 1000
    npc_elo = 1000

    for cycle in range(1, int(args.cycles) + 1):

        # Train NPC
        npc_policy = run_pool_training(config, device, cycle, model_dir, npc_model, mp_ego, "NPC")

        if config.get("model_pool") == "prioritized":
            run_pool_evaluation(config, device, cycle, npc_policy, mp_ego, "NPC", True)
            mp_ego.model_pool.eval = False
            npc_elo = mp_ego.model_pool.opponent_elo

            print(f"NPC{cycle}: {npc_elo}")


        # Add NPC to Pool
        mp_npc.add_model(npc_policy, npc_elo)
        mp_npc.model_pool.update_probabilities()

        npc_model = os.path.join(
                    model_dir,
                    f"NPC_{cycle}/model_{config.get("total_timesteps")}.pth"
                )
        
        # Train Ego
        ego_policy = run_pool_training(config, device, cycle, model_dir, ego_model, mp_npc, "EGO")

        if config.get("model_pool") == "prioritized":
            run_pool_evaluation(config, device, cycle, ego_policy, mp_npc, "EGO", True)
            mp_npc.model_pool.eval = False
            ego_elo = mp_npc.model_pool.opponent_elo

            print(f"EGO{cycle}: {ego_elo}")

        # Add Ego to Pool
        mp_ego.add_model(ego_policy, ego_elo)
        mp_ego.model_pool.update_probabilities()

        ego_model = os.path.join(
                    model_dir,
                    f"EGO_{cycle}/model_{config.get("total_timesteps")}.pth"
                )        
        
    mp_ego.model_pool.write_model_pool_log(os.path.join(config.get("root_directory"), "ego_model_pool.json"))
    mp_npc.model_pool.write_model_pool_log(os.path.join(config.get("root_directory"), "npc_model_pool.json"))
    


if __name__ == "__main__":
    main()
