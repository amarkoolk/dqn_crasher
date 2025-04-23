import torch
import gymnasium as gym

from dqn_agent import DQN_Agent
from multi_agent_dqn import train_agents

from scenarios import Slowdown, SlowdownSameLane, SpeedUp, CutIn
from config import load_config
import helpers
import wandb
from wandb_logging import initialize_logging, log_stats
from tqdm import tqdm
import random
import time

import highway_env

class DeviceHelper:
    @staticmethod
    def get(config):
        if config["device"] == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config["device"] == "mps":
            return torch.device("mps"  if torch.backends.mps.is_available() else "cpu")
        return torch.device("cpu")

def train_agent(config: dict):

    device = DeviceHelper.get(config)

    gym_config = config.get("gym_config", {})
    gym_config['observation']['observation_config']['frame_stack'] = config.get('frame_stack', 1)
    env = gym.make(config['env_name'], config = gym_config, render_mode = 'rgb_array')

    action_space = env.action_space[0]
    n_actions = action_space.n
    n_obs = 10 * gym_config['observation']['observation_config']['frame_stack']

    train_ego = config.get('train_ego', True)
    trajectory_path = config.get('trajectory_path', None)

    ego_agent = DQN_Agent(n_obs, n_actions, action_space, config, device)
    npc_agent = DQN_Agent(n_obs, n_actions, action_space, config, device)

    ego_agent.load_model(path = config['ego_model'])
    npc_agent.load_model(path = config['npc_model'])

    train_agents(env, ego_agent, npc_agent, config, device, trajectory_path, train_ego)
    env.close()

    return ego_agent, npc_agent

    
def test_scenarios(
    config: dict
):
    """    
    Args:
        env: The environment.
        ego_agent (DQN_Agent): The ego agent (multi-agent).
        config: Training/testing arguments (must contain total_timesteps, track, etc.).
        device: Torch device.
        use_pbar: Whether to display a progress bar.
    """

    device = DeviceHelper.get(config)

    total_episodes = config.get('total_episodes', 5)

    scenario_types = config.get('scenarios', [])
    scenarios = []
    if len(scenario_types) == 0:
        raise ValueError("No scenarios provided for testing.")
    else:
        for scenario in scenario_types:
            scenarios.append(helpers.class_from_path(scenario))

    gym_config = config.get('gym_config', None)
    gym_config['observation']['observation_config']['frame_stack'] = config.get('frame_stack', 1)
    if gym_config is None:
        raise ValueError("No gym_config provided for testing.")
    
    env = gym.make('crash-v0', config=gym_config, render_mode = 'rgb_array')

    action_space = env.action_space[0]
    n_actions = action_space.n
    n_obs = 10 * config.get('frame_stack', 1)
    vs_mobil = gym_config.get('vs_mobil', False)

    ego_agent = DQN_Agent(n_obs, n_actions, action_space, config, device)
    ego_agent.load_model(config['ego_model'])
    

    for scenario in scenarios:
        # Initialize the scenario
        scenario = scenario()
        scenario.set_config(gym_config)
        env = gym.make(config['env_name'], config=gym_config, render_mode = 'rgb_array')

        # Logging
        if config.get('track', False):
            if wandb.run is not None:
                wandb.finish()
            run = initialize_logging(
                config,
                train_ego=None,
                npc_pool_size=None,
                ego_pool_size=None
            )

        obs, info = env.reset()
        ego_state, npc_state = helpers.unpack_states(obs, ego_agent, device, vs_mobil)


        # Main loop
        for ep_num in tqdm(range(total_episodes)):

            done = truncated = False
            episode_statistics = helpers.initialize_stats()

            obs, info = env.reset()
            ego_state, npc_state = helpers.unpack_states(obs, ego_agent, device, vs_mobil)
            scenario.reset(ego_state, npc_state, info)

            while not (done or truncated):

                action = scenario.get_action()
                npc_action = torch.squeeze(torch.tensor([action])).view(1, 1).cpu().numpy()
                if vs_mobil is False:
                    ego_action = torch.squeeze(ego_agent.predict(ego_state)).view(1, 1).cpu().numpy()
                else:
                    ego_action = npc_action
                actions = helpers.make_step_actions(ego_action, npc_action, vs_mobil)

                obs, reward, terminated, truncated, info = env.step(
                    actions
                )


                done = terminated or truncated

                ego_state, npc_state = helpers.unpack_states(obs, ego_agent, device, vs_mobil)

                scenario.set_state(
                    ego_state[0, 1], ego_state[0, 2],
                    npc_state[0, 1], npc_state[0, 2]
                )

                # Populate stats
                helpers.populate_stats(
                    info,
                    ego_agent,
                    ego_state,
                    npc_state,
                    reward,
                    episode_statistics
                )

                if done:
                    # Logging
                    if config.get('track', False):
                        log_stats(info, episode_statistics, ego=True)

                    helpers.reset_stats(episode_statistics)

wandb.finish()

def train_scenarios(
    config: dict,
    use_pbar: bool = True
):
    """    
    Args:
        env: The environment.
        ego_agent (DQN_Agent): The ego agent (multi-agent).
        config: Training/testing arguments (must contain total_timesteps, track, etc.).
        device: Torch device.
        use_pbar: Whether to display a progress bar.
    """

    device = DeviceHelper.get(config)

    total_timesteps = config.get('total_timesteps', 100000)
    if use_pbar:
        pbar = tqdm(total = total_timesteps)
    else:
        pbar = None

    scenario_types = config.get('scenarios', [])
    scenarios = []
    if len(scenario_types) == 0:
        raise ValueError("No scenarios provided for testing.")
    else:
        for scenario in scenario_types:
            scenarios.append(helpers.class_from_path(scenario))

    gym_config = config.get('gym_config', None)
    gym_config['observation']['observation_config']['frame_stack'] = config.get('frame_stack', 1)
    gym_config['adversarial'] = False
    gym_config['normalize_reward'] = True
    gym_config['collision_reward'] = -100
    if gym_config is None:
        raise ValueError("No gym_config provided for testing.")
    
    env = gym.make('crash-v0', config=gym_config, render_mode = 'rgb_array')

    action_space = env.action_space[0]
    n_actions = action_space.n
    n_obs = 10 * config.get('frame_stack', 1)

    ego_agent = DQN_Agent(n_obs, n_actions, action_space, config, device)
    ego_agent.load_model(config['ego_model'])

    t_step = 0
    episode_statistics = helpers.initialize_stats()

    # Logging
    if config.get('track', False):
        if wandb.run is not None:
            wandb.finish()
        run = initialize_logging(
            config,
            train_ego=None,
            npc_pool_size=None,
            ego_pool_size=None
        )

    # Initialize the scenario
    scenario = random.choice(scenarios)
    scenario = scenario()
    scenario.set_config(gym_config)
    env.unwrapped.configure(gym_config)
    obs, info = env.reset()
    ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, ego_agent, device)

    scenario.set_state(
        ego_state[0, 1], ego_state[0, 2],
        npc_state[0, 1], npc_state[0, 2]
    )
    

    while t_step < total_timesteps:

        action = scenario.get_action()
        npc_action = torch.squeeze(torch.tensor([action])).view(1, 1)
        ego_action = ego_agent.select_action(ego_state, t_step)

        obs, reward, terminated, truncated, info = env.step(
            (ego_action.cpu().numpy(), npc_action.cpu().numpy())
        )

        env.render()

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated or truncated
        int_frames = info['int_frames']

        next_state, npc_state = helpers.obs_to_state(
            obs, ego_agent, ego_agent, device
        )

        ego_state = ego_agent.update(ego_state, ego_action, next_state, reward, terminated)

        scenario.set_state(
            ego_state[0, 1], ego_state[0, 2],
            npc_state[0, 1], npc_state[0, 2]
        )

        # Populate stats
        helpers.populate_stats(
            info,
            ego_agent,
            ego_state,
            npc_state,
            reward.cpu().numpy(),
            episode_statistics
        )

        if done:
            # Logging
            if config.get('track', False):
                log_stats(info, episode_statistics, ego=True)

            helpers.reset_stats(episode_statistics)

            scenario = random.choice(scenarios)
            scenario = scenario()
            scenario.set_config(gym_config)
            env.unwrapped.configure(gym_config)
            obs, info = env.reset()
            ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, ego_agent, device)

            scenario.set_state(
                ego_state[0, 1], ego_state[0, 2],
                npc_state[0, 1], npc_state[0, 2]
            )

        t_step += 1
        if pbar is not None:
            pbar.update(1)

wandb.finish()

def train_vs_mobil(
    config: dict,
    use_pbar: bool = True
):
    """    
    Args:
        env: The environment.
        ego_agent (DQN_Agent): The ego agent (multi-agent).
        config: Training/testing arguments (must contain total_timesteps, track, etc.).
        device: Torch device.
        use_pbar: Whether to display a progress bar.
    """
    device = DeviceHelper.get(config)

    total_timesteps = config.get('total_timesteps', 100000)
    if use_pbar:
        pbar = tqdm(total = total_timesteps)
    else:
        pbar = None

    gym_config = config.get('gym_config', None)
    gym_config['observation']['observation_config']['frame_stack'] = config.get('frame_stack', 1)
    gym_config['adversarial'] = False
    gym_config['normalize_reward'] = True
    gym_config['collision_reward'] = -100
    gym_config['use_mobil'] = True
    gym_config['ego_vs_mobil'] = True
    if gym_config is None:
        raise ValueError("No gym_config provided for testing.")
    
    env = gym.make('crash-v0', config=gym_config, render_mode = 'rgb_array')

    action_space = env.action_space[0]
    n_actions = action_space.n
    n_obs = 10 * config.get('frame_stack', 1)

    ego_agent = DQN_Agent(n_obs, n_actions, action_space, config, device)
    ego_agent.load_model(config['ego_model'])

    t_step = 0
    episode_statistics = helpers.initialize_stats()

    # Logging
    if config.get('track', False):
        if wandb.run is not None:
            wandb.finish()
        run = initialize_logging(
            config,
            train_ego=None,
            npc_pool_size=None,
            ego_pool_size=None
        )

    # Initialize the scenario
    obs, info = env.reset()
    ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, ego_agent, device)

    while t_step < total_timesteps:

        npc_action = torch.squeeze(torch.tensor([0])).view(1, 1)
        ego_action = ego_agent.select_action(ego_state, t_step)

        obs, reward, terminated, truncated, info = env.step(
            (ego_action.cpu().numpy(), npc_action.cpu().numpy())
        )

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated or truncated
        int_frames = info['int_frames']

        next_state, npc_state = helpers.obs_to_state(
            obs, ego_agent, ego_agent, device
        )

        ego_state = ego_agent.update(ego_state, ego_action, next_state, reward, terminated)


        # Populate stats
        helpers.populate_stats(
            info,
            ego_agent,
            ego_state,
            npc_state,
            reward.cpu().numpy(),
            episode_statistics
        )

        if done:
            # Logging
            if config.get('track', False):
                log_stats(info, episode_statistics, ego=True)

            helpers.reset_stats(episode_statistics)

            obs, info = env.reset()
            ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, ego_agent, device)

        t_step += 1
        if pbar is not None:
            pbar.update(1)

wandb.finish()