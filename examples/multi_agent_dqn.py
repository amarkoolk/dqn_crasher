import os
from collections import deque

import gymnasium as gym
import helpers
import highway_env
import numpy as np
import torch
import torch.nn.functional as F
from buffers import Transition
from create_env import make_vector_env
from dqn_agent import DQN_Agent
from model_pool import ModelPool
from scenarios import CutIn, Slowdown, SlowdownSameLane, SpeedUp
from tqdm import tqdm
from wandb_logging import initialize_logging, log_stats

import wandb


def ego_vs_npc_pool(
    env,
    ego_agent: DQN_Agent,
    npc_pool: ModelPool,
    args,
    device,
    ego_version,
    npc_version,
    model_path,
    use_pbar=True,
):
    assert ego_agent.multi_agent, "Ego Agent must be a multi-agent agent"
    assert npc_pool.size > 0, "NPC Pool must have models"

    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None

    if args.track:
        if wandb.run is not None:
            wandb.finish()
            run = initialize_logging(
                args,
                ego_version=ego_version,
                npc_version=npc_version,
                train_ego=True,
                npc_pool_size=npc_pool.size,
                ego_pool_size=None,
                sampling=args.sampling,
            )
        else:
            run = initialize_logging(
                args,
                ego_version=ego_version,
                npc_version=npc_version,
                train_ego=True,
                npc_pool_size=npc_pool.size,
                ego_pool_size=None,
                sampling=args.sampling,
            )

    num_crashes = []
    episode_rewards = 0
    duration = 0
    episode_speed = 0
    npc_speed = 0
    ep_rew_total = np.zeros(0)
    ep_len_total = np.zeros(0)
    ep_speed_total = np.zeros(0)

    t_step = 0
    ep_num = 0

    # Choose a model from the pool every episode
    npc_pool.choose_model()
    npc_mobil = npc_pool.models[npc_pool.model_idx] == "mobil"
    config = {"use_mobil": False, "ego_vs_mobil": False}
    if npc_mobil:
        config["use_mobil"] = True
        config["ego_vs_mobil"] = npc_mobil
    else:
        config["use_mobil"] = False

    env.unwrapped.configure(config)
    obs, info = env.reset()

    flattened_ego_obs = obs[0].flatten()
    ego_obs = flattened_ego_obs[: ego_agent.n_observations]
    ego_state = torch.tensor(
        ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
    )
    ego_state[0, ego_agent.n_observations - 1] = float(not npc_mobil)
    if not npc_mobil:
        flattened_npc_obs = obs[1].flatten()
        npc_obs = flattened_npc_obs[: npc_pool.n_observations]
        npc_state = torch.tensor(
            npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
        )

    # Testing Loop
    while t_step < args.total_timesteps:
        if npc_mobil:
            ego_action = ego_agent.select_action(ego_state, t_step)
        else:
            ego_action = ego_agent.select_action(ego_state, t_step)
            npc_action = npc_pool.predict(npc_state)

        if npc_mobil:
            obs, reward, terminated, truncated, info = env.step(
                (ego_action.cpu().numpy(),)
            )
        else:
            obs, reward, terminated, truncated, info = env.step(
                (ego_action.cpu().numpy(), npc_action.cpu().numpy())
            )

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated | truncated

        int_frames = info["int_frames"]

        if args.save_trajectories:
            save_state = ego_state.cpu().numpy()
            save_action = ego_action.cpu().numpy()
            save_reward = reward.cpu().numpy()
            if terminated:
                ego_agent.trajectory_store.add(
                    Transition(save_state, save_action, None, save_reward),
                    int_frames[:, : ego_agent.n_observations],
                )
            else:
                ego_agent.trajectory_store.add(
                    Transition(
                        save_state,
                        save_action,
                        obs[0].flatten()[: ego_agent.n_observations],
                        save_reward,
                    ),
                    int_frames[:, : ego_agent.n_observations],
                )

        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[: ego_agent.n_observations]
        next_state = torch.tensor(
            ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
        )
        next_state[0, ego_agent.n_observations - 1] = float(not npc_mobil)
        ego_state = ego_agent.update(
            ego_state, ego_action, next_state, reward, terminated
        )
        if not npc_mobil:
            flattened_npc_obs = obs[1].flatten()
            npc_obs = flattened_npc_obs[: npc_pool.n_observations]
            npc_state = torch.tensor(
                npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
            )

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration += 1
        episode_speed += ego_state[0, 3].cpu().numpy()
        if not npc_mobil:
            npc_speed += npc_state[0, 3].cpu().numpy()
        else:
            npc_speed += ego_state[0, 3].cpu().numpy() + ego_state[0, 8].cpu().numpy()

        if done:
            # Save Trajectories that end in a Crash
            if args.save_trajectories:
                ego_agent.trajectory_store.clear()

            num_crashes.append(float(info["crashed"]))
            if args.track:
                ep_rew_total = np.append(ep_rew_total, episode_rewards)
                ep_len_total = np.append(ep_len_total, duration)
                ep_speed_total = np.append(ep_speed_total, episode_speed / duration)
                if ep_rew_total.size > 100:
                    ep_rew_total = np.delete(ep_rew_total, 0)
                if ep_len_total.size > 100:
                    ep_len_total = np.delete(ep_len_total, 0)
                if ep_speed_total.size > 100:
                    ep_speed_total = np.delete(ep_speed_total, 0)

                npc_pool.update_model_crashes(int(info["crashed"]))
                npc_pool.update_model_speed(ego_state[0, 3].cpu().numpy())
                # npc_pool.update_model_elo(1-int(info['final_info'][worker]['crashed']),int(info['final_info'][worker]['crashed']), info['final_info'][worker]['spawn_config'])
                # npc_pool.update_probabilities(False)

                wandb.log(
                    {
                        "rollout/ep_rew_mean": ep_rew_total.mean(),
                        "rollout/ep_len_mean": ep_len_total.mean(),
                        "rollout/num_crashes": num_crashes[-1],
                        "rollout/sr100": np.mean(num_crashes[-100:]),
                        "rollout/ego_speed_mean": episode_speed / duration,
                        "rollout/npc_speed_mean": npc_speed / duration,
                        "rollout/opponent_elo": npc_pool.opponent_elo,
                        "rollout/spawn_config": info["spawn_config"],
                        "rollout/use_mobil": npc_mobil,
                        "rollout/epsilon": ego_agent.eps_threshold,
                    },
                    step=ep_num,
                )

                for idx, model in enumerate(npc_pool.models):
                    wandb.log(
                        {
                            f"rollout/model_{idx}_ep_freq": npc_pool.model_ep_freq[idx],
                            f"rollout/model_{idx}_transition_freq": npc_pool.model_transition_freq[
                                idx
                            ],
                            f"rollout/model_{idx}_crash_freq": npc_pool.model_crash_freq[
                                idx
                            ],
                            f"rollout/model_{idx}_sr100": npc_pool.model_sr100[idx],
                            f"rollout/model_{idx}_elo": npc_pool.model_elo[idx],
                        },
                        step=ep_num,
                    )

            episode_rewards = 0
            duration = 0
            episode_speed = 0
            npc_speed = 0
            ep_num += 1
            # Update Model Choice
            npc_pool.choose_model()
            npc_mobil = npc_pool.models[npc_pool.model_idx] == "mobil"
            config = {"use_mobil": False, "ego_vs_mobil": False}
            if npc_mobil:
                config["use_mobil"] = True
                config["ego_vs_mobil"] = npc_mobil
            else:
                config["use_mobil"] = False

            env.unwrapped.configure(config)
            obs, info = env.reset()

            flattened_ego_obs = obs[0].flatten()
            ego_obs = flattened_ego_obs[: ego_agent.n_observations]
            ego_state = torch.tensor(
                ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
            )
            ego_state[0, ego_agent.n_observations - 1] = float(not npc_mobil)
            if not npc_mobil:
                flattened_npc_obs = obs[1].flatten()
                npc_obs = flattened_npc_obs[: npc_pool.n_observations]
                npc_state = torch.tensor(
                    npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
                )

        t_step += 1
        pbar.update(1)

    if use_pbar:
        pbar.close()

    ego_agent.save_model(path=model_path)

    if args.save_trajectories:
        file_path = os.path.join(
            ego_agent.trajectory_store.file_dir,
            f"{ego_agent.trajectory_store.file_interval}",
        )
        ego_agent.trajectory_store.write(file_path, "json")

    wandb.finish()
    env.close()


def npc_vs_ego_pool(
    env,
    npc_agent: DQN_Agent,
    ego_pool: ModelPool,
    args,
    device,
    ego_version,
    npc_version,
    model_path,
    use_pbar=True,
):
    assert npc_agent.multi_agent, "NPC Agent must be a multi-agent agent"
    assert ego_pool.size > 0, "EGO Pool must have models"

    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None

    if args.track:
        if wandb.run is not None:
            wandb.finish()
            run = initialize_logging(
                args,
                ego_version=ego_version,
                npc_version=npc_version,
                train_ego=False,
                npc_pool_size=None,
                ego_pool_size=ego_pool.size,
                sampling=args.sampling,
            )
        else:
            run = initialize_logging(
                args,
                ego_version=ego_version,
                npc_version=npc_version,
                train_ego=False,
                npc_pool_size=None,
                ego_pool_size=ego_pool.size,
                sampling=args.sampling,
            )

    num_crashes = []
    ego_speed = 0
    npc_speed = 0
    episode_rewards = 0
    duration = 0
    ep_rew_total = 0
    ep_len_total = 0
    ep_speed_total = 0

    t_step = 0
    ep_num = 0

    # Choose a model from the pool every episode
    ego_pool.choose_model()
    ego_mobil = ego_pool.models[ego_pool.model_idx] == "mobil"

    config = {"use_mobil": False, "ego_vs_mobil": False}
    if ego_mobil:
        config["use_mobil"] = True
        config["ego_vs_mobil"] = not ego_mobil
    else:
        config["use_mobil"] = False

    env.unwrapped.configure(config)
    obs, info = env.reset()

    if ego_mobil:
        flattened_npc_obs = obs[0].flatten()
        npc_obs = flattened_npc_obs[: npc_agent.n_observations]
        npc_state = torch.tensor(
            npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
        )
    else:
        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[: ego_pool.n_observations]
        ego_state = torch.tensor(
            ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
        )
        ego_state[0, ego_pool.n_observations - 1] = 1.0

        flattened_npc_obs = obs[1].flatten()
        npc_obs = flattened_npc_obs[: npc_agent.n_observations]
        npc_state = torch.tensor(
            npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
        )

    # Testing Loop
    while t_step < args.total_timesteps:
        if ego_mobil:
            npc_action = npc_agent.select_action(npc_state, t_step)
        else:
            ego_action = ego_pool.predict(ego_state)
            npc_action = npc_agent.select_action(npc_state, t_step)

        if ego_mobil:
            obs, reward, terminated, truncated, info = env.step(
                (npc_action.cpu().numpy(),)
            )
        else:
            obs, reward, terminated, truncated, info = env.step(
                (ego_action.cpu().numpy(), npc_action.cpu().numpy())
            )

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated | truncated

        int_frames = info["int_frames"]

        if args.save_trajectories:
            if ego_mobil:
                save_state = npc_state.cpu().numpy()
                save_action = npc_action.cpu().numpy()
            else:
                save_state = ego_state.cpu().numpy()
                save_action = ego_action.cpu().numpy()
            save_reward = reward.cpu().numpy()
            if terminated:
                npc_agent.trajectory_store.add(
                    Transition(save_state, save_action, None, save_reward),
                    int_frames[:, : npc_agent.n_observations],
                )
            else:
                npc_agent.trajectory_store.add(
                    Transition(
                        save_state,
                        save_action,
                        obs[0].flatten()[: npc_agent.n_observations],
                        save_reward,
                    ),
                    int_frames[:, : npc_agent.n_observations],
                )

        if ego_mobil:
            flattened_npc_obs = obs[0].flatten()
            npc_obs = flattened_npc_obs[: npc_agent.n_observations]
            next_state = torch.tensor(
                npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
            )
        else:
            flattened_ego_obs = obs[0].flatten()
            ego_obs = flattened_ego_obs[: ego_pool.n_observations]
            ego_state = torch.tensor(
                ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
            )
            ego_state[0, ego_pool.n_observations - 1] = 1.0

            flattened_npc_obs = obs[1].flatten()
            npc_obs = flattened_npc_obs[: npc_agent.n_observations]
            next_state = torch.tensor(
                npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
            )

        npc_state = npc_agent.update(
            npc_state, npc_action, next_state, reward, terminated
        )

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration = duration + np.ones(args.num_envs)
        if ego_mobil:
            npc_speed += npc_state[0, 3].cpu().numpy()
            ego_speed += npc_state[0, 3].cpu().numpy() + npc_state[0, 8].cpu().numpy()
        else:
            ego_speed += ego_state[0, 3].cpu().numpy()
            npc_speed += npc_state[0, 3].cpu().numpy()

        if done:
            # Save Trajectories that end in a Crash
            if args.save_trajectories:
                npc_agent.trajectory_store.save(ep_num)

            num_crashes.append(float(info["crashed"]))
            if args.track:
                ep_rew_total = np.append(ep_rew_total, episode_rewards)
                ep_len_total = np.append(ep_len_total, duration)

                ego_pool.update_model_crashes(int(info["crashed"]))
                ego_pool.update_model_speed(npc_state[0, 3].cpu().numpy())
                # ego_pool.update_model_elo(int(info['final_info'][worker]['crashed']),1-int(info['final_info'][worker]['crashed']), info['final_info'][worker]['spawn_config'])
                # ego_pool.update_probabilities(True)

                wandb.log(
                    {
                        "rollout/ep_rew_mean": np.mean(episode_rewards),
                        "rollout/ep_len_mean": ep_len_total.mean(),
                        "rollout/num_crashes": num_crashes[-1],
                        "rollout/sr100": np.mean(num_crashes[-100:]),
                        "rollout/ego_speed_mean": ego_speed / duration,
                        "rollout/npc_speed_mean": npc_speed / duration,
                        "rollout/opponent_elo": ego_pool.opponent_elo,
                        "rollout/spawn_config": info["spawn_config"],
                        "rollout/use_mobil": ego_mobil,
                        "rollout/epsilon": npc_agent.eps_threshold,
                    },
                    step=ep_num,
                )

                for idx, model in enumerate(ego_pool.models):
                    wandb.log(
                        {
                            f"rollout/model_{idx}_ep_freq": ego_pool.model_ep_freq[idx],
                            f"rollout/model_{idx}_transition_freq": ego_pool.model_transition_freq[
                                idx
                            ],
                            f"rollout/model_{idx}_crash_freq": ego_pool.model_crash_freq[
                                idx
                            ],
                            f"rollout/model_{idx}_sr100": ego_pool.model_sr100[idx],
                            f"rollout/model_{idx}_elo": ego_pool.model_elo[idx],
                        },
                        step=ep_num,
                    )

            episode_rewards = 0
            ego_speed = 0
            npc_speed = 0
            duration = 0
            ep_num += 1
            # Update Model Choice
            ego_pool.choose_model()
            ego_mobil = ego_pool.models[ego_pool.model_idx] == "mobil"
            config = {"use_mobil": False, "ego_vs_mobil": False}
            if ego_mobil:
                config["use_mobil"] = True
                config["ego_vs_mobil"] = not ego_mobil
            else:
                config["use_mobil"] = False

            env.unwrapped.configure(config)
            obs, info = env.reset()

            if ego_mobil:
                flattened_npc_obs = obs[0].flatten()
                npc_obs = flattened_npc_obs[: npc_agent.n_observations]
                npc_state = torch.tensor(
                    npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
                )
            else:
                flattened_ego_obs = obs[0].flatten()
                ego_obs = flattened_ego_obs[: ego_pool.n_observations]
                ego_state = torch.tensor(
                    ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
                )
                ego_state[0, ego_pool.n_observations - 1] = 1.0

                flattened_npc_obs = obs[1].flatten()
                npc_obs = flattened_npc_obs[: npc_agent.n_observations]
                npc_state = torch.tensor(
                    npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
                )

        t_step += 1
        pbar.update(1)

    if use_pbar:
        pbar.close()

    npc_agent.save_model(path=model_path)

    if args.save_trajectories:
        file_path = os.path.join(
            npc_agent.trajectory_store.file_dir,
            f"{npc_agent.trajectory_store.file_interval}",
        )
        npc_agent.trajectory_store.write(file_path, "json")

    wandb.finish()
    env.close()


def pool_evaluation(
    env,
    cycle,
    ego_pool: ModelPool,
    npc_pool: ModelPool,
    args,
    device,
    ego: bool,
    use_pbar=True,
    n_obs=25,
):
    not_ego = not ego
    ego_eval = ego_pool.init_eval(args.evaluation_episodes, latest_model=ego)
    npc_eval = npc_pool.init_eval(args.evaluation_episodes, latest_model=not_ego)

    ego_pool.set_opponent_elo(npc_pool.model_elo[npc_pool.model_idx])
    npc_pool.set_opponent_elo(ego_pool.model_elo[ego_pool.model_idx])

    n_egos = ego_pool.size
    n_npcs = npc_pool.size
    n_agents = n_npcs if ego else n_egos

    if use_pbar:
        pbar = tqdm(total=args.evaluation_episodes * (n_agents))
    else:
        pbar = None

    ep_num = 0

    ego_state = None
    npc_state = None

    # Testing Loop
    while ego_eval or npc_eval:
        done = False

        ego_mobil = ego_pool.models[ego_pool.model_idx] == "mobil"
        npc_mobil = npc_pool.models[npc_pool.model_idx] == "mobil"

        config = {"use_mobil": False, "ego_vs_mobil": False}
        if ego_mobil or npc_mobil:
            config["use_mobil"] = True
            config["ego_vs_mobil"] = npc_mobil
        else:
            config["use_mobil"] = False

        env.unwrapped.configure(config)
        obs, info = env.reset()

        if ego_mobil:
            flattened_npc_obs = obs[0].flatten()
            npc_obs = flattened_npc_obs[: npc_pool.n_observations]
            npc_state = torch.tensor(
                npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
            )
        elif npc_mobil:
            flattened_ego_obs = obs[0].flatten()
            ego_obs = flattened_ego_obs[: ego_pool.n_observations]
            ego_state = torch.tensor(
                ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
            )
            ego_state[0, ego_pool.n_observations - 1] = float(not npc_mobil)
        else:
            flattened_ego_obs = obs[0].flatten()
            ego_obs = flattened_ego_obs[: ego_pool.n_observations]
            ego_state = torch.tensor(
                ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
            )
            ego_state[0, ego_pool.n_observations - 1] = 1.0

            flattened_npc_obs = obs[1].flatten()
            npc_obs = flattened_npc_obs[: npc_pool.n_observations]
            npc_state = torch.tensor(
                npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
            )

        while not done:
            if ego_mobil:
                npc_action = npc_pool.predict(npc_state)
            elif npc_mobil:
                ego_action = ego_pool.predict(ego_state)
            else:
                ego_action = ego_pool.predict(ego_state)
                npc_action = npc_pool.predict(npc_state)

            if ego_mobil:
                obs, reward, terminated, truncated, info = env.step(
                    (npc_action.cpu().numpy(),)
                )
            elif npc_mobil:
                obs, reward, terminated, truncated, info = env.step(
                    (ego_action.cpu().numpy(),)
                )
            else:
                obs, reward, terminated, truncated, info = env.step(
                    (ego_action.cpu().numpy(), npc_action.cpu().numpy())
                )

            done = terminated | truncated

            if ego_mobil:
                flattened_npc_obs = obs[0].flatten()
                npc_obs = flattened_npc_obs[: npc_pool.n_observations]
                npc_state = torch.tensor(
                    npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
                )
            elif npc_mobil:
                flattened_ego_obs = obs[0].flatten()
                ego_obs = flattened_ego_obs[: ego_pool.n_observations]
                ego_state = torch.tensor(
                    ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
                )
                ego_state[0, ego_pool.n_observations - 1] = float(not npc_mobil)
            else:
                flattened_ego_obs = obs[0].flatten()
                ego_obs = flattened_ego_obs[: ego_pool.n_observations]
                ego_state = torch.tensor(
                    ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
                )
                ego_state[0, ego_pool.n_observations - 1] = 1.0

                flattened_npc_obs = obs[1].flatten()
                npc_obs = flattened_npc_obs[: npc_pool.n_observations]
                npc_state = torch.tensor(
                    npc_obs.reshape(1, len(npc_obs)), dtype=torch.float32, device=device
                )

            if done:
                crash = int(info["crashed"])
                ego_pool.update_model_elo(crash, 1 - crash)
                npc_pool.update_model_elo(1 - crash, crash)

                ego_pool.log_model_pool(
                    cycle, ep_num, npc_pool.model_idx, crash, 1 - crash
                )
                npc_pool.log_model_pool(
                    cycle, ep_num, ego_pool.model_idx, 1 - crash, crash
                )

                ego_eval = ego_pool.choose_eval_opponent(randomized=True)
                npc_eval = npc_pool.choose_eval_opponent(randomized=True)

                ego_pool.set_opponent_elo(npc_pool.model_elo[npc_pool.model_idx])
                npc_pool.set_opponent_elo(ego_pool.model_elo[ego_pool.model_idx])

                ep_num += 1
                pbar.update(1)

    if use_pbar:
        pbar.close()

    env.close()


def train_ego(
    env,
    ego_agent: DQN_Agent,
    args,
    device,
    model_path,
    use_pbar=True,
    extra_state=False,
):
    assert ego_agent.multi_agent, "Ego Agent must be a multi-agent agent"

    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None

    if args.track:
        if wandb.run is not None:
            wandb.finish()
            run = initialize_logging(
                args, train_ego=True, ego_pool_size=None, sampling=args.sampling
            )
        else:
            run = initialize_logging(
                args, train_ego=True, ego_pool_size=None, sampling=args.sampling
            )

    num_crashes = []
    episode_rewards = 0
    duration = 0
    episode_speed = 0
    npc_speed = 0
    ep_rew_total = np.zeros(0)
    ep_len_total = np.zeros(0)
    ep_speed_total = np.zeros(0)
    right_lane_reward = 0.0
    high_speed_reward = 0.0
    collision_reward = 0.0

    t_step = 0
    ep_num = 0

    # Choose a model from the pool every episode
    npc_mobil = True
    config = {"use_mobil": True, "ego_vs_mobil": npc_mobil}
    env.unwrapped.configure(config)
    obs, info = env.reset()

    flattened_ego_obs = obs[0].flatten()
    ego_obs = flattened_ego_obs[: ego_agent.n_observations]
    ego_state = torch.tensor(
        ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
    )
    if extra_state:
        ego_state[0, ego_agent.n_observations - 1] = float(not npc_mobil)

    # Testing Loop
    while t_step < args.total_timesteps:
        ego_action = ego_agent.select_action(ego_state, t_step)
        obs, reward, terminated, truncated, info = env.step((ego_action.cpu().numpy(),))

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated | truncated

        int_frames = info["int_frames"]

        if args.save_trajectories:
            save_state = ego_state.cpu().numpy()
            save_action = ego_action.cpu().numpy()
            save_reward = reward.cpu().numpy()
            if terminated:
                ego_agent.trajectory_store.add(
                    Transition(save_state, save_action, None, save_reward),
                    int_frames[:, : ego_agent.n_observations],
                )
            else:
                ego_agent.trajectory_store.add(
                    Transition(
                        save_state,
                        save_action,
                        obs[0].flatten()[: ego_agent.n_observations],
                        save_reward,
                    ),
                    int_frames[:, : ego_agent.n_observations],
                )

        flattened_ego_obs = obs[0].flatten()
        ego_obs = flattened_ego_obs[: ego_agent.n_observations]
        next_state = torch.tensor(
            ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
        )
        if extra_state:
            next_state[0, ego_agent.n_observations - 1] = float(not npc_mobil)
        ego_state = ego_agent.update(
            ego_state, ego_action, next_state, reward, terminated
        )

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration += 1
        episode_speed += ego_state[0, 3].cpu().numpy()
        npc_speed += ego_state[0, 3].cpu().numpy() + ego_state[0, 8].cpu().numpy()

        high_speed_reward += info["rewards"]["high_speed_reward"]
        right_lane_reward += info["rewards"]["right_lane_reward"]
        collision_reward += info["rewards"]["collision_reward"]

        if done:
            # Save Trajectories that end in a Crash
            if args.save_trajectories:
                ego_agent.trajectory_store.save(ep_num)

            num_crashes.append(float(info["crashed"]))
            if args.track:
                ep_rew_total = np.append(ep_rew_total, episode_rewards)
                ep_len_total = np.append(ep_len_total, duration)
                ep_speed_total = np.append(ep_speed_total, episode_speed / duration)
                if ep_rew_total.size > 100:
                    ep_rew_total = np.delete(ep_rew_total, 0)
                if ep_len_total.size > 100:
                    ep_len_total = np.delete(ep_len_total, 0)
                if ep_speed_total.size > 100:
                    ep_speed_total = np.delete(ep_speed_total, 0)

                wandb.log(
                    {
                        "rollout/ep_rew_mean": ep_rew_total.mean(),
                        "rollout/ep_len_mean": ep_len_total.mean(),
                        "rollout/num_crashes": num_crashes[-1],
                        "rollout/sr100": np.mean(num_crashes[-100:]),
                        "rollout/ego_speed_mean": episode_speed / duration,
                        "rollout/npc_speed_mean": npc_speed / duration,
                        "rollout/spawn_config": info["spawn_config"],
                        "rollout/use_mobil": npc_mobil,
                        "rollout/epsilon": ego_agent.eps_threshold,
                        "rollout/right_lane_reward": right_lane_reward / duration,
                        "rollout/high_speed_reward": high_speed_reward / duration,
                        "rollout/collision_reward": collision_reward / duration,
                    },
                    step=ep_num,
                )

            episode_rewards = 0
            duration = 0
            episode_speed = 0
            npc_speed = 0
            ep_num += 1

            right_lane_reward = 0.0
            high_speed_reward = 0.0
            collision_reward = 0.0

            obs, info = env.reset()

            flattened_ego_obs = obs[0].flatten()
            ego_obs = flattened_ego_obs[: ego_agent.n_observations]
            ego_state = torch.tensor(
                ego_obs.reshape(1, len(ego_obs)), dtype=torch.float32, device=device
            )
            if extra_state:
                ego_state[0, ego_agent.n_observations - 1] = float(not npc_mobil)

        t_step += 1
        pbar.update(1)

    if use_pbar:
        pbar.close()

    ego_agent.save_model(path=model_path)

    if args.save_trajectories:
        file_path = os.path.join(
            ego_agent.trajectory_store.file_dir,
            f"{ego_agent.trajectory_store.file_interval}",
        )
        ego_agent.trajectory_store.write(file_path, "json")

    wandb.finish()
    env.close()


def test_agents(
    env,
    ego_agent: DQN_Agent,
    npc_agent: DQN_Agent,
    args,
    device,
    ego_version,
    npc_version,
    use_pbar=True,
):
    if use_pbar:
        pbar = tqdm(total=args.evaluation_episodes)
    else:
        pbar = None

    if args.track:
        if wandb.run is not None:
            wandb.finish()
            run = initialize_logging(
                args,
                ego_version=ego_version,
                npc_version=npc_version,
                train_ego=False,
                eval=True,
                sampling=args.sampling,
            )
        else:
            run = initialize_logging(
                args,
                ego_version=ego_version,
                npc_version=npc_version,
                train_ego=False,
                eval=True,
                sampling=args.sampling,
            )

    episode_statistics = helpers.initialize_stats()

    t_step = 0

    obs, info = env.reset()
    ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, npc_agent, device)

    # Testing Loop
    while episode_statistics["episode_num"] < args.evaluation_episodes:
        ego_action = ego_agent.predict(ego_state)
        npc_action = npc_agent.predict(npc_state)

        obs, reward, terminated, truncated, info = env.step(
            (ego_action.cpu().numpy(), npc_action.cpu().numpy())
        )

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated | truncated

        int_frames = info["int_frames"]

        if args.save_trajectories:
            save_state = ego_state.cpu().numpy()
            save_action = ego_action.cpu().numpy()
            save_reward = reward.cpu().numpy()
            if terminated:
                ego_agent.trajectory_store.add(
                    Transition(save_state, save_action, None, save_reward),
                    int_frames[:, : ego_agent.n_observations],
                )
            else:
                ego_agent.trajectory_store.add(
                    Transition(
                        save_state,
                        save_action,
                        obs[0].flatten()[: ego_agent.n_observations],
                        save_reward,
                    ),
                    int_frames[:, : ego_agent.n_observations],
                )

        ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, npc_agent, device)
        helpers.populate_stats(
            info, npc_agent, ego_state, npc_state, reward, episode_statistics, False
        )

        if done:
            # Save Trajectories that end in a Crash
            if args.save_trajectories:
                ego_agent.trajectory_store.save(episode_statistics["episode_num"])

            if args.track:
                log_stats(info, episode_statistics, ego=False)

            helpers.reset_stats(episode_statistics)

            obs, info = env.reset()
            ego_state, npc_state = helpers.obs_to_state(
                obs, ego_agent, npc_agent, device
            )

            pbar.update(1)
        t_step += 1

    if use_pbar:
        pbar.close()

    if args.save_trajectories:
        file_path = os.path.join(
            ego_agent.trajectory_store.file_dir,
            f"{ego_agent.trajectory_store.file_interval}",
        )
        ego_agent.trajectory_store.write(file_path, "json")

    wandb.finish()


def train_agents(
    env,
    ego_agent: DQN_Agent,
    npc_agent: DQN_Agent,
    config: dict,
    device,
    model_path,
    train_ego: bool = True,
    use_pbar: bool = True,
):
    """
    A combined function that can:
      - Run EGO vs. NPC (train_ego=True)
      - Run NPC vs. EGO (train_ego=False)

    Args:
        env: The environment.
        ego_agent (DQN_Agent): The ego agent (multi-agent).
        npc_agent (DQN_Agent): The npc agent (multi-agent).
        args: Training/testing arguments (must contain total_timesteps, track, etc.).
        device: Torch device.
        model_path: Where to save the model of whichever agent is being trained.
        train_ego: If True, ego_agent is training; if False, npc_agent is training.
        use_pbar: Whether to display a progress bar.
    """

    # Decide which agent does "select_action" vs "predict"
    if train_ego:
        train_agent = ego_agent
        fixed_agent = npc_agent
        # For logging
        train_ego_flag = True
    else:
        train_agent = npc_agent
        fixed_agent = ego_agent
        train_ego_flag = False

    total_timesteps = config.get("total_timesteps", 100000)
    # Progress bar
    if use_pbar:
        pbar = tqdm(total=total_timesteps)
    else:
        pbar = None

    # Logging
    if config.get("track", False):
        if wandb.run is not None:
            wandb.finish()
        run = initialize_logging(
            config, train_ego=train_ego_flag, npc_pool_size=None, ego_pool_size=None
        )

    episode_statistics = helpers.initialize_stats()
    t_step = 0

    obs, info = env.reset()
    ego_state, npc_state = helpers.obs_to_state(obs, ego_agent, npc_agent, device)

    # Main loop
    while t_step < total_timesteps:
        # The training agent calls select_action
        # The fixed agent calls predict
        if train_ego:
            ego_action = train_agent.select_action(ego_state, t_step)
            npc_action = fixed_agent.predict(npc_state)
        else:
            ego_action = fixed_agent.predict(ego_state)
            npc_action = train_agent.select_action(npc_state, t_step)

        # Step in the environment
        obs, reward, terminated, truncated, info = env.step(
            (ego_action.cpu().numpy(), npc_action.cpu().numpy())
        )

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated or truncated
        int_frames = info["int_frames"]

        # Save trajectories for the *training agent*
        # Here we assume we only store data for the agent that’s currently training.
        if config.get("save_trajectories", False):
            if train_ego:
                # Ego is being trained
                save_state = ego_state.cpu().numpy()
                save_action = ego_action.cpu().numpy()
            else:
                # NPC is being trained
                save_state = npc_state.cpu().numpy()
                save_action = npc_action.cpu().numpy()

            save_reward = reward.cpu().numpy()

            if terminated:
                train_agent.trajectory_store.add(
                    Transition(save_state, save_action, None, save_reward),
                    int_frames[:, : train_agent.n_observations],
                )
            else:
                # obs[0] is ego's observation, obs[1] is NPC's observation
                # so we index properly depending on who is training.
                obs_index = 0 if train_ego else 1
                train_agent.trajectory_store.add(
                    Transition(
                        save_state,
                        save_action,
                        obs[obs_index].flatten()[: train_agent.n_observations],
                        save_reward,
                    ),
                    int_frames[:, : train_agent.n_observations],
                )

        # Update next state
        if train_ego:
            next_state, npc_state = helpers.obs_to_state(
                obs, ego_agent, npc_agent, device
            )
            ego_state = ego_agent.update(
                ego_state, ego_action, next_state, reward, terminated
            )
        else:
            ego_state, next_state = helpers.obs_to_state(
                obs, ego_agent, npc_agent, device
            )
            npc_state = npc_agent.update(
                npc_state, npc_action, next_state, reward, terminated
            )

        # Populate stats
        # The original functions used:
        #   - populate_stats(..., ego=True) in ego_vs_npc
        #   - populate_stats(..., ego=False) in npc_vs_ego
        # So we simply use train_ego_flag to specify which side is "ego"
        helpers.populate_stats(
            info,
            train_agent,
            ego_state,
            npc_state,
            reward.cpu().numpy(),
            episode_statistics,
            is_ego=train_ego_flag,
        )

        if done:
            # Save trajectories that end in a Crash
            if config.get("save_trajectories", False):
                train_agent.trajectory_store.save(episode_statistics["episode_num"])

            # Logging
            if config.get("track", False):
                log_stats(info, episode_statistics, ego=train_ego_flag)

            helpers.reset_stats(episode_statistics)

            obs, info = env.reset()
            ego_state, npc_state = helpers.obs_to_state(
                obs, ego_agent, npc_agent, device
            )

        t_step += 1
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Save the model of whichever agent we trained
    train_agent.save_model(path=model_path)

    # Optionally, write out the stored trajectories
    if config.get("save_trajectories", False):
        file_path = os.path.join(
            train_agent.trajectory_store.file_dir,
            f"{train_agent.trajectory_store.file_interval}",
        )
        train_agent.trajectory_store.write(file_path, "json")

    wandb.finish()
