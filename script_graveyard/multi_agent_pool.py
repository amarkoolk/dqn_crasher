import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import random
import json
import math
import numpy as np
from typing import TypeAlias, List, Tuple
from collections import namedtuple

from buffers import ReplayMemory, PrioritizedExperienceReplay, Transition

from tqdm import tqdm
import wandb
import gymnasium as gym
from dqn_agent import DQN, DQN_Agent, TrajectoryStore
from model_pool import ModelPool
from wandb_logging import initialize_logging
from config import load_config
from create_env import make_env, make_vector_env


def multi_agent_loop(
    env,
    mode,
    train_ego,
    ego_version,
    npc_version,
    ego_agent,
    npc_agent,
    env_config,
    args,
    device,
    trajectory_path,
    record_video=False,
    use_pbar=True,
    ego_pool=None,
    npc_pool=None,
):
    if use_pbar:
        pbar = tqdm(total=args.total_timesteps)
    else:
        pbar = None

    print(f"Mode: {mode}, Ego Version: {ego_version}, NPC Version: {npc_version}")

    if args.track:
        if wandb.run is not None:
            wandb.finish()
            run = initialize_logging(
                args,
                ego_version,
                npc_version,
                train_ego,
                npc_pool_size=npc_pool.size if npc_pool else None,
                ego_pool_size=ego_pool.size if ego_pool else None,
            )
        else:
            run = initialize_logging(
                args,
                ego_version,
                npc_version,
                train_ego,
                npc_pool_size=npc_pool.size if npc_pool else None,
                ego_pool_size=ego_pool.size if ego_pool else None,
            )

    (
        num_crashes,
        episode_rewards,
        duration,
        episode_speed,
        ep_rew_total,
        ep_len_total,
        ep_speed_total,
    ) = (
        [],
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(args.num_envs),
        np.zeros(0),
        np.zeros(0),
        np.zeros(0),
    )

    t_step = 0
    ep_num = 0

    obs, info = env.reset()
    ego_state = torch.tensor(
        obs[0].reshape(args.num_envs, ego_agent.n_observations),
        dtype=torch.float32,
        device=device,
    )
    npc_state = torch.tensor(
        obs[1].reshape(args.num_envs, npc_agent.n_observations),
        dtype=torch.float32,
        device=device,
    )

    if npc_pool:
        npc_pool.choose_model()
        npc_state = torch.tensor(
            obs[1].reshape(args.num_envs, npc_pool.models[0].n_observations),
            dtype=torch.float32,
            device=device,
        )
    if ego_pool:
        ego_pool.choose_model()
        ego_state = torch.tensor(
            obs[0].reshape(args.num_envs, ego_pool.models[0].n_observations),
            dtype=torch.float32,
            device=device,
        )

    while t_step < args.total_timesteps:
        if mode == "train_ego":
            ego_action = torch.squeeze(ego_agent.select_action(ego_state, env, t_step))
            if npc_pool:
                npc_action = torch.squeeze(npc_pool.predict(npc_state))
            else:
                npc_action = torch.squeeze(npc_agent.predict(npc_state))
        elif mode == "train_npc":
            npc_action = torch.squeeze(npc_agent.select_action(npc_state, env, t_step))
            if ego_pool:
                ego_action = torch.squeeze(ego_pool.predict(ego_state))
            else:
                ego_action = torch.squeeze(ego_agent.predict(ego_state))
        else:
            if ego_pool:
                ego_action = torch.squeeze(ego_pool.predict(ego_state))
            else:
                ego_action = torch.squeeze(ego_agent.predict(ego_state))
            if npc_pool:
                npc_action = torch.squeeze(npc_pool.predict(npc_state))
            else:
                npc_action = torch.squeeze(npc_agent.predict(npc_state))

        if args.num_envs == 1:
            ego_action = ego_action.view(1, 1)
            npc_action = npc_action.view(1, 1)

        obs, reward, terminated, truncated, info = env.step(
            (ego_action.cpu().numpy(), npc_action.cpu().numpy())
        )

        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = terminated | truncated

        if done:
            int_frames = info["final_info"][0]["int_frames"]
        else:
            int_frames = info["int_frames"][0]

        if args.save_trajectories:
            for worker in range(args.num_envs):
                if terminated[worker]:
                    if mode == "train_ego" or mode == "eval":
                        ego_agent.trajectory_store.add(
                            worker,
                            Transition(
                                ego_state[worker].cpu().numpy(),
                                ego_action[worker].cpu().numpy(),
                                None,
                                reward[worker].cpu().numpy(),
                            ),
                            int_frames,
                        )
                    else:
                        npc_agent.trajectory_store.add(
                            worker,
                            Transition(
                                npc_state[worker].cpu().numpy(),
                                npc_action[worker].cpu().numpy(),
                                None,
                                reward[worker].cpu().numpy(),
                            ),
                            int_frames,
                        )
                else:
                    if mode == "train_ego" or mode == "eval":
                        ego_agent.trajectory_store.add(
                            worker,
                            Transition(
                                ego_state[worker].cpu().numpy(),
                                ego_action[worker].cpu().numpy(),
                                obs[0][worker].flatten(),
                                reward[worker].cpu().numpy(),
                            ),
                            int_frames,
                        )
                    else:
                        npc_agent.trajectory_store.add(
                            worker,
                            Transition(
                                npc_state[worker].cpu().numpy(),
                                npc_action[worker].cpu().numpy(),
                                obs[1][worker].flatten(),
                                reward[worker].cpu().numpy(),
                            ),
                            int_frames,
                        )

        if mode == "train_ego":
            ego_state = ego_agent.update(
                ego_state, ego_action, obs[0], reward, terminated
            )
            if npc_pool:
                npc_state = torch.tensor(
                    obs[1].reshape(args.num_envs, npc_pool.models[0].n_observations),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                npc_state = torch.tensor(
                    obs[1].reshape(args.num_envs, npc_agent.n_observations),
                    dtype=torch.float32,
                    device=device,
                )
        elif mode == "train_npc":
            if ego_pool:
                ego_state = torch.tensor(
                    obs[0].reshape(args.num_envs, ego_pool.models[0].n_observations),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                ego_state = torch.tensor(
                    obs[0].reshape(args.num_envs, ego_agent.n_observations),
                    dtype=torch.float32,
                    device=device,
                )
            npc_state = npc_agent.update(
                npc_state, npc_action, obs[1], reward, terminated
            )
        else:
            if ego_pool:
                ego_state = torch.tensor(
                    obs[0].reshape(args.num_envs, ego_pool.models[0].n_observations),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                ego_state = torch.tensor(
                    obs[0].reshape(args.num_envs, ego_agent.n_observations),
                    dtype=torch.float32,
                    device=device,
                )
            if npc_pool:
                npc_state = torch.tensor(
                    obs[1].reshape(args.num_envs, npc_pool.models[0].n_observations),
                    dtype=torch.float32,
                    device=device,
                )
            else:
                npc_state = torch.tensor(
                    obs[1].reshape(args.num_envs, npc_agent.n_observations),
                    dtype=torch.float32,
                    device=device,
                )

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration = duration + np.ones(args.num_envs)
        episode_speed = episode_speed + np.linalg.norm(
            ego_state[:, 3:5].cpu().numpy(), axis=1
        )

        for worker in range(args.num_envs):
            if done[worker]:
                # Save Trajectories that end in a Crash
                if args.save_trajectories:
                    if mode == "train_ego" or mode == "eval":
                        ego_agent.trajectory_store.save(worker, ep_num)
                    else:
                        npc_agent.trajectory_store.save(worker, ep_num)

                num_crashes.append(float(info["final_info"][worker]["crashed"]))
                if args.track:
                    ep_rew_total = np.append(ep_rew_total, episode_rewards[worker])
                    ep_len_total = np.append(ep_len_total, duration[worker])
                    ep_speed_total = np.append(
                        ep_speed_total, episode_speed[worker] / duration[worker]
                    )
                    if ep_rew_total.size > 100:
                        ep_rew_total = np.delete(ep_rew_total, 0)
                    if ep_len_total.size > 100:
                        ep_len_total = np.delete(ep_len_total, 0)
                    if ep_speed_total.size > 100:
                        ep_speed_total = np.delete(ep_speed_total, 0)

                    if npc_pool:
                        npc_pool.update_model_crashes(
                            int(info["final_info"][worker]["crashed"])
                        )
                    if ego_pool:
                        ego_pool.update_model_crashes(
                            int(info["final_info"][worker]["crashed"])
                        )

                    wandb.log(
                        {
                            "rollout/ep_rew_mean": ep_rew_total.mean(),
                            "rollout/ep_len_mean": ep_len_total.mean(),
                            "rollout/num_crashes": num_crashes[-1],
                            "rollout/sr100": np.mean(num_crashes[-100:]),
                            "rollout/ego_speed_mean": ep_speed_total.mean(),
                        },
                        step=ep_num,
                    )

                    if npc_pool:
                        for idx, model in enumerate(npc_pool.models):
                            wandb.log(
                                {
                                    f"rollout/model_{idx}_ep_freq": npc_pool.model_ep_freq[
                                        idx
                                    ],
                                    f"rollout/model_{idx}_transition_freq": npc_pool.model_transition_freq[
                                        idx
                                    ],
                                    f"rollout/model_{idx}_crash_freq": npc_pool.model_crash_freq[
                                        idx
                                    ],
                                    f"rollout/model_{idx}_sr100": npc_pool.model_sr100[
                                        idx
                                    ],
                                },
                                step=ep_num,
                            )
                    if ego_pool:
                        for idx, model in enumerate(ego_pool.models):
                            wandb.log(
                                {
                                    f"rollout/model_{idx}_ep_freq": ego_pool.model_ep_freq[
                                        idx
                                    ],
                                    f"rollout/model_{idx}_transition_freq": ego_pool.model_transition_freq[
                                        idx
                                    ],
                                    f"rollout/model_{idx}_crash_freq": ego_pool.model_crash_freq[
                                        idx
                                    ],
                                    f"rollout/model_{idx}_sr100": ego_pool.model_sr100[
                                        idx
                                    ],
                                },
                                step=ep_num,
                            )

                episode_rewards[worker] = 0
                duration[worker] = 0
                episode_speed[worker] = 0
                ep_num += 1

                if npc_pool:
                    npc_pool.choose_model()
                if ego_pool:
                    ego_pool.choose_model()

            t_step += 1
            pbar.update(1)

    if use_pbar:
        pbar.close()

    if mode == "train_ego":
        ego_agent.save_model(
            path=f"E{ego_version}_V{npc_version}_TrainEgo_{str(True)}.pth"
        )
    elif mode == "train_npc":
        npc_agent.save_model(
            path=f"E{ego_version}_V{npc_version}_TrainEgo_{str(False)}.pth"
        )

    if args.save_trajectories:
        if mode == "train_ego":
            file_path = os.path.join(
                ego_agent.trajectory_store.file_dir,
                f"{ego_agent.trajectory_store.file_interval}",
            )
            ego_agent.trajectory_store.write(file_path, "json")
        else:
            file_path = os.path.join(
                npc_agent.trajectory_store.file_dir,
                f"{npc_agent.trajectory_store.file_interval}",
            )
            npc_agent.trajectory_store.write(file_path, "json")

    wandb.finish()
    env.close()
