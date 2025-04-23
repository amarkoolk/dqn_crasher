# runner.py
import os
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import helpers
import wandb
from wandb_logging import initialize_logging, log_stats
from dqn_agent import DQN_Agent
import policies
from buffers import Transition
from training_distribution import DistributionScheduler

class MultiAgentRunner:
    def __init__(self, env_name, config, gym_cfg, device, policy_a, policy_b):
        self.env_name = env_name
        self.cfg      = config
        self.gym_cfg  = gym_cfg
        self.dev      = device
        self.A        = policy_a
        self.B        = policy_b

        # a single env instance just to pull action_space / obs dims
        tmp = gym.make(env_name, config=gym_cfg)
        self.action_space = tmp.action_space[0]
        self.n_actions = self.action_space.n
        self.n_obs = 10 * config.get('frame_stack', 1)
        tmp.close()

    def train(self, train_player="A"):
        total_timesteps = self.cfg.get('total_timesteps', 100000)
        t     = 0
        pbar  = tqdm(total=total_timesteps)

        if self.cfg.get('track', False):
            initialize_logging(self.cfg, train_ego=(train_player=="A"))
            stats = helpers.initialize_stats()

        while t < total_timesteps:
            steps, _ = self._run_episode(train_mode=True, train_player=train_player, t_start=t, stats=stats, pbar=pbar)
            t += steps

        if train_player == "A" and type(self.A.agent) == DQN_Agent:
            self.A.save_model(self.cfg.get('model_save_path', 'models/model.pth'))
        elif type(self.B.agent) == DQN_Agent:
            self.B.save_model(self.cfg.get('model_save_path', 'models/model.pth'))

        if pbar: pbar.close()
        wandb.finish()

    def test(self):
        total_eps = self.cfg.get('total_episodes', 100)
        eps_done  = 0
        pbar      = tqdm(total=total_eps)

        if self.cfg.get('track', False):
            initialize_logging(self.cfg, train_ego=False, eval=True)
            stats = helpers.initialize_stats()

        while eps_done < total_eps:
            _, ended = self._run_episode(train_mode=False, train_player=None, t_start=0, stats=stats, pbar=pbar)
            eps_done += ended

            if pbar is not None:
                pbar.update(1)

        if pbar: 
            pbar.close()

        wandb.finish()

    def _run_episode(self, train_mode, train_player, t_start, stats, pbar):
        total_timesteps = self.cfg.get('total_timesteps', 100000)
        env  = gym.make(self.env_name, config=self.gym_cfg, render_mode="rgb_array")
        obs, info = env.reset()
        ego_s, npc_s = helpers.obs_to_state(obs, self.n_obs, self.dev)

        # reset both policies
        self.A.reset(ego_s, npc_s, info, train_mode)
        self.B.reset(npc_s, ego_s, info, train_mode)  # note swapped order

        if self.cfg.get('save_trajectories', False):
            self.A.store.start_episode(stats['episode_num'])

        t     = t_start
        done  = False
        steps = 0

        while not done and (not train_mode or t < total_timesteps):
            # pick actions
            if train_mode:
                if train_player == "A":
                    a_A = self.A.select_action(ego_s, npc_s, t)
                    a_B = self.B.predict(npc_s, ego_s)
                else:
                    a_A = self.A.predict(ego_s, npc_s)
                    a_B = self.B.select_action(npc_s, ego_s, t)
            else:
                a_A = self.A.predict(ego_s, npc_s)
                a_B = self.B.predict(npc_s, ego_s)

            actions = (a_A.cpu().numpy(), a_B.cpu().numpy())

            # env step
            obs, reward, term, trunc, info = env.step(actions)
            done = term or trunc

            if self.cfg.get('render', False):
                env.render()

            # update trainable policy
            if train_mode:
                if train_player == "A":
                    next_A, next_B = helpers.obs_to_state(obs, self.n_obs, self.dev)
                    transition = Transition(ego_s, a_A, next_A, torch.tensor(reward, device=self.dev))
                    ego_s = self.A.update(transition, term)

                    transition = Transition(npc_s, a_B, next_B, torch.tensor(reward, device=self.dev))
                    npc_s = next_B
                    if type(self.B) == policies.ScenarioPolicy:
                        self.B.update(transition, None)

                else:
                    next_A, next_B = helpers.obs_to_state(obs, self.n_obs, self.dev)
                    transition = Transition(npc_s, a_B, next_B, torch.tensor(reward, device=self.dev))
                    npc_s  = self.B.update(transition, term)
                    ego_s  = next_A
            else:
                ego_s, npc_s = helpers.obs_to_state(obs, self.n_obs, self.dev)
                transition_A = Transition(ego_s, a_A, ego_s, torch.tensor(reward, device=self.dev))
                if type(self.A) == policies.ScenarioPolicy:
                    self.A.update(transition_A, None)
                
                transition_B = Transition(npc_s, a_A, npc_s, torch.tensor(reward, device=self.dev))
                if type(self.B) == policies.ScenarioPolicy:
                    self.B.update(transition_B, None)

            # store trajectories for the  policy
            if self.cfg.get('save_trajectories'):
                self.A.store.add(transition_A)

            # stats & logging
            if train_player is None or train_player=='A':
                is_ego = True

            helpers.populate_stats(info,
                agent=(self.A.agent if train_player=="A" else self.B.agent),
                ego_state=ego_s, npc_state=npc_s,
                reward=reward, episode_statistics=stats,
                is_ego=is_ego
            )


            t += 1
            steps += 1
            if train_mode and pbar:
                pbar.update(1)

        if self.cfg.get('track', False):
            log_stats(info, stats, ego=(train_player=="A"))
            helpers.reset_stats(stats)

        if self.cfg.get('save_trajectories', False):
            self.A.store.end_episode()


        env.close()
        return steps, (1 if not train_mode else 0)