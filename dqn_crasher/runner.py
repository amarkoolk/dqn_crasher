# runner.py
from logging import raiseExceptions
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

class MultiAgentRunner:
    def __init__(self, env_name, config, gym_cfg, device, policy_a, policy_b):
        self.env_name = env_name
        self.cfg : dict      = config
        self.gym_cfg : dict  = gym_cfg
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
            steps = self._run_episode(train_player=train_player, t_start=t, stats=stats, pbar=pbar)
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
            _ = self._run_episode(train_player=None, t_start=0, stats=stats, pbar=None)
            eps_done += 1

            if pbar is not None:
                pbar.update(1)

        if pbar: pbar.close()
        wandb.finish()

    def _set_config(self, train_player):
        def unwrap(policy_obj):
            # if it's a distribution, grab the inner scenario policy
            return policy_obj.current_policy if isinstance(policy_obj, policies.PolicyDistribution) else policy_obj

        for name in ('A', 'B'):
            policy = unwrap(getattr(self, name))
            # ScenarioPolicy always gets configured
            if isinstance(policy, policies.ScenarioPolicy):
                return policy.set_config(self.gym_cfg)
            # DQNPolicy only if it's the one we're training
            if isinstance(policy, policies.DQNPolicy) and name == train_player:
                return policy.set_config(self.gym_cfg)

        raise Exception("Invalid Config Combination")

    def _run_episode(self, train_player, t_start, stats, pbar):
        total_timesteps = self.cfg.get('total_timesteps', 100000)

        # reset both policies
        self.A.reset()
        self.B.reset()

        self._set_config(train_player)

        env  = gym.make(self.env_name, config=self.gym_cfg, render_mode="rgb_array")
        obs, info = env.reset()
        ego_s, npc_s = helpers.obs_to_state(obs, self.n_obs, self.dev)

        self.A.set_state(ego_s, npc_s)
        self.B.set_state(ego_s, npc_s)

        if self.cfg.get('save_trajectories', False):
            self.A.store.start_episode(stats['episode_num'])

        t     = t_start
        done  = False
        steps = 0

        while not done and t < total_timesteps:

            a_A = self.A.select_action(ego_s, npc_s, t)
            a_B = self.B.select_action(npc_s, ego_s, t)

            actions = (a_A.cpu().numpy(), a_B.cpu().numpy())

            # env step
            obs, reward, term, trunc, info = env.step(actions)
            done = term or trunc

            if self.cfg.get('render', False):
                env.render()

            next_A, next_B = helpers.obs_to_state(obs, self.n_obs, self.dev)
            transition_A = Transition(ego_s, a_A, next_A, torch.tensor(reward, device=self.dev))
            transition_B = Transition(npc_s, a_B, next_B, torch.tensor(reward, device=self.dev))
            ego_s = self.A.update(transition_A, term)
            npc_s = self.B.update(transition_B, term)

            self.A.set_state(ego_s, npc_s)
            self.B.set_state(ego_s, npc_s)


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
            if pbar:
                pbar.update(1)

        if self.cfg.get('track', False):
            log_stats(info, stats, ego=(train_player=="A"))
            helpers.reset_stats(stats)

        if self.cfg.get('save_trajectories', False):
            self.A.store.end_episode()


        env.close()
        return steps
