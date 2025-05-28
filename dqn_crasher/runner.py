# runner.py
from logging import raiseExceptions
import os
import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
import copy
import helpers
import wandb
from wandb_logging import initialize_logging, log_stats, log_checkpoint_summary
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

        # Create global dictionary to track specific policy stats
        if 'specific_policy_stats' not in globals():
            globals()['specific_policy_stats'] = {}

        # a single env instance just to pull action_space / obs dims
        tmp = gym.make(env_name, config=gym_cfg)
        self.action_space = tmp.action_space[0]
        self.n_actions = self.action_space.n
        self.n_obs = 10 * config.get('frame_stack', 1)
        tmp.close()


    def save_model(self, train_player, step_number: int = 0):
        if train_player == "A" and type(self.A.agent) == DQN_Agent:
            self.A.save_model(os.path.join(self.cfg.get('root_directory', '') ,self.cfg.get('model_save_path', 'models/model') + f'/model_{step_number}.pth'))
        elif type(self.B.agent) == DQN_Agent:
            self.B.save_model(os.path.join(self.cfg.get('root_directory', '') ,self.cfg.get('model_save_path', 'models/model') + f'/model_{step_number}.pth'))

    def train(self, train_player="A"):
        total_timesteps = self.cfg.get('total_timesteps', 100000)
        test_interval = self.cfg.get('test_interval', 10000)  # Run tests every 10k steps by default
        t     = 0
        pbar  = tqdm(total=total_timesteps)

        if self.cfg.get('track', False):
            initialize_logging(self.cfg, train_ego=(train_player=="A"))
            stats = helpers.initialize_stats()

        if self.cfg.get('save_trajectories', False):
            self.A.store.save_metadata(self.cfg)

        while t < total_timesteps:
            self.A.reset()
            self.B.reset()
            self._set_config()
            # Use t as the training step counter
            stats['training_step'] = t
            steps = self._run_episode(train_player=train_player, t_start=t, stats=stats, pbar=pbar, checkpoint_step=None)
            t += steps

            # Run incremental testing at specified intervals
            if self.cfg.get('enable_checkpoint_testing', True) and t > 0 and t % test_interval < steps:
                # Use the current step as the checkpoint identifier
                self.test_checkpoint(checkpoint_step=t)
                self.save_model(train_player, t)

        self.save_model(train_player, t)

        if pbar: pbar.close()
        wandb.finish()

    def test(self):
        total_eps = self.cfg.get('testing_episodes', 10)
        eps_done  = 0

        if self.cfg.get('track', False):
            initialize_logging(self.cfg, train_ego=False, eval=True)
            stats = helpers.initialize_stats()
            stats['metrics_type'] = 'testing'

        if self.cfg.get('save_trajectories', False):
            self.A.store.save_metadata(self.cfg)

        scenario_vs_mobil = isinstance(self.A.policies[0], policies.ScenarioPolicy)
        if scenario_vs_mobil:
            policy_iterate = self.A.policies
        else:
            policy_iterate = self.B.policies


        pbar      = tqdm(total=total_eps * len(policy_iterate))

        policy_stats = {}

        for i in range(len(policy_iterate)):
            policy_name = str(type(policy_iterate[i]).__name__)
            if policy_name not in policy_stats:
                policy_stats[policy_name] = helpers.initialize_stats()

            for j in range(total_eps):
                if scenario_vs_mobil:
                    self.A.reset(i, test = True)
                    self.B.reset(test = True)
                else:
                    self.A.reset(test = True)
                    self.B.reset(i, test = True)
                self._set_config()
                _ = self._run_episode(train_player=None, t_start=0, stats=stats,
                                     policy_stats=policy_stats, policy_name=policy_name, pbar=None)
                eps_done += 1

                if pbar is not None:
                    pbar.update(1)

        if self.cfg.get('track', False):
            # Log policy-specific statistics
            for policy_name, policy_stat in policy_stats.items():
                log_stats(None, policy_stat, ego=False, policy_prefix=policy_name)

        if pbar: pbar.close()
        wandb.finish()

    def test_checkpoint(self, checkpoint_step):
        """Run a smaller test during training to track progress"""
        total_eps = self.cfg.get('checkpoint_testing_episodes', 5)  # Fewer episodes for checkpoints

        # Save current state to restore after testing
        save_state = (self.A.save_state(), self.B.save_state())

        # Track is already True if we're in training
        stats = helpers.initialize_stats()
        # Mark these stats as belonging to checkpoint testing
        stats['checkpoint_testing'] = True
        stats['metrics_type'] = 'checkpoint_episode'

        # Create containers to store all episode stats for later aggregation
        all_episode_stats = []
        policy_episode_stats = {}
        specific_policy_episode_stats = {}
        policy_stats = {}

        scenario_vs_mobil = False
        if isinstance(self.A, policies.PolicyDistribution):
            scenario_vs_mobil = isinstance(self.A.policies[0], policies.ScenarioPolicy)

        if scenario_vs_mobil:
            policy_iterate = self.A.policies
        else:
            policy_iterate = self.B.policies


        for i in range(len(policy_iterate)):
            policy_name = str(type(policy_iterate[i]).__name__)
            if policy_name not in policy_stats:
                policy_stats[policy_name] = helpers.initialize_stats()

            if policy_name not in policy_episode_stats:
                policy_episode_stats[policy_name] = []

            for j in range(total_eps):
                if scenario_vs_mobil:
                    self.A.reset(i, test = True)
                    self.B.reset(test = True)
                else:
                    self.A.reset(test = True)
                    self.B.reset(i, test = True)
                self._set_config()

                # Run the episode
                # Use a unique offset for each test episode
                unique_offset = j + (i * total_eps)
                testing_step = unique_offset + 1  # Start from 1

                # Set counters for different metric types
                stats['episode_num'] = testing_step
                stats['testing_step'] = testing_step
                stats['metrics_type'] = 'checkpoint_episode'

                if policy_name in policy_stats:
                    policy_stats[policy_name]['episode_num'] = testing_step
                    policy_stats[policy_name]['testing_step'] = testing_step
                    policy_stats[policy_name]['metrics_type'] = 'checkpoint_episode'

                episode_results = self._run_episode(train_player=None, t_start=0, stats=stats,
                                        policy_stats=policy_stats, policy_name=policy_name, pbar=None, checkpoint_step=testing_step)

                # Save a copy of this episode's stats for later aggregation
                if self.cfg.get('track', False):
                    # Make a true deep copy of the current stats
                    episode_stats_copy = copy.deepcopy(stats)

                    # Store the copy for later aggregation
                    all_episode_stats.append(episode_stats_copy)

                    # Also collect policy-specific stats
                    if policy_name in policy_stats:
                        policy_stats_copy = copy.deepcopy(policy_stats[policy_name])
                        policy_episode_stats[policy_name].append(policy_stats_copy)

                    # Collect specific policy stats
                    if 'specific_policy_name' in stats and stats['specific_policy_name']:
                        specific_policy = stats['specific_policy_name']

                        # Get specific policy stats
                        if 'specific_policy_stats' in globals() and specific_policy in globals()['specific_policy_stats']:
                            specific_stats_copy = copy.deepcopy(globals()['specific_policy_stats'][specific_policy])

                            # Store in the appropriate collection
                            if specific_policy not in specific_policy_episode_stats:
                                specific_policy_episode_stats[specific_policy] = []

                            specific_policy_episode_stats[specific_policy].append(specific_stats_copy)


        if self.cfg.get('track', False):
            # Log overall checkpoint statistics
            # Use checkpoint step for summary metrics
            summary_counter = checkpoint_step // self.cfg.get('test_interval', 10000)

            # Log overall checkpoint statistics
            stats['checkpoint_step'] = summary_counter
            stats['metrics_type'] = 'checkpoint_summary'
            log_stats(None, stats, checkpoint=True, checkpoint_step=summary_counter)

            # Log policy-specific statistics at checkpoint
            for policy_name, policy_stat in policy_stats.items():
                policy_stat['checkpoint_step'] = summary_counter
                policy_stat['metrics_type'] = 'checkpoint_summary'
                log_stats(None, policy_stat, ego=False, policy_prefix=policy_name,
                          checkpoint=True, checkpoint_step=summary_counter)

            # Aggregate and log summary statistics across all episodes in this checkpoint
            checkpoint_stats = helpers.aggregate_checkpoint_stats(all_episode_stats, checkpoint_step=summary_counter)
            checkpoint_stats['metrics_type'] = 'checkpoint_summary'

            # Aggregate policy-specific stats
            checkpoint_policy_stats = {}
            for policy_name, episodes in policy_episode_stats.items():
                # Each policy gets the same summary step to align them in WandB charts
                checkpoint_policy_stats[policy_name] = helpers.aggregate_checkpoint_stats(episodes, checkpoint_step=summary_counter)
                checkpoint_policy_stats[policy_name]['metrics_type'] = 'checkpoint_summary'

            # Aggregate specific policy stats
            specific_policy_checkpoint_stats = {}
            for specific_policy, episodes in specific_policy_episode_stats.items():
                if episodes:  # Only aggregate if we have episodes
                    specific_policy_checkpoint_stats[specific_policy] = helpers.aggregate_checkpoint_stats(episodes, checkpoint_step=summary_counter)
                    specific_policy_checkpoint_stats[specific_policy]['metrics_type'] = 'checkpoint_summary'

                    # Log the specific policy summary
                    log_stats(None, specific_policy_checkpoint_stats[specific_policy],
                              ego=False, policy_prefix=f"specific/{specific_policy}",
                              checkpoint_summary=True, checkpoint_step=summary_counter)

            log_checkpoint_summary(checkpoint_stats, checkpoint_policy_stats,
                                  checkpoint_step=summary_counter, ego=False,
                                  specific_policy_stats=specific_policy_checkpoint_stats)

        self.A.restore_state(save_state[0])
        self.B.restore_state(save_state[1])

    def _set_config(self):
        def unwrap(policy_obj):
            # if it's a distribution, grab the inner scenario policy
            return policy_obj.current_policy if isinstance(policy_obj, policies.PolicyDistribution) else policy_obj

        scenario_policy = False
        mobil_policy = False
        for name in ('A', 'B'):
            policy = unwrap(getattr(self, name))
            # ScenarioPolicy always gets configured
            if isinstance(policy, policies.ScenarioPolicy):
                policy.set_config(self.cfg)
                scenario_policy = True
            # If there is no Scenario Policy use other policy
            elif isinstance(policy, policies.MobilPolicy):
                mobil_policy = True
                if not scenario_policy:
                    policy.set_config(self.cfg)

        if mobil_policy is False:
            self.cfg['gym_config']['vs_mobil'] = False
            self.cfg['gym_config']['use_mobil'] = False
            self.cfg['gym_config']['controlled_vehicles'] = 2
            self.cfg['gym_config']['other_vehicles'] = 0
        else:
            self.cfg['gym_config']['vs_mobil'] = True
            self.cfg['gym_config']['use_mobil'] = True
            self.cfg['gym_config']['controlled_vehicles'] = 1
            self.cfg['gym_config']['other_vehicles'] = 1

        self.gym_cfg = self.cfg['gym_config']

    def _run_episode(self, train_player, t_start, stats, pbar, policy_stats=None, policy_name=None, checkpoint_step=None):
        total_timesteps = self.cfg.get('total_timesteps', 100000)

        env  = gym.make(self.env_name, config=self.gym_cfg, render_mode="rgb_array")
        obs, info = env.reset()
        ego_s, npc_s = helpers.obs_to_state(obs, self.n_obs, self.dev, frame_stack = self.cfg['frame_stack'])

        self.A.set_state(ego_s, npc_s)
        self.B.set_state(npc_s, ego_s)

        # Classify type of scenario
        classification_source = None
        scenario = None
        if train_player:
            if isinstance(self.B, policies.PolicyDistribution):
                classification_source = self.B
        elif isinstance(self.A, policies.PolicyDistribution):
            if isinstance(self.A.current_policy, policies.ScenarioPolicy):
                classification_source = self.A

        if isinstance(classification_source, policies.PolicyDistribution):
            if isinstance(classification_source.current_policy, policies.ScenarioPolicy):
                scenario = str(type(classification_source.current_policy.scenario).__name__)
            elif isinstance(classification_source.current_policy, policies.MobilPolicy):
                scenario = str(type(classification_source.current_policy).__name__) + '.' + classification_source.current_policy.spawn_configs[0]


        if self.cfg.get('save_trajectories', False):
            if train_player:
                self.A.store.start_episode(stats['episode_num'])
                self.B.store.start_episode(stats['episode_num'])
            else:
                self.A.test_store.start_episode(stats['episode_num'])
                self.B.test_store.start_episode(stats['episode_num'])

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

            next_A, next_B = helpers.obs_to_state(obs, self.n_obs, self.dev, frame_stack = self.cfg['frame_stack'])
            transition_A = Transition(ego_s, a_A, next_A, torch.tensor(reward, device=self.dev))
            transition_B = Transition(npc_s, a_B, next_B, torch.tensor(reward, device=self.dev))
            ego_s = self.A.update(transition_A, term)
            npc_s = self.B.update(transition_B, term)

            self.A.set_state(ego_s, npc_s)
            self.B.set_state(npc_s, ego_s)


            # store trajectories for the  policy
            if self.cfg.get('save_trajectories'):
                if train_player:
                    self.A.store.add(transition_A)
                    self.B.store.add(transition_B)
                else:
                    self.A.test_store.add(transition_A)
                    self.B.test_store.add(transition_B)

            if train_player:
                info['eps_threshold'] = self.A.agent.eps_threshold
            else:
                info['eps_threshold'] = 0.0

            info['rewards']['total'] = reward
            info['ego_speed'] = (ego_s[0,3].cpu().numpy()**2 + ego_s[0,4].cpu().numpy()**2)**0.5
            info['npc_speed'] = (npc_s[0,3].cpu().numpy()**2 + npc_s[0,4].cpu().numpy()**2)**0.5
            info['scenario'] = scenario

            # Populate the main stats dictionary
            helpers.populate_stats(info,episode_statistics=stats)

            t += 1
            steps += 1
            if pbar:
                pbar.update(1)

        if self.cfg.get('track', False):
            # Determine which metric type we're using based on context flags
            metrics_type = stats.get('metrics_type', 'training')

            if metrics_type == 'checkpoint_episode':
                # For checkpoint testing episodes - use testing_step
                current_step = stats.get('testing_step', 0)
                log_stats(info, stats, checkpoint=True, checkpoint_step=current_step)

                # Reset stats but preserve all counters and metadata
                helpers.reset_stats(stats, preserve_episode_num=True)
            elif metrics_type == 'checkpoint_summary':
                # For checkpoint summary metrics - use checkpoint_step
                current_step = stats.get('checkpoint_step', 0)
                log_stats(info, stats, checkpoint_summary=True, checkpoint_step=current_step)
                helpers.reset_stats(stats, preserve_episode_num=True)
            elif metrics_type == 'testing':
                # For evaluation testing - use normal episode numbers
                log_stats(info, stats, checkpoint=True)
                helpers.reset_stats(stats, preserve_episode_num=True)
            else:
                # Normal training logging - use training_step
                current_step = stats.get('training_step', 0)
                log_stats(info, stats)
                helpers.reset_stats(stats, preserve_episode_num=False)


        if self.cfg.get('save_trajectories', False):
            if train_player:
                self.A.store.end_episode()
                self.B.store.end_episode()
            else:
                self.A.test_store.end_episode()
                self.B.test_store.end_episode()


        env.close()
        return steps
