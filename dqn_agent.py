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

import matplotlib.pyplot as plt


class TrajectoryStore(object):
    def __init__(self, num_envs: int, episode_interval : int = 5000, file_dir : str = 'trajectories'):
        self.num_envs = num_envs
        self.trajectories = {}
        for i in range(num_envs):
            self.trajectories[i] = None

        self.episode_interval = episode_interval
        self.file_interval = 0
        
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        self.file_dir = file_dir
        self.crash_trajectories = {}

    def add(self, worker_id: int, transition: Transition, int_frames: np.ndarray):



        state_history = np.vstack((transition.state, int_frames[:-1,:]))

        # TTC Calculation
        dx = state_history[:,6]
        dy = state_history[:,7]
        dvx = state_history[:,8]
        dvy = state_history[:,9]
        ttc_x = np.where(np.abs(dvx) > 1e-6, dx/dvx, dx/1e-6)
        ttc_y = np.where(np.abs(dvy) > 1e-6, dy/dvy, dy/1e-6)

        # Populate Actions
        action_array = np.zeros(state_history.shape[0], dtype=int)
        action_array[0] = transition.action
        action_array[1:] = -1

        # Populate Rewards
        reward_array = np.zeros(state_history.shape[0], dtype=float)
        reward_array[0] = transition.reward
        reward_array[1:] = 0.0

        save_data = np.column_stack((state_history, action_array, reward_array, ttc_x, ttc_y))

        if self.trajectories[worker_id] is None:
            self.trajectories[worker_id] = save_data
        else:
            self.trajectories[worker_id] = np.vstack((self.trajectories[worker_id], save_data))

    def clear(self, worker_id: int):
        self.trajectories[worker_id] = None

    def save(self, worker_id: int, episode_num: int):
        self.crash_trajectories[episode_num] = self.trajectories[worker_id].tolist()
        self.clear(worker_id)

        if len(self.crash_trajectories.keys()) % self.episode_interval == 0:
            file_path = os.path.join(self.file_dir, f'{self.file_interval}')
            self.write(file_path, 'json')
            self.file_interval += 1
            self.crash_trajectories = {}

    def trajectory_to_dict(self, trajectory) -> dict:
        trajectory_dict = {}
        for episode, transition in enumerate(trajectory):
            trajectory_dict[episode] = self.transition_to_dict(transition)

        return trajectory_dict

    def transition_to_dict(self, transition: Transition) -> dict:
        if transition.next_state is None:
            return {
                "state": transition.state.tolist(),
                "action": int(transition.action),
                "next_state": None,
                "reward": float(transition.reward)
            }
        else:
            return {
                "state": transition.state.tolist(),
                "action": int(transition.action),
                "next_state": transition.next_state.tolist(),
                "reward": float(transition.reward)
            }

    def write(self, path: str, type: str):
        path_name = path + '.' + type
        if type == "csv":
            with open(path_name, 'w') as f:
                for episode in self.crash_trajectories.keys():
                    f.write(f'Episode: {episode}\n')
                    for transition in self.crash_trajectories[episode]:
                        f.write(f'{transition.state},{transition.action},{transition.next_state},{transition.reward}\n')

        elif type == "json":
            with open(path_name, 'w') as f:

                json.dump(self.crash_trajectories, f, indent = 6)

        print(f'Collision Trajectories saved to {path_name}')
        

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions, hidden_layer = 128):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, hidden_layer)
        self.layer2 = nn.Linear(hidden_layer, hidden_layer)
        self.layer3 = nn.Linear(hidden_layer, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    
class DQN_Agent(object):

    def __init__(self, env, args, device = 'cpu', save_trajectories = False, multi_agent = False, trajectory_path = 'trajectories', cycle = 0):

        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer

        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.start_e = args.start_e
        self.end_e = args.end_e
        self.decay_e = args.decay_e
        self.tau = args.tau
        self.lr = args.learning_rate

        self.num_envs = args.num_envs
        self.device = device
        self.track = args.track
        self.multi_agent = multi_agent

        self.cycle = cycle
        
        if self.num_envs > 1:
            if self.multi_agent:
                self.n_actions = env.single_action_space[0].n
                state, _ = env.reset()
                self.n_observations = len(state[0][0].flatten())
            else:
                self.n_actions = env.single_action_space.n
                state, _ = env.reset()
                print(state[0])
                self.n_observations = len(state[0].flatten())
        elif self.num_envs == 1:
            if self.multi_agent:
                self.n_actions = env.single_action_space[0].n
                state, _ = env.reset()
                self.n_observations = len(state[0].flatten())
            else:
                self.n_actions = env.single_action_space.n
                state, _ = env.reset()
                self.n_observations = len(state[0].flatten())

        self.policy_net = DQN(self.n_observations, self.n_actions, hidden_layer=args.hidden_layer).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions, hidden_layer=args.hidden_layer).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        if args.buffer_type == "ER":
            self.memory = ReplayMemory(args.buffer_size)
        elif args.buffer_type == "PER":
            self.memory = PrioritizedExperienceReplay(args.buffer_size)

        self.save_trajectories = save_trajectories
        if save_trajectories:
            self.trajectory_store = TrajectoryStore(self.num_envs, episode_interval = 1000, file_dir = trajectory_path)


    def select_action(self, state, env, steps_done):

        sample = random.random()
        eps_threshold = self.end_e + (self.start_e - self.end_e) * \
            math.exp(-1. * steps_done / self.decay_e)
        if sample > eps_threshold:
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # return self.policy_net(state).max(1)[1].view(self.num_envs, 1)
            return self.predict(state)
        else:
            # print('Action Space: {}'.format(torch.tensor(np.array([[env.action_space.sample()]]), device=device, dtype=torch.long).shape))
            if self.multi_agent:
                return torch.tensor(np.array([[env.action_space[0].sample()]]), device=self.device, dtype=torch.long)
            else:
                return torch.tensor(np.array([[env.action_space.sample()]]), device=self.device, dtype=torch.long)
        
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        if isinstance(self.memory, PrioritizedExperienceReplay):
            idxs, transitions, is_weights = self.memory.sample(self.batch_size)
        else:
            transitions = self.memory.sample(self.batch_size)
        
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values.unsqueeze(1) * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)
        
        # update priorities
        if isinstance(self.memory, PrioritizedExperienceReplay):
            errors = (state_action_values - expected_state_action_values).detach().cpu().squeeze().tolist()
            self.memory.update(idxs, errors)
        
            loss = (torch.FloatTensor(is_weights).to(self.device) * loss).mean()


        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    @torch.no_grad
    def predict(self, state):
        if self.num_envs == 1:
            return torch.argmax(self.policy_net(state))
        else:
            return self.policy_net(state).max(1)[1].view(self.num_envs, 1)
        
    def update(self, state, action, observation, reward, terminated):

        next_state = torch.tensor(observation.reshape(self.num_envs,self.n_observations), dtype=torch.float32, device=self.device)

        for worker in range(self.num_envs):
            if terminated[worker]:
                self.memory.push(state[worker].view(1,self.n_observations),action[worker].view(1,1),None,reward[worker].view(1,1))
            else:
                self.memory.push(state[worker].view(1,self.n_observations),action[worker].view(1,1),next_state[worker].view(1,self.n_observations),reward[worker].view(1,1))

            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target_net.load_state_dict(target_net_state_dict)

        return state

    def learn(self, env, total_timesteps, use_pbar = True):
        if use_pbar:
            pbar = tqdm(total=total_timesteps)
        else:
            pbar = None

        # Get the number of state observations
        state, info = env.reset()
        state = torch.tensor(state.reshape(self.num_envs,self.n_observations), dtype=torch.float32, device=self.device)

        num_crashes = []
        episode_rewards = np.zeros(self.num_envs)
        duration = np.zeros(self.num_envs)
        episode_speed = np.zeros(self.num_envs)
        ep_rew_total = np.zeros(0)
        ep_len_total = np.zeros(0)
        ep_speed_total = np.zeros(0)

        t_step = 0
        ep_num = 0

        while(True):
            action = torch.squeeze(self.select_action(state, env, t_step))
            if self.num_envs == 1:
                action = action.view(1,1)
            observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            reward = torch.tensor(reward, dtype = torch.float32, device=self.device)
            done = terminated | truncated

            if done:
                int_frames = info['final_info'][0]['int_frames']
            else:
                int_frames = info['int_frames'][0]

            if self.save_trajectories:
                for worker in range(self.num_envs):
                    if terminated[worker]:
                        self.trajectory_store.add(worker, Transition(state[worker].cpu().numpy(), action[worker].cpu().numpy(), None, reward[worker].cpu().numpy()), int_frames)
                    else:
                        self.trajectory_store.add(worker, Transition(state[worker].cpu().numpy(), action[worker].cpu().numpy(), observation[worker].flatten(), reward[worker].cpu().numpy()), int_frames)

            state = self.update(state, action, observation, reward, terminated)
            episode_rewards = episode_rewards + reward.cpu().numpy()
            duration = duration + np.ones(self.num_envs)
            episode_speed = episode_speed + np.linalg.norm(state[:,3:5].cpu().numpy(), axis=1)

            for worker in range(self.num_envs):
                if done[worker]:
                    # Save Trajectories that end in a Crash
                    if self.save_trajectories:
                        # if info['final_info'][worker]['crashed']:
                        #     self.trajectory_store.save(worker, ep_num)
                        # else:
                        #     self.trajectory_store.clear(worker)
                        self.trajectory_store.save(worker, ep_num)


                    num_crashes.append(float(info['final_info'][worker]['crashed']))
                    if self.track:
                        ep_rew_total = np.append(ep_rew_total, episode_rewards[worker])
                        ep_len_total = np.append(ep_len_total, duration[worker])
                        ep_speed_total = np.append(ep_speed_total, episode_speed[worker]/duration[worker])
                        if ep_rew_total.size > 100:
                            ep_rew_total = np.delete(ep_rew_total, 0)
                        if ep_len_total.size > 100:
                            ep_len_total = np.delete(ep_len_total, 0)
                        if ep_speed_total.size > 100:
                            ep_speed_total = np.delete(ep_speed_total, 0)

                        wandb.log({"rollout/ep_rew_mean": ep_rew_total.mean(),
                                "rollout/ep_len_mean": ep_len_total.mean(),
                                "rollout/num_crashes": num_crashes[-1],
                                "rollout/num_crashes_mean": np.mean(num_crashes),
                                "rollout/ep_speed_mean": ep_speed_total.mean()},
                                step = ep_num)

                    episode_rewards[worker] = 0
                    duration[worker] = 0
                    episode_speed[worker] = 0
                    ep_num += 1

                t_step += 1
                pbar.update(1)

                if t_step >= total_timesteps:
                    pbar.close()
                    return
                
    def test(self, env, total_timesteps, use_pbar = True):
        if use_pbar:
            pbar = tqdm(total=total_timesteps)
        else:
            pbar = None

        # Get the number of state observations
        state, info = env.reset()
        state = torch.tensor(state.reshape(self.num_envs,self.n_observations), dtype=torch.float32, device=self.device)

        num_crashes = []
        episode_rewards = np.zeros(self.num_envs)
        duration = np.zeros(self.num_envs)
        episode_speed = np.zeros(self.num_envs)
        ep_rew_mean = np.zeros(0)
        ep_len_mean = np.zeros(0)
        ep_speed_mean = np.zeros(0)

        t_step = 0
        ep_num = 0

        while(True):
            action = torch.squeeze(self.predict(state))
            observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
            reward = torch.tensor(reward, dtype = torch.float32, device=self.device)
            done = terminated | truncated

            state = torch.tensor(observation.reshape(self.num_envs,self.n_observations), dtype=torch.float32, device=self.device)

            episode_rewards = episode_rewards + reward.cpu().numpy()
            duration = duration + np.ones(self.num_envs)

            for worker in range(self.num_envs):
                if done[worker]:
                    num_crashes.append(float(info['final_info'][worker]['crashed']))
                    if self.track:
                        ep_rew_mean = np.append(ep_rew_mean, episode_rewards[worker])
                        ep_len_mean = np.append(ep_len_mean, duration[worker])
                        if ep_rew_mean.size > 100:
                            ep_rew_mean = np.delete(ep_rew_mean, 0)
                        if ep_len_mean.size > 100:
                            ep_len_mean = np.delete(ep_len_mean, 0)

                        wandb.log({"rollout/ep_rew_mean": ep_rew_mean.mean(),
                                "rollout/ep_len_mean": ep_len_mean.mean(),
                                "rollout/num_crashes": num_crashes[-1],
                                "rollout/num_crashes_mean": np.mean(num_crashes)},
                                step = ep_num)

                    episode_rewards[worker] = 0
                    duration[worker] = 0
                    ep_num += 1

                t_step += 1
                pbar.update(1)

                if t_step >= total_timesteps:
                    pbar.close()
                    print(f'Average # of Crashes: {np.mean(num_crashes)}')
                    return

    def save_model(self, path = 'model.pth'):
        torch.save(self.policy_net.state_dict(), path)
        print(f'Model Saved to {path}')

    def load_model(self, path):
        try:
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f'Model Loaded from {path}')
        except:
            print(f'Failed to Load Model from {path}')