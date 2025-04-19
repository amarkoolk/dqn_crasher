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
    def __init__(self, episode_interval : int = 5000, file_dir : str = 'trajectories', ego_or_npc = 'EGO'):
        self.trajectories = {}
        self.trajectories = None

        self.episode_interval = episode_interval
        self.file_interval = 0

        self.file_dir = os.path.join(file_dir, ego_or_npc)
        
        if not os.path.exists(self.file_dir):
            os.makedirs(self.file_dir)

        self.crash_trajectories = {}

    def add(self, transition: Transition, int_frames: np.ndarray, ego_pool = None, npc_pool = None):

        state_history = np.vstack((transition.state, int_frames[:-1,:]))

        # TTC Calculation
        epsilon = 1e-6
        dx = state_history[:,6]
        dy = state_history[:,7]
        dvx = state_history[:,8]
        dvy = state_history[:,9]
        ttc_x = np.divide(dx, dvx, out=np.full_like(dx, np.nan), where=(np.abs(dvx) > epsilon))
        ttc_y = np.divide(dy, dvy, out=np.full_like(dy, np.nan), where=(np.abs(dvy) > epsilon))

        small_mask = np.abs(dvx) <= epsilon
        ttc_x[small_mask] = dx[small_mask] / epsilon
        ttc_y[small_mask] = dy[small_mask] / epsilon


        # Populate Actions
        action_array = np.zeros(state_history.shape[0], dtype=int)
        action_array[0] = transition.action
        action_array[1:] = -1

        # Populate Rewards
        reward_array = np.zeros(state_history.shape[0], dtype=float)
        reward_array[0] = transition.reward
        reward_array[1:] = 0.0

        # Save Model Pool Data
        if ego_pool is not None:
            model_idx = ego_pool.model_idx
            ego_pool_data = np.zeros((state_history.shape[0], 5))
            ego_pool_data[0,0] = ego_pool.model_ep_freq[model_idx]
            ego_pool_data[0,1] = ego_pool.model_crash_freq[model_idx]
            ego_pool_data[0,2] = ego_pool.model_sr100[model_idx]
            ego_pool_data[0,3] = ego_pool.model_elo[model_idx]
            ego_pool_data[0,4] = ego_pool.opponent_elo
            ego_pool_data[1:,0] = -1
            ego_pool_data[1:,1] = -1
            ego_pool_data[1:,2] = -1
            ego_pool_data[1:,3] = -1
            ego_pool_data[1:,4] = -1
            state_history = np.column_stack((state_history, ego_pool_data))
        elif npc_pool is not None:
            model_idx = npc_pool.model_idx
            npc_pool_data = np.zeros((state_history.shape[0], 5))
            npc_pool_data[0,0] = npc_pool.model_ep_freq[model_idx]
            npc_pool_data[0,1] = npc_pool.model_crash_freq[model_idx]
            npc_pool_data[0,2] = npc_pool.model_sr100[model_idx]
            npc_pool_data[0,3] = npc_pool.model_elo[model_idx]
            npc_pool_data[0,4] = npc_pool.opponent_elo
            npc_pool_data[1:,0] = -1
            npc_pool_data[1:,1] = -1
            npc_pool_data[1:,2] = -1
            npc_pool_data[1:,3] = -1
            npc_pool_data[1:,4] = -1
            state_history = np.column_stack((state_history, npc_pool_data))

        save_data = np.column_stack((state_history, action_array, reward_array, ttc_x, ttc_y))

        if self.trajectories is None:
            self.trajectories = save_data
        else:
            self.trajectories= np.vstack((self.trajectories, save_data))

    def clear(self):
        self.trajectories = None

    def save(self, episode_num: int):
        self.crash_trajectories[episode_num] = self.trajectories.tolist()
        self.clear()

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

    def __init__(self, n_observations, n_actions, num_hidden_layers = 1, hidden_layer = 128):
        super(DQN, self).__init__()

        layers = []

        input_layer = nn.Linear(n_observations, hidden_layer)

        layers.append(input_layer)
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_layer, hidden_layer))

        self.hidden_layers = nn.ModuleList(layers)

        self.output = nn.Linear(hidden_layer, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output(x)
    
class DQN_Agent(object):

    def __init__(self, n_state, n_action, action_space, config, device = 'cpu', trajectory_path = 'trajectories', cycle = 0, ego_or_npc = 'EGO', override_obs = -1):

        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer

        # Set Seed
        random.seed(config.get('seed', 42))

        self.batch_size = config.get('batch_size', 32)
        self.gamma = config.get('gamma', 0.8)  # discount factor
        self.start_e = config.get('start_e', 1.0)  # initial epsilon
        self.end_e = config.get('end_e', 0.05)  # final epsilon
        self.decay_e = config.get('decay_e', 6000)  # decay period for epsilon
        self.tau = config.get('tau', 0.005)  # target network update rate
        self.lr = config.get('learning_rate', 5e-4)  # learning rate for the optimizer

        self.device = device
        self.track = config.get('track', False)  # whether to track the training progress
        self.eps_threshold = 1.0

        self.cycle = cycle

        self.n_actions = n_action
        self.n_observations = n_state
        self.action_space = action_space

        if override_obs != -1:
            self.n_observations = override_obs

        self.policy_net = DQN(self.n_observations, self.n_actions, num_hidden_layers=config.get('num_hidden_layers', 1), hidden_layer=config.get('hidden_layer', 256)).to(device)
        self.target_net = DQN(self.n_observations, self.n_actions, num_hidden_layers=config.get('num_hidden_layers', 1), hidden_layer=config.get('hidden_layer', 256)).to(device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        if config.get('buffer_type', 'ER') == "ER":
            self.memory = ReplayMemory(config.get('buffer_size', 15000))
        elif config.get('buffer_type', 'ER') == "PER":
            self.memory = PrioritizedExperienceReplay(config.get('buffer_size', 15000))

        if config.get('save_trajectories', False):
            self.trajectory_store = TrajectoryStore(episode_interval = 1000, file_dir = trajectory_path, ego_or_npc = ego_or_npc)


    def select_action(self, state, steps_done):

        sample = random.random()
        self.eps_threshold = self.end_e + (self.start_e - self.end_e) * \
            math.exp(-1. * steps_done / self.decay_e)
        if sample > self.eps_threshold:
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.predict(state)
        else:

            sampled_action = self.action_space.sample()
            return torch.tensor(np.array([[sampled_action]]), device=self.device, dtype=torch.long)
        
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
        return torch.argmax(self.policy_net(state))
        
    def update(self, state, action, next_state, reward, terminated):

        if terminated:
            self.memory.push(state,action.view(1,1),None,reward.view(1,1))
        else:
            self.memory.push(state,action.view(1,1),next_state,reward.view(1,1))

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