import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dqn_crasher.buffers.experience_replay import (PrioritizedExperienceReplay,
                                                   ReplayMemory, Transition)


class DQN(nn.Module):
    def __init__(
        self, n_observations, n_actions, num_hidden_layers=1, hidden_layer=128
    ):
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
    def __init__(
        self,
        n_state,
        n_action,
        action_space,
        config,
        device="cpu",
        trajectory_path="trajectories",
        cycle=0,
        ego_or_npc="EGO",
        override_obs=-1,
    ):
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer

        # Set Seed
        random.seed(config.get("seed", 42))

        self.batch_size = config.get("batch_size", 32)
        self.gamma = config.get("gamma", 0.8)  # discount factor
        self.start_e = config.get("start_e", 1.0)  # initial epsilon
        self.end_e = config.get("end_e", 0.05)  # final epsilon
        self.decay_e = config.get("decay_e", 6000)  # decay period for epsilon
        self.tau = config.get("tau", 0.005)  # target network update rate
        self.lr = config.get("learning_rate", 5e-4)  # learning rate for the optimizer

        self.device = device
        self.track = config.get(
            "track", False
        )  # whether to track the training progress
        self.eps_threshold = 1.0

        self.cycle = cycle

        self.n_actions = n_action
        self.n_observations = n_state
        self.action_space = action_space

        if override_obs != -1:
            self.n_observations = override_obs

        self.policy_net = torch.compile(
            DQN(
                self.n_observations,
                self.n_actions,
                num_hidden_layers=config.get("num_hidden_layers", 1),
                hidden_layer=config.get("hidden_layer", 256),
            ).to(device)
        )
        self.target_net = torch.compile(
            DQN(
                self.n_observations,
                self.n_actions,
                num_hidden_layers=config.get("num_hidden_layers", 1),
                hidden_layer=config.get("hidden_layer", 256),
            ).to(device)
        )

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)

        if config.get("buffer_type", "ER") == "ER":
            self.memory = ReplayMemory(config.get("buffer_size", 15000))
        elif config.get("buffer_type", "ER") == "PER":
            self.memory = PrioritizedExperienceReplay(config.get("buffer_size", 15000))

    def select_action(self, state, steps_done):
        sample = random.random()
        self.eps_threshold = self.end_e + (self.start_e - self.end_e) * math.exp(
            -1.0 * steps_done / self.decay_e
        )
        if sample > self.eps_threshold:
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            action_logit = self.policy_net(state)
            return action_logit
        else:
            sampled_action = self.action_space.sample()
            sampled_action_tensor = torch.tensor(
                np.zeros((1, self.action_space.n)), device=self.device, dtype=torch.long
            )
            sampled_action_tensor[0, int(sampled_action)] = 1.0
            return sampled_action_tensor

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
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

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
            # Select Actions for Target net based on Value Net ( better action )
            _, action_prime = self.policy_net(non_final_next_states).max(1)
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states)
                .gather(1, action_prime.unsqueeze(1))
                .squeeze()
            )
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values.unsqueeze(1) * self.gamma
        ) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values)

        # update priorities
        if isinstance(self.memory, PrioritizedExperienceReplay):
            errors = (
                (state_action_values - expected_state_action_values)
                .detach()
                .cpu()
                .squeeze()
                .tolist()
            )
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
        return self.policy_net(state)

    def update(self, state, action, next_state, reward, terminated):
        if terminated:
            self.memory.push(state, action.view(1, 1), None, reward.view(1, 1))
        else:
            self.memory.push(state, action.view(1, 1), next_state, reward.view(1, 1))

        state = next_state

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return state

    def save_model(self, path="model.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model Saved to {path}")

    def load_model(self, path):
        try:
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
            print(f"Model Loaded from {path}")
        except:
            print(f"Failed to Load Model from {path}")
