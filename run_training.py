import wandb
import gymnasium as gym
from gymnasium import logger
from gymnasium.wrappers.record_video import RecordVideo
import math
import random
import matplotlib
from matplotlib import pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", message="overflow encountered in exp")
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

import argparse
from dqn import DQN

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



parser = argparse.ArgumentParser(description='Crash DQN')
# GENERIC RL/MODEL PARAMETERS
parser.add_argument('--lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--env-name', type=str, default='crash-v0',
                    help='gym environment name')
parser.add_argument('--num-workers', type=int, default=1,
                    help='number of parallel environments to run')
parser.add_argument('--max-steps', type=int, default=int(1e8),
                    help='maximum number of training steps in total')
parser.add_argument('--cuda', type=bool, default=True,
                    help='Add cuda')

# SPECIFIC DQN PARAMETERS  
parser.add_argument('--batch-size', type=int, default=128,
                    help='Manager horizon (c)')
parser.add_argument('--hidden-dim', type=int, default=128,
                    help='Hidden dim (d)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help="discount factor")
parser.add_argument('--eps_start', type=float, default=0.9,
                    help='Random Gausian goal for exploration')
parser.add_argument('--eps_end', type=float, default=0.05,
                    help='Random Gausian goal for exploration')
parser.add_argument('--eps_decay', type=int, default=1000,
                    help='Random Gausian goal for exploration')
parser.add_argument('--tau', type=float, default=0.005,
                    help='Rate at which to update target network')
parser.add_argument('--replay-buffer', type=int, default=10000,
                    help='Size of replay memory buffer')

# CRASH PARAMETERS

parser.add_argument('--collision-coefficient', type=int, default=400,
                    help='reward for collision')
parser.add_argument('--ttc-x-coefficient', type=int, default=4,
                    help='coefficient for ttx-x reward')
parser.add_argument('--ttc-y-coefficient', type=int, default=1,
                    help='coefficient for ttx-y reward')

parser.add_argument('--seed', type=int, default=0,
                    help='reproducibility seed.')

args = parser.parse_args()

wlog = True
save_model = False
record_video = False
save_every = 1000

max_steps = args.max_steps
collision_coefficient = args.collision_coefficient
ttc_x_coefficient = args.ttc_x_coefficient
ttc_y_coefficient = args.ttc_y_coefficient
num_workers = args.num_workers

spawn_configs =  ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center']
num_configs = 8
# spawn_configs =  ['forward_left', 'forward_right', 'forward_center']
# num_configs = 3

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
EPS_START = args.eps_start
EPS_END = args.eps_end
EPS_DECAY = args.eps_decay
TAU = args.tau
LR = args.lr
ReplayBuffer = args.replay_buffer

hidden_dim = args.hidden_dim

if wlog:
    wandb.init(
        # set the wandb project where this run will be logged
        project="rl-crash-course",
        
        # track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "architecture": "DQN",
        "max_duration": 100,
        "dataset": "Highway-Env",
        "max_steps": max_steps,
        "collision_reward": collision_coefficient,
        "ttc_x_reward": ttc_x_coefficient,
        "ttc_y_reward": ttc_y_coefficient,
        "num_configs": num_configs,
        "BATCH_SIZE": BATCH_SIZE,
        "GAMMA": GAMMA,
        "EPS_START": EPS_START,
        "EPS_END": EPS_END,
        "EPS_DECAY": EPS_DECAY,
        "TAU": TAU,
        "hidden_dim": hidden_dim,
        "ReplayBuffer": ReplayBuffer
        }
    )

    logging.basicConfig(filename=wandb.run.dir + '/episode_rollouts.log', level=logging.INFO)

env_config = {
    "observation": {
        "type": "Kinematics",
        "normalize": False
    },
    "action": {
        "type": "DiscreteMetaAction",
        "target_speeds": list(range(15,35))
    },
    "lanes_count" : 2,
    "vehicles_count" : 1,
    "duration" : 100,
    "initial_lane_id" : None,
    "mean_distance": 20,
    "mean_delta_v": 0,
    "policy_frequency": 1,
    "collision_reward": collision_coefficient,    # The reward received when colliding with a vehicle.
    "ttc_x_reward": ttc_x_coefficient,  # The reward range for time to collision in the x direction with the ego vehicle.
    "ttc_y_reward": ttc_y_coefficient,  # The reward range for time to collision in the y direction with the ego vehicle.
    "spawn_configs": spawn_configs[:num_configs]
}

if record_video:
    env = gym.vector.AsyncVectorEnv([
        lambda: RecordVideo(gym.make('crash-v0', config = env_config, render_mode='rgb_array'), \
                                        video_folder = './video', \
                                        episode_trigger=lambda e: e % 100 == 0),
        lambda: RecordVideo(gym.make('crash-v0', config = env_config, render_mode='rgb_array'), \
                                        video_folder = './video', \
                                        episode_trigger=lambda e: e % 100 == 0)
    ])
else:
    env = gym.vector.make('crash-v0', num_envs = num_workers, config = env_config, render_mode='rgb_array')

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

device = torch.device("cuda" if (torch.cuda.is_available() and args.cuda)  else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



# Get number of actions from gym action space
n_actions = env.action_space.nvec[0]
# Get the number of state observations
state, info = env.reset()
n_observations = len(state[0].flatten())

policy_net = DQN(n_observations, n_actions, hidden_dim).to(device)
target_net = DQN(n_observations, n_actions, hidden_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(ReplayBuffer)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(num_workers, 1)
    else:
        # print('Action Space: {}'.format(torch.tensor(np.array([[env.action_space.sample()]]), device=device, dtype=torch.long).shape))
        return torch.tensor(np.array([[env.action_space.sample()]]), device=device, dtype=torch.long)


episode_durations = []
episode_rewards = []


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                        batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values.unsqueeze(1) * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# if torch.cuda.is_available():
#     num_episodes = 1000
# else:
#     num_episodes = 100
num_crashes = []
i_episode = 0
episode_rewards = np.zeros(num_workers)
duration = np.zeros(num_workers)
ttc_x = 0
ttc_y = 0
state, info = env.reset()
if record_video and (i_episode % 100 == 0):
    env.unwrapped.automatic_rendering_callback = env.video_recorder.capture_frame
state = torch.tensor(state.reshape(num_workers,n_observations), dtype=torch.float32, device=device)

max_steps = 1e7
t_step = 0
with tqdm(total=max_steps) as pbar:
    while(True):
        action = torch.squeeze(select_action(state))
        observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())
        if record_video and (i_episode % 100 == 0):
            env.render()

        reward = torch.tensor(reward, device=device)
        done = terminated | truncated

        next_state = torch.tensor(observation.reshape(num_workers,n_observations), dtype=torch.float32, device=device)
        for worker in range(num_workers):
            if terminated[worker]:
                memory.push(state[worker].view(1,n_observations),action[worker].view(1,1),None,reward[worker].view(1,1))
            else:
                memory.push(state[worker].view(1,n_observations),action[worker].view(1,1),next_state[worker].view(1,n_observations),reward[worker].view(1,1))

        state = next_state

        episode_rewards = episode_rewards + reward.cpu().numpy()
        duration = duration + np.ones(num_workers)

        # Perform one step of the optimization (on the policy network)
        optimize_model()

    # Perform one step of the optimization (on the policy network)
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

    for worker in range(num_workers):
        if done[worker]:
            num_crashes.append(float(info['final_info'][worker]['crashed']))
            i_episode += 1
            if wlog:
                wandb.log({"train/reward": episode_rewards[worker],
                            "train/num_crashes": sum(num_crashes),
                            "train/duration" : duration[worker],
                            "train/sr_100": sum(num_crashes[-100:])/100})
            episode_rewards[worker] = 0
            duration[worker] = 0

        t_step += 1
        pbar.update(1)

    if t_step >= max_steps:
        break

if save_model:
    torch.save(policy_net.state_dict(), wandb.run.dir + "/model-{}.pt".format(len(episode_rewards)))

print('Complete')
env.close()