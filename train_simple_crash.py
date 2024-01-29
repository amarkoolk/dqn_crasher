import wandb
import gymnasium as gym

import tyro
import math
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from arguments import Args
from crash_wrappers import CrashRewardWrapper, CrashResetWrapper
from buffers import ReplayMemory, PrioritizedExperienceReplay, Transition
from create_env import make_vector_env
from models import DQN

# from itertools import count
# import warnings




if __name__ == "__main__":
    # Parse command line arguments
    args = tyro.cli(Args)
    print(args)
    
    # Check Argument Inputs
    assert args.num_envs > 0
    assert args.total_timesteps > 0
    assert args.learning_rate > 0
    assert args.buffer_size > 0
    assert args.gamma > 0
    assert args.tau > 0
    assert args.batch_size > 0
    assert args.start_e > 0
    assert args.buffer_type in ["ER", "PER", "HER"]
    assert args.model_type in ["DQN"]

    assert args.max_duration > 0

    if args.buffer_type == "HER":
        raise NotImplementedError("HER is not implemented yet")

    # Use wandb to log training runs
    if args.track:
        wandb.init(
            # set the wandb project where this run will be logged
            project="rl-crash-course",
            
            # track hyperparameters and run metadata
            config={
            "learning_rate": args.learning_rate,
            "architecture": args.model_type,
            "max_duration": args.max_duration,
            "dataset": "Highway-Env",
            "max_steps": args.total_timesteps,
            "collision_reward": args.crash_reward,
            "ttc_x_reward": args.ttc_x_reward,
            "ttc_y_reward": args.ttc_y_reward,
            "BATCH_SIZE": args.batch_size,
            "GAMMA": args.gamma,
            "EPS_START": args.start_e,
            "EPS_END": args.end_e,
            "EPS_DECAY": args.decay_e,
            "TAU": args.tau,
            "ReplayBuffer": args.buffer_type
            }
        )

    # BATCH_SIZE is the number of transitions sampled from the replay buffer
    # GAMMA is the discount factor as mentioned in the previous section
    # EPS_START is the starting value of epsilon
    # EPS_END is the final value of epsilon
    # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    # TAU is the update rate of the target network
    # LR is the learning rate of the ``AdamW`` optimizer
    BATCH_SIZE = args.batch_size
    GAMMA = args.gamma
    EPS_START = args.start_e
    EPS_END = args.end_e
    EPS_DECAY = args.decay_e
    TAU = args.tau
    LR = args.learning_rate


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
        "duration" : args.max_duration,
        "initial_lane_id" : None,
        "policy_frequency": 1,
        # Reset Configs
        'spawn_configs' : ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center'],
        'mean_distance' : 20,
        'initial_speed' : 20,
        'mean_delta_v' : 0,
        # Crash Configs
        'ttc_x_reward' : args.ttc_x_reward,
        'ttc_y_reward' : args.ttc_y_reward,
        'crash_reward' : args.crash_reward,
        'tolerance' : 1e-3
    }

    # Create Vector Env with Adversarial Rewards
    env = make_vector_env(env_config, num_envs = args.num_envs, adversarial = args.adversarial)


    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    elif args.metal:
        device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
    else:
        device = torch.device("cpu")

    # Get number of actions from gym action space
    n_actions = env.single_action_space.n
    # Get the number of state observations
    state, info = env.reset()
    n_observations = len(state[0].flatten())

    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(args.buffer_size)


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
                return policy_net(state).max(1)[1].view(args.num_envs, 1)
        else:
            # print('Action Space: {}'.format(torch.tensor(np.array([[env.action_space.sample()]]), device=device, dtype=torch.long).shape))
            return torch.tensor(np.array([[env.action_space.sample()]]), device=device, dtype=torch.long)

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
    episode_rewards = np.zeros(args.num_envs)
    duration = np.zeros(args.num_envs)
    ttc_x = 0
    ttc_y = 0
    state, info = env.reset()
    state = torch.tensor(state.reshape(args.num_envs,n_observations), dtype=torch.float32, device=device)

    max_steps = 1e7
    t_step = 0
    with tqdm(total=max_steps) as pbar:
        while(True):
            action = torch.squeeze(select_action(state))
            observation, reward, terminated, truncated, info = env.step(action.cpu().numpy())

            reward = torch.tensor(reward, device=device)
            done = terminated | truncated

            next_state = torch.tensor(observation.reshape(args.num_envs,n_observations), dtype=torch.float32, device=device)
            for worker in range(args.num_envs):
                if terminated[worker]:
                    memory.push(state[worker].view(1,n_observations),action[worker].view(1,1),None,reward[worker].view(1,1))
                else:
                    memory.push(state[worker].view(1,n_observations),action[worker].view(1,1),next_state[worker].view(1,n_observations),reward[worker].view(1,1))

            state = next_state

            episode_rewards = episode_rewards + reward.cpu().numpy()
            duration = duration + np.ones(args.num_envs)

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
            target_net.load_state_dict(target_net_state_dict)

            for worker in range(args.num_envs):
                if done[worker]:
                    num_crashes.append(float(info['final_info'][worker]['crashed']))
                    i_episode += 1
                    if args.track:
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

        # if save_model:
        #     torch.save(policy_net.state_dict(), wandb.run.dir + "/model-{}.pt".format(len(episode_rewards)))

    print('Complete')
    env.close()