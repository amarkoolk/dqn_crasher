import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
from scenarios import CutIn, Action
from config import load_config
import torch
import tyro
from tqdm import tqdm
from arguments import Args
from dqn_agent import DQN_Agent
from create_env import make_vector_env




# na_env_cfg = load_config("env_configs/cut_in.yaml")
# env = gym.make('crash-v0', config=na_env_cfg, render_mode='rgb_array')
# scenario = CutIn()

# for _ in range(10):
#   done = truncated = False
#   obs, info = env.reset()
#   scenario.reset(obs, info)
#   while not (done or truncated or scenario.end_frames > 2):
#     action = scenario.get_action()
#     obs, reward, done, truncated, info = env.step(action)
#     if 'int_frames' in info.keys():
#       ego_x = info['int_frames'][-1,1]
#       npc_x = info['int_frames'][-1,6]
#       npc_y = info['int_frames'][-1,7]
#     else:
#       ego_x = obs[0,1]
#       npc_x = obs[1,1]
#       npc_y = obs[1,2]

#     scenario.set_state(ego_x, npc_x, npc_y)
#     env.render()

if __name__ == "__main__":

  args = tyro.cli(Args)
  if args.cuda:
      device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
  elif args.metal:
      device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
  else:
      device = torch.device("cpu")

  ma_env_cfg = load_config("env_configs/ma_cut_in.yaml")
  env : AsyncVectorEnv = make_vector_env(ma_env_cfg, args.num_envs, record_video=False)
  scenario = CutIn()

  ego_model0 = "E0_MOBIL.pth"
  ego_models = [f"E{i}_V{i}_TrainEgo_True.pth" for i in range(1,6)]
  ego_models = [ego_model0] + ego_models
  ego_crashes = [0,0,0,0,0,0]
  for ego_version in range(0,6):
    ego_version = 0
    ego_agent = DQN_Agent(env, args, device, save_trajectories=False, multi_agent=True)
    ego_agent.load_model(path = ego_models[ego_version])

    while True:
      done = truncated = False
      obs, info = env.reset()
      ego_state = torch.tensor(obs[0].reshape(args.num_envs,ego_agent.n_observations), dtype=torch.float32, device=device)
      scenario.reset(obs, info)
      while not (done or truncated):# or scenario.end_frames > 15):
        action = scenario.get_action()
        npc_action = torch.squeeze(torch.tensor([action])).view(1,1).cpu().numpy()
        ego_action = torch.squeeze(ego_agent.predict(ego_state)).view(1,1).cpu().numpy()
        obs, reward, terminated, truncated, info = env.step((ego_action,npc_action))
        done = terminated or truncated

        ego_x = obs[0][0,0,1]
        ego_y = obs[0][0,0,2]
        npc_x = obs[1][0,0,1]
        npc_y = obs[1][0,0,2]
        
        if done:
           if info['final_info'][0]['crashed']:
              ego_crashes[ego_version] += 1

        scenario.set_state(ego_x, ego_y, npc_x, npc_y)
        env.call('render')

  env.close()

  for i in range(0,6):
    print(f"Ego {i} crashes: {ego_crashes[i]/1000}")