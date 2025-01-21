import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
from scenarios import Slowdown, SlowdownSameLane, SpeedUp, CutIn
from config import load_config
import torch
import tyro
from tqdm import tqdm
from arguments import Args
from dqn_agent import DQN_Agent
from create_env import make_vector_env
import json

if __name__ == "__main__":

  args = tyro.cli(Args)
  if args.cuda:
      device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
  elif args.metal:
      device = torch.device("mps" if torch.backends.mps.is_available()  else "cpu")
  else:
      device = torch.device("cpu")

  scenarios = [Slowdown(), SlowdownSameLane(), SpeedUp(), CutIn()]
  model_config = ["AdjustableK_C5_100000", "AdjustableK_C5_100000_Asymmetric", "UniformSampling_C5_100000", "PrioritizedSampling_C5_100000_FixedUpdate", "K0_C5_100000", "SequentialTraining_C5"]
  ma_env_cfg = load_config("env_configs/ma_cut_in.yaml")
  num_cycles = 5
  num_test = 100
  crashes = {}
  for cfg in model_config:
      crashes[cfg] = {}
      for scenario in scenarios:
          crashes[cfg][scenario.__class__.__name__] = []

  for cfg in model_config:
      
      model_dir = f"trained_models/{cfg}/"
      for scenario in tqdm(scenarios):
          ma_env_cfg['spawn_configs'] = scenario.spawn_configs
          env : AsyncVectorEnv = make_vector_env(ma_env_cfg, args.num_envs, record_video=False)

          ego_model0 = "E0_MOBIL.pth"
          ego_models = [model_dir + f"E{i}_V{i}_TrainEgo_True.pth" for i in range(1,num_cycles)]
          ego_models = [ego_model0] + ego_models
          ego_crashes = [[] for _ in range(len(ego_models))]

          for ego_version in tqdm(range(0,num_cycles), leave=False):
              # ego_version = 0
              ego_agent = DQN_Agent(env, args, device, save_trajectories=False, multi_agent=True)
              ego_agent.load_model(path = ego_models[ego_version])

              for _ in tqdm(range(num_test), leave=False):
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
                          ego_crashes[ego_version].append(info['final_info'][0]['crashed'])

                  scenario.set_state(ego_x, ego_y, npc_x, npc_y)
                  # env.call('render')
          
          crashes[cfg][scenario.__class__.__name__] = ego_crashes

          env.close()

  for cfg in crashes.keys():
      print(f"Model: {cfg}")
      for key in crashes[cfg].keys():
          print(f"Scenario: {key} ")
          for i in range(len(ego_models)):
              print(f"Cycle {i+1}: Mean: {np.mean(crashes[cfg][key][i])}, Stdev: {np.std(crashes[cfg][key][i])}")
  
  json.dump(crashes, open("crashes.json", "w"))