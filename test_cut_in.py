import gymnasium as gym

from scenarios import CutIn
from config import load_config




na_env_cfg = load_config("env_configs/cut_in.yaml")
env = gym.make('crash-v0', config=na_env_cfg, render_mode='rgb_array')
scenario = CutIn()

while True:
  done = truncated = False
  obs, info = env.reset()
  scenario.reset(obs, info)
  while not (done or truncated or scenario.end_frames > 2):
    action = scenario.get_action()
    obs, reward, done, truncated, info = env.step(action)
    if 'int_frames' in info.keys():
      ego_x = info['int_frames'][-1,1]
      npc_x = info['int_frames'][-1,6]
      npc_y = info['int_frames'][-1,7]
    else:
      ego_x = obs[0,1]
      npc_x = obs[1,1]
      npc_y = obs[1,2]

    scenario.set_state(ego_x, npc_x, npc_y)


    env.render()