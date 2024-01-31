from create_env import make_env, make_vector_env
import gymnasium as gym

import wandb

def test_multi_agent():
    env_config = {
            "controlled_vehicles": 2,
            "observation": {
                "type": "MultiAgentObservation",
                "observation_config": {
                    "type": "Kinematics",
                }
            },
            "action": {
                "type": "MultiAgentAction",
                "action_config": {
                    "type": "DiscreteMetaAction",
                }
            },
            "lanes_count" : 2,
            "vehicles_count" : 0,
            "duration" : 40,
            "initial_lane_id" : None,
            "policy_frequency": 1,
            # Reset Configs
            'spawn_configs' : ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center'],
            'mean_distance' : 20,
            'initial_speed' : 20,
            'mean_delta_v' : 0,
            # Crash Configs
            'ttc_x_reward' : 4,
            'ttc_y_reward' : 1,
            'crash_reward' : 400,
            'tolerance' : 1e-3
    }
    
    config = {
        "total_timesteps": int(1e3),
        "num_envs": 4,
        "env_name": "highway-v0",
    }

    # run = wandb.init(
    #     project="rl_crash_course",
    #     config=config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     monitor_gym=False,  # auto-upload the videos of agents playing the game
    #     save_code=True,  # optional
    # )
    env = gym.make(config["env_name"],render_mode='rgb_array')
    env.configure(env_config)
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        # Dispatch the observations to the model to get the tuple of actions
        action = env.action_space.sample()
        # Execute the actions
        next_obs, reward, done, truncated, info = env.step(action)
        obs = next_obs
        env.render()

if __name__ == "__main__":
    test_multi_agent()