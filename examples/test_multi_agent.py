from create_env import make_env, make_vector_env
from crash_wrappers import CrashResetWrapper, CrashRewardWrapper
import gymnasium as gym

import wandb


def test_multi_agent():
    env_config = {
        "controlled_vehicles": 2,
        "observation": {
            "type": "MultiAgentObservation",
            "observation_config": {
                "type": "Kinematics",
            },
        },
        "action": {
            "type": "MultiAgentAction",
            "action_config": {
                "type": "DiscreteMetaAction",
            },
        },
        "lanes_count": 2,
        "vehicles_count": 0,
        "duration": 10,
        "initial_lane_id": None,
        "policy_frequency": 1,
        # Reset Configs
        "spawn_configs": [
            "behind_left",
            "behind_right",
            "behind_center",
            "adjacent_left",
            "adjacent_right",
            "forward_left",
            "forward_right",
            "forward_center",
        ],
        "mean_distance": 20,
        "initial_speed": 20,
        "mean_delta_v": 0,
        # Crash Configs
        "ttc_x_reward": 4,
        "ttc_y_reward": 1,
        "crash_reward": 400,
        "tolerance": 1e-3,
        "use_mobil": True,
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
    env = gym.make("crash-v0", config=env_config, render_mode="rgb_array")
    # env = CrashResetWrapper(env)
    obs, info = env.reset()
    done = truncated = False
    step = 0
    while True:
        done = truncated = False
        if step % 2 == 0:
            env.configure({"use_mobil": True})
        else:
            env.configure({"use_mobil": False})
        obs, info = env.reset()
        while not (done or truncated):
            # Dispatch the observations to the model to get the tuple of actions
            action = env.action_space.sample()
            # print(len(obs))
            print(len(action))
            # Execute the actions
            next_obs, reward, done, truncated, info = env.step(action)
            obs = next_obs
            env.render()
        step += 1


if __name__ == "__main__":
    test_multi_agent()
