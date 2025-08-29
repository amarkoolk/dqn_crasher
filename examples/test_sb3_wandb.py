import gymnasium as gym
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback

import wandb

config = {
    "total_timesteps": int(10e6),
    "num_envs": 40,
    "env_name": "highway-v0",
}
run = wandb.init(
    project="rl_crash_course",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

env_config = {
    "observation": {"type": "Kinematics", "normalize": False},
    "action": {"type": "DiscreteMetaAction", "target_speeds": list(range(15, 35))},
    "lanes_count": 2,
    "vehicles_count": 1,
    "duration": 100,
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
    "tolerance": 1e-3,
}


def custom_highway_env(env_id, env_config):
    def _init():
        env = gym.make(env_id, render_mode="rgb_array")
        env.configure(env_config)  # Assuming your env has a configure method
        return env

    return _init


env = make_vec_env(
    custom_highway_env(config["env_name"], env_config),
    n_envs=config["num_envs"],
    seed=0,
)
# env = VecFrameStack(env, n_stack=4)
# env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=2000)  # record videos
model = PPO(
    "MlpPolicy",
    env,
    n_steps=128,
    n_epochs=4,
    learning_rate=lambda progression: 2.5e-4 * progression,
    ent_coef=0.01,
    clip_range=0.1,
    batch_size=256,
    verbose=1,
    tensorboard_log=f"runs",
)

# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[128, 128]),
#               learning_rate=1e-4,
#               buffer_size=10000,
#               learning_starts=200,
#               batch_size=128,
#               gamma=0.99,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               tensorboard_log="highway_dqn/")

model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
    ),
)
