import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecVideoRecorder
from wandb.integration.sb3 import WandbCallback


config = {
    "total_timesteps": int(10e6),
    "num_envs": 8,
    "env_name": "highway-fast-v0",
}
run = wandb.init(
    project="rl_crash_course",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)


env = make_vec_env(config["env_name"], n_envs=config["num_envs"], seed=0)
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
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
    ),
)
