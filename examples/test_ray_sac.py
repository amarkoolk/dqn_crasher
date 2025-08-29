import gymnasium as gym
import matplotlib.pyplot as plt
from crash_wrappers import CrashRewardWrapper
from gymnasium.wrappers.record_video import RecordVideo
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.tune.registry import register_env
from tqdm import tqdm


def env_creator(env_config):
    return gym.make("highway-v0", render_mode="rgb_array")


register_env("highway-test", env_creator=env_creator)

replay_config = {
    "type": "MultiAgentPrioritizedReplayBuffer",
    "capacity": 20000,
    "prioritized_replay_alpha": 0.5,
    "prioritized_replay_beta": 0.5,
    "prioritized_replay_eps": 3e-6,
}

config = (
    SACConfig()
    .framework(
        "torch",
        torch_compile_worker=True,
        torch_compile_worker_dynamo_backend="ipex",
        torch_compile_learner_dynamo_mode="default",
    )
    .training(gamma=0.9, lr=0.01, train_batch_size=32)
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=12)
    .environment("highway-test")
)

algo = config.build()

for _ in range(100):
    algo.train()
    print(f"iteration {_}")

# View Algo Prediction
env = gym.make("highway-v0", render_mode="rgb_array")
while True:
    done = truncated = False
    obs, info = env.reset()
    while not (done or truncated):
        action = algo.compute_single_action(obs, explore=False)
        obs, reward, done, truncated, info = env.step(action)
        env.render()
