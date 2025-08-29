import gymnasium as gym
import matplotlib.pyplot as plt
from crash_wrappers import CrashRewardWrapper
from gymnasium.wrappers.record_video import RecordVideo
from ray.rllib.algorithms.dqn.dqn import DQNConfig
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
    DQNConfig()
    .framework(
        "torch",
        torch_compile_worker=True,
        torch_compile_worker_dynamo_backend="ipex",
        torch_compile_learner_dynamo_mode="default",
    )
    .training(replay_buffer_config=replay_config)
    .resources(num_gpus=0)
    .rollouts(num_rollout_workers=1)
    .environment("highway-test")
)

algo = config.build()

for _ in range(5):
    print(algo.train())

algo.evaluate()
