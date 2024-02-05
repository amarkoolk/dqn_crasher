import gymnasium as gym
from crash_wrappers import CrashResetWrapper, CrashRewardWrapper
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

import multiprocessing

def make_env(env_config, adversarial : bool = False):
    def _env_fn():
        env = gym.make('crash-v0', config=env_config, render_mode='rgb_array')

        env = CrashResetWrapper(env)
        if adversarial:
            env = CrashRewardWrapper(env)
            env.configure({'adversarial' : True})
        return env
    return _env_fn

def make_vector_env(env_config, num_envs = 1, adversarial : bool = False):
    
    assert num_envs > 0

    env_fns = [make_env(env_config, adversarial) for _ in range(num_envs)]
    envs = AsyncVectorEnv(env_fns)
    return envs