import gymnasium as gym
from gymnasium.wrappers.rendering import RecordVideo
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv


def make_env(env_config, record_video = False, record_dir = '', record_every = 1, env_fn_count = 0):
    def _env_fn():
        env = gym.make('crash-v0', config=env_config, render_mode='rgb_array')
        if record_video:
            trigger = lambda t: t % record_every == 0
            env = RecordVideo(env, video_folder = record_dir+f'_{env_fn_count}', episode_trigger = trigger, disable_logger=True)
            env.unwrapped.set_record_video_wrapper(env)
        return env
    return _env_fn

def make_vector_env(env_config, num_envs = 1, record_video = False, record_dir = '.', record_every = 1):
    
    assert num_envs > 0

    env_fns = [make_env(env_config, record_video, record_dir, record_every, env_fn_count=i) for i in range(num_envs)]
    envs = AsyncVectorEnv(env_fns)
    return envs