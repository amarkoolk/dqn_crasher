import gymnasium as gym
from crash_wrappers import CrashResetWrapper, CrashRewardWrapper
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers.record_video import RecordVideo


def make_env(env_config, adversarial: bool = False):
    def _env_fn():
        env = gym.make("highway-v0", config=env_config, render_mode="rgb_array")
        if adversarial:
            env = CrashResetWrapper(env)
            env = CrashRewardWrapper(env)
        return env

    return _env_fn


def test_vector_env():
    env_config = {
        "observation": {"type": "Kinematics", "normalize": False},
        "action": {"type": "DiscreteMetaAction", "target_speeds": list(range(15, 35))},
        "lanes_count": 2,
        "vehicles_count": 1,
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
    }

    # Create Vector Env with Adversarial Rewards
    env_fns = [make_env(env_config) for _ in range(4)]

    envs = AsyncVectorEnv(env_fns)
    envs.reset()
    for _ in range(100):
        obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
        print(reward)


# If you don't include this, sometimes an error can be thrown when starting multiple processes
if __name__ == "__main__":
    test_vector_env()
