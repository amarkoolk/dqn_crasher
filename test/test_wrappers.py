import gymnasium as gym
from crash_wrappers import CrashResetWrapper, CrashRewardWrapper


def test_wrappers():
    env = gym.make("highway-v0", render_mode="rgb_array")
    env = CrashResetWrapper(env)

    env = CrashRewardWrapper(env)

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
    env.configure(env_config)
    env.reset()
    for _ in range(1000):
        env.render()
        obs, reward, terminated, truncated, info = env.step(
            env.action_space.sample()
        )  # take a random action
        done = terminated | truncated
        if done:
            env.reset()


if __name__ == "__main__":
    test_wrappers()
