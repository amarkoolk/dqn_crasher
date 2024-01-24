import gymnasium as gym
from crash_wrappers import CrashResetWrapper, CrashRewardWrapper
from gymnasium.wrappers.record_video import RecordVideo
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

def test_vector_env():
    reset_config = {'spawn_configs' : ['behind_left', 'behind_right', 'behind_center', 'adjacent_left', 'adjacent_right', 'forward_left', 'forward_right', 'forward_center'],
                    'mean_distance' : 20,
                    'initial_speed' : 20,
                    'mean_delta_v' : 0}

    crash_config = {'ttc_x_reward' : 4,
                    'ttc_y_reward' : 1,
                    'crash_reward' : 400,
                    'tolerance' : 1e-3}


    # Create Normal Vector Env
    envs = gym.make_vec('highway-v0', num_envs=4,  vectorization_mode="async", render_mode='rgb_array')
    envs.reset()

    print(f'Observation Space: {envs.observation_space}')
    print(f'Action Space: {envs.action_space}')
    print(f'Single Observation Space: {envs.single_observation_space}')
    print(f'Single Action Space: {envs.single_action_space}')

if __name__ == '__main__':
    test_vector_env()