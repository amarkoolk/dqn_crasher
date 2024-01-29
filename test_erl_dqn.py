import gymnasium as gym
from rl_agents.agents.deep_q_network.pytorch import DQNAgent

def test_eleurent_dqn():

    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    agent = DQNAgent(env, config = None)

    state, info = env.reset()
    n = 2 * agent.config['batch_size']
    for _ in range(10000):
        action = agent.act(state)
        assert action is not None

        next_state, reward, done, truncated, info = env.step(action)
        agent.record(state, action, reward, next_state, done, info)
        env.render()

        print(f'step: {_}, reward: {reward}, done: {done}, truncated: {truncated}')
        if done or truncated:
            state, info = env.reset()
        else:
            state = next_state

    assert (len(agent.memory) == n or
            len(agent.memory) == agent.config['memory_capacity'])

if __name__ == "__main__":
    test_eleurent_dqn()