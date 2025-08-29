import gymnasium as gym
from rl_agents.agents.tree_search.mcts import MCTSAgent


def test_eleurent_mcts():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    agent = MCTSAgent(env, config=dict(budget=400, temperature=200, max_depth=10))

    state, info = env.reset()
    done = truncated = False
    steps = 0
    while True:
        env.reset()
        if steps > 10000:
            break
        while not done and not truncated:
            action = agent.act(state)
            assert action is not None

            next_state, reward, done, truncated, info = env.step(action)
            env.render()
            steps += 1

    assert steps == env._max_episode_steps


if __name__ == "__main__":
    test_eleurent_mcts()
