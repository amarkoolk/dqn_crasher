import gymnasium as gym
env = gym.make("highway-v0", config={
    "manual_control": True
})
env.reset()
done = False
while not done:
    env.step(env.action_space.sample())