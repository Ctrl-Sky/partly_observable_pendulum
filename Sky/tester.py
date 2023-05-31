import gym

env = gym.make('MountainCar-v0', render_mode="human")
env.reset()

done = False
while not done:
    env.render()
    action = env.action_space.sample()  # Random action selection
    observation, reward, done, info = env.step(action)
    done = False
    if reward == -200:
        done = True

env.close()