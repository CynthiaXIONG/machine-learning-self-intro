import gym

env = gym.make('CartPole-v0')
env.reset()

print(env.action_space)
print(env.observation_space)

if True:
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(1)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break