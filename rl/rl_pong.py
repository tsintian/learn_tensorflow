import gym
import time

env = gym.make('Pong-v0')
env.reset()

env.render()
env.step(env.action_space.sample())
env.close()