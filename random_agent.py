import gym
import Blackjack
import numpy as np

"""
Creates an environment of the Blackjack game and plays one game in it by randomly choosing the actions.
"""
env = gym.make('Blackjack-v1')

state = env.reset()
terminal = False
while not terminal:
    action = np.random.randint(2)
    state, reward, terminal, info = env.step(action)
    print(info)
