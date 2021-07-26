import gym
import sys
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from dqn_agent import Agent

sys.path.append("..")

# checkout the environment
import Blackjack
env = gym.make("Blackjack-v1")
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)
environment_configurations = [{"one_card_dealer": True},
                              {},
                              {"card_values": [2] * 52},
                              {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
                                               4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
                                               2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
                                               5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}]
# environment_configurations = [
#                               {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
#                                                4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
#                                                2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
#                                                5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}]

##############################################################################################
# Helper function to convert the state to a one-hot encoding array for both player and dealer hand
# The final state will be 104, (52 + 52) dimensions
def transform_state(state):
    t1 = torch.tensor(state[0].astype(np.uint8)).int()
    t2 = torch.zeros(52).int()
    t2[state[1]] = 1
    new_obs = torch.cat((t1, t2), 0)
    return new_obs

class DQN:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, n_episodes=30000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        """Deep Q-Learning.

        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """
        scores = []                        # Holds average scores for final plot
        scores_window = deque(maxlen=100)  # maintain average score of last 100 episodes
        eps = eps_start
        for i_episode in range(1, n_episodes+1):
            state = transform_state(self.env.reset())
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.env.step(action)
                next_state = transform_state(next_state)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break

            scores_window.append(score)
            scores.append(np.mean(scores_window))
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
            if i_episode%1000==0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                sys.stdout.flush()

        return scores

    def test(self, episodes):
        scores_window = []
        for i_episode in range(episodes):
            state = transform_state(self.env.reset())
            score = 0
            done = False
            while not done:
                action = self.agent.eval(state)
                next_state, reward, done, _ = self.env.step(action)
                state = transform_state(next_state)
                score += reward

            scores_window.append(score)

        return np.mean(scores_window)


for config in environment_configurations:
    print("Running {}".format(config))
    env = gym.make("Blackjack-v1", env_config=config)
    agent = Agent(state_size=104, action_size=2, seed=0)
    dqn = DQN(env, agent)
    scores = dqn.train()

    score = dqn.test(1000)
    print("Test score for 1000 games: {:.4f}".format(score))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
