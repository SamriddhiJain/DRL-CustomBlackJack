import gym
import sys
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from agent import Agent

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
#     {"card_values": [3,  1,  3,  9,  6,  0,  7, -2,  2,  6,  8,  1,  3,
#                      4, -1,  4,  3,  9, -1,  4,  0,  4,  7, -2, -1,  5,
#                      2,  6, -3, -1,  2,  2, -1,  7,  1,  0,  7,  8,  4,
#                      5,  3, -1,  0,  3, -1,  3,  0,  6, -2,  4, -3,  4]}]

##############################################################################################
# Helper function to convert the state to a one-hot encoding array for both player and dealer hand
# The final state will be 104, (52 + 52) dimensions
def transform_state(state):
    t1 = torch.tensor(state[0].astype(np.uint8)).int()
    t2 = torch.zeros(52).int()
    t2[state[1]] = 1
    new_obs = torch.cat((t1, t2), 0)
    return new_obs


class PPO:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def train(self, n_episodes=30000, update_after=2000, log_after=2000):
        scores = []                        # Holds average scores for final plot
        policy_loss = []
        value_loss = []
        scores_window = deque(maxlen=100)  # maintain average score of last 100 episodes

        for i_episode in range(1, n_episodes+1):
            state = transform_state(self.env.reset())
            score = 0
            done = False
            while not done:
                action, value = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = transform_state(next_state)
                self.agent.step(state, action, reward, next_state, done, value)
                state = next_state
                score += reward

            scores_window.append(score)
            scores.append(np.mean(scores_window))

            if i_episode%update_after == 0:
                p, v = self.agent.update()
                policy_loss.append(p)
                value_loss.append(v)

            if i_episode%log_after == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                sys.stdout.flush()

        return scores, policy_loss, value_loss

    def test(self, episodes):
        scores_window = []
        for i_episode in range(episodes):
            state = transform_state(self.env.reset())
            score = 0
            done = False
            while not done:
                action, _ = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                state = transform_state(next_state)
                score += reward

            scores_window.append(score)

        return np.mean(scores_window)


for config in environment_configurations:
    print("Running {}".format(config))
    env = gym.make("Blackjack-v1", env_config=config)
    agent = Agent(state_size=104, action_size=2, seed=0)
    ppo = PPO(env, agent)
    scores, policy_loss, value_loss = ppo.train(n_episodes=100000, update_after=2000)

    score = ppo.test(1000)
    print("Test score for 1000 games: {:.4f}".format(score))

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(policy_loss)), policy_loss)
    plt.ylabel('Loss')
    plt.xlabel('Update #')
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(value_loss)), value_loss)
    plt.ylabel('Loss')
    plt.xlabel('Update #')
    plt.show()
